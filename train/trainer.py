"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import os.path
import sys

sys.path.append('../moses')
from itertools import chain
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from utils import check_novelty, sample, canonic_smiles
from moses.utils import get_mol
import re
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') +
                  F.kl_div(m.log(), q, reduction='batchmean'))


def wasserstein_distance(p, q):
    return torch.mean(p - q)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, model2, train_dataset, test_dataset, config, stoi, itos, discriminator):
        self.model = model
        self.model2 = model2
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.discriminator = discriminator
        # take over whatever gpus are on the system
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos
        self.cur_iteration = 0
        self.cycle = 5000
        self.cycle_ratio = 0.5

        if torch.cuda.is_available():
            self.device = config.device
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)
            self.model2 = self.model2.to(self.device)
            self.discriminator = self.discriminator.to(self.device)

    def save_checkpoint(self, type='gen'):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        path = self.config.ckpt_path + type + '.pt'
        torch.save(raw_model.state_dict(), path)

    def train(self, wandb):
        model, config = self.model, self.config
        model2 = self.model2
        loss_weight = config.loss_weight
        discriminator = self.discriminator
        temperature = config.t
        # raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = optim.AdamW(chain(model.parameters(), model2.parameters()),
                                lr=config.learning_rate,
                                betas=(0.9, 0.98), eps=1e-9)
        d_optimizer = optim.AdamW(discriminator.parameters(),
                                  lr=config.learning_rate * 0.5,
                                  betas=(0.9, 0.98), eps=1e-9)
        # optimizer = model2.configure_optimizers(config)
        scaler = GradScaler()
        scaler2 = GradScaler()

        # [' ', '#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'B', 'C', 'F', 'H', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']
        # ['#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'Br', 'C', 'Cl', 'F', 'N', 'O', 'S', '[H]', '[nH]', 'c', 'n', 'o', 's']

        def run_epoch(split):
            step = 0
            is_train = split == 'train'
            model.train(is_train)
            self.discriminator.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            losses2, losses3 = [], []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, p, scaffold) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                p = p.to(self.device)
                scaffold = scaffold.to(self.device)

                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        step += 1
                        logits, feature, loss, _ = model(x, y, p, scaffold)
                        logits2, feature2, loss2, _ = model2(x, y, p, scaffold)
                        # logits3, feature3, lossx, _ = model2.forward_decoder(x, y, p, feature)
                        log_prob_dist1 = F.log_softmax(logits / temperature, dim=-1)
                        prob_dist1 = F.softmax(logits / temperature, dim=-1)
                        prob_dist2 = F.softmax(logits2 / temperature, dim=-1)
                        log_prob_dist2 = F.log_softmax(logits2 / temperature, dim=-1)
                        # prob_feature1 = F.softmax(feature/temperature, dim=-1)
                        # log_prob_feature1 = F.log_softmax(feature/temperature, dim=-1)
                        # prob_feature2 = F.softmax(feature2/temperature, dim=-1)
                        r = self.cur_iteration % self.cycle
                        beta = r / self.cycle * self.cycle_ratio if r <= self.cycle_ratio * self.cycle else 1
                        self.cur_iteration = r + 1
                        loss3 = beta * F.kl_div(log_prob_dist1, prob_dist2, reduction='batchmean') * (temperature ** 2)
                        # loss3 = F.kl_div(log_prob_dist2, prob_dist1, reduction='batchmean')*(temperature**2)
                        # loss3 = F.kl_div(log_prob_feature1, prob_feature2, reduction='batchmean')*(temperature**2)
                        # loss3 = js_divergence(prob_feature1,prob_feature2)
                        # loss3 = wasserstein_distance(prob_feature1,prob_feature2)
                        # loss3 = js_divergence(prob_dist1,prob_dist2)
                        # loss3 = wasserstein_distance(prob_dist1,prob_dist2)
                        # loss3 = torch.tensor(-1)
                        losses2.append(loss2.item())
                        losses3.append(loss3.item())
                        loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())
                        # loss = torch.tensor(0)
                        # loss2 = torch.tensor(0)
                        # loss3 = torch.tensor(0)
                        final_loss = loss_weight[0] * loss + loss_weight[1] * loss2 + loss_weight[2] * loss3
                        # final_loss = loss

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    model2.zero_grad()
                    discriminator.zero_grad()
                    loss_dis = torch.tensor(0)
                    loss_gen = torch.tensor(0)
                    if epoch >= config.adst:
                        loss_gen = 1 * discriminator(feature, 1)
                    final_loss += loss_gen
                    # final_loss.backward()
                    # loss.backward()
                    # loss2.backward()
                    # loss3.backward()
                    scaler.scale(final_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    if step % 5 == 0 and epoch >= config.adst:
                        discriminator.zero_grad()
                        d_optimizer.zero_grad()
                        loss_dis = (discriminator(feature.detach(), 0) + discriminator(feature2.detach(),
                                                                                       1)) / 2
                        #     print(loss_dis.item())
                        scaler2.scale(loss_dis).backward()
                        scaler2.step(d_optimizer)
                        scaler2.update()
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    # loss4: {loss4.item(): .5f}.
                    # report progress
                    wandb.log({'step_train_loss': loss, 'train_step': it + epoch * len(loader), 'learning_rate': lr})
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. loss2 {loss2.item(): .5f}. loss3 {loss3.item(): .5f}. "
                        f"loss_gen : {loss_gen.item(): .5f}. loss_dis : {loss_dis.item(): .5f} {lr:e}")

            if is_train:
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                test_loss2 = float(np.mean(losses2))
                test_loss3 = float(np.mean(losses3))
                print("test loss: %f loss2: %f loss3 : %f", test_loss, test_loss2, test_loss3)
                logger.info("test loss: %f,%f,%f", test_loss, test_loss2, test_loss3)
                return test_loss, test_loss2, test_loss3

        best_loss = float('inf')
        best_loss2 = float('inf')
        save_model2 = False
        self.tokens = 0  # counter used for learning rate decay
        molecules = []

        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss, test_loss2, test_loss3 = run_epoch('test')

            wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss - 0.001
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()
            if save_model2:
                good_model2 = self.test_dataset is None or test_loss2 < best_loss2 - 0.001
                if self.config.ckpt_path is not None and good_model2:
                    best_loss2 = test_loss2
                    print(f'Saving at epoch {epoch + 1}')
                    self.save_checkpoint(type='scaffold')
            if self.config.generate:
                pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
                regex = re.compile(pattern)
                context = "C"
                for i in range(2):
                    x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[
                        None, ...].repeat(512, 1).to('cuda')
                    p = None
                    sca = None
                    y = sample(model, x, self.config.block_size, temperature=0.8, sample=True, top_k=10, prop=p,
                               scaffold=sca)
                    for gen_mol in y:
                        completion = ''.join([self.itos[int(i)] for i in gen_mol])
                        completion = completion.replace('<', '')
                        mol = get_mol(completion)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)
                            molecules.append((mol, smiles, epoch))

        if self.config.generate:
            df = pd.DataFrame(molecules, columns=['molecule', 'smiles', 'epoch'])
            return df

        return None
