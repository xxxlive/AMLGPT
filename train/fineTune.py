import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig, Transformer, Discriminator, LSTM_Discriminator
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default=0, help="number of properties to use for condition",
                        required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--l1', type=float, default=1.0,
                        help="weight for lce1", required=False)
    parser.add_argument('--l2', type=float, default=1.0,
                        help="weight for lce2", required=False)
    parser.add_argument('--l3', type=float, default=1.0,
                        help="weight for kldiv", required=False)
    parser.add_argument('--t', type=float, default=1.0,
                        help="weight for kldiv temperature", required=False)
    parser.add_argument('--adst', type=int, default=0,
                        help="start epoch for adversarial learning", required=False)
    parser.add_argument('--device', type=str, default='cuda',
                        help="weight for kldiv temperature", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)

    # model_weight (pretrained)
    parser.add_argument('--model_weight', type=str, help="path of pretrained GPT model weights", required=True)
    parser.add_argument('--model_weight2', type=str, help="path of pretrained Transformer model weights", required=True)

    args = parser.parse_args()

    set_seed(42)

    wandb.init(project="lig_gpt", name=args.run_name)

    data = pd.read_csv('datasets/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    # data = data.sample(frac = 0.1).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    if 'moses' in args.data_name:
        train_data = data[data['split'] == 'train'].reset_index(
            drop=True)  # 'split' instead of 'source' in moses
    else:
        train_data = data[data['source'] == 'train'].reset_index(
            drop=True)  # 'split' instead of 'source' in moses

    # train_data = train_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    if 'moses' in args.data_name:
        val_data = data[data['split'] == 'test'].reset_index(
            drop=True)  # test for Moses. val for guacamol
    else:
        val_data = data[data['source'] == 'val'].reset_index(
            drop=True)  # test for Moses. val for guacamol

    # val_data = val_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    # prop = train_data[['qed']]
    # vprop = val_data[['qed']]

    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    num_props = args.num_props

    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
            for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    if 'guacamol' in args.data_name or 'JAK' in args.data_name:
        max_len = 100

    print('Max len: ', max_len)

    lens = [len(regex.findall(i.strip()))
            for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = max(lens)

    if 'JAK' in args.data_name:
        scaffold_max_len = 100      # align with guacamol
    print('Scaffold max len: ', scaffold_max_len)

    smiles = [i + str('<') * (max_len - len(regex.findall(i.strip())))
              for i in smiles]
    vsmiles = [i + str('<') * (max_len - len(regex.findall(i.strip())))
               for i in vsmiles]

    scaffold = [i + str('<') * (scaffold_max_len -
                                len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<') * (scaffold_max_len -
                                 len(regex.findall(i.strip()))) for i in vscaffold]

    # whole_string = ' '.join(smiles + vsmiles + scaffold + vscaffold)
    # whole_string = sorted(list(set(regex.findall(whole_string))))
    # print(whole_string)

    # len = 97
    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B',
                    'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]',
                    '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]',
                    '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]',
                    '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]',
                    '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]',
                    '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p',
                    's', 'unk']        # , , .', '[K+]', '[Na+]

    # print(f"len(whole_string) = {len(whole_string)}")

    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0, scaffold=scaffold,
                                 scaffold_maxlen=scaffold_max_len)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0, scaffold=vscaffold,
                                 scaffold_maxlen=scaffold_max_len)

    print(f"vocab_size = {train_dataset.vocab_size}")
    print(f"max_len = {train_dataset.max_len}")
    # assert 1 == 2

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  # args.num_props,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold,
                      scaffold_maxlen=scaffold_max_len,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)
    model2 = Transformer(mconf)
    # model2 = GPT(mconf)
    discriminator = LSTM_Discriminator(mconf)

    ckpt_dir = '../cond_gpt/weights'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # load parameters
    model.load_state_dict(torch.load(args.model_weight))
    model2.load_state_dict(torch.load(args.model_weight2))

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=args.max_epochs * len(train_data) * max_len,
                          num_workers=8, ckpt_path=f'../cond_gpt/weights/{args.run_name}.pt',
                          block_size=train_dataset.max_len, generate=False, loss_weight=[args.l1, args.l2, args.l3],
                          device=args.device, t=args.t, adst=args.adst)
    trainer = Trainer(model, model2, train_dataset, valid_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos, discriminator)
    df = trainer.train(wandb)

    print(df.head(3))
    df.to_csv(f'{args.run_name}.csv', index=False)
