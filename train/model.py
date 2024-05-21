"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_props)) + int(
            config.scaffold_maxlen)  # int(config.lstm_layers)    #  int(config.scaffold)
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                             .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head

    def forward(self, x, memory=None, layer_past=None):
        B, T, C = x.size()
        if memory is None:  # self attention
            memory = x
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(memory).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(memory).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, memory=None):
        y, attn = self.attn(self.ln1(x), memory)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  # max_len
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])  # decoder layer
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # projection

        self.block_size = config.block_size

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, prop=None, scaffold=None, memory=None):
        b, t = idx.size()
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=idx.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x, memory=memory)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, x, loss, attn_maps  # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, config.n_layer)
        self.decoder = GPT(config)

    def get_block_size(self):
        return self.decoder.get_block_size()

    def create_pad_mask(self, seq, pad_token_id):
        return (seq == pad_token_id).to(seq.device)

    def forward(self, smiles, target, _, frag):
        b, t = frag.size()
        token_embeddings = self.tok_emb(frag)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=frag.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        memory = self.encoder(x, mask=None, src_key_padding_mask=self.create_pad_mask(frag, 16))
        logits, x, loss, attn_maps = self.decoder(smiles, target, memory)
        return logits, x, loss, attn_maps

    def forward_encoder(self, sca):
        b, t = sca.size()
        token_embeddings = self.tok_emb(sca)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=sca.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        memory = self.encoder(x, mask=None, src_key_padding_mask=self.create_pad_mask(sca, 16))
        return memory

    def forward_decoder(self, smiles, target, _, memory):
        logits, x, loss, attn_maps = self.decoder(smiles, target, memory)
        return logits, x, loss, attn_maps


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.st_hidden = config.n_embd
        self.out_len = config.block_size - 1
        self.mlp1 = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, config.n_embd // 8),
            nn.Dropout(config.resid_pdrop),
        )
        self.sec_hi_len = (config.n_embd // 8) * self.out_len
        self.mlp2 = nn.Sequential(
            nn.Linear(self.sec_hi_len, self.sec_hi_len // 2),
            nn.GELU(),
            nn.Linear(self.sec_hi_len // 2, self.sec_hi_len // 8),
            nn.Dropout(config.resid_pdrop),
        )
        self.th_hi_len = self.sec_hi_len // 8
        self.mlp3 = nn.Sequential(
            nn.Linear(self.th_hi_len, self.th_hi_len // 2),
            nn.GELU(),
            nn.Linear(self.th_hi_len // 2, 1),
        )

    def forward(self, x, ans):
        x = self.mlp1(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = x.view(x.shape[0])
        ans = torch.ones(x.shape[0], device=x.device) * ans
        loss = F.binary_cross_entropy_with_logits(x, ans)
        return loss


class LSTM_Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.st_hidden = config.n_embd
        self.out_len = config.block_size - 1
        self.lstm = nn.LSTM(input_size=self.st_hidden, hidden_size=self.st_hidden, batch_first=True, dropout=0.1,
                            bidirectional=True, num_layers=2)
        self.out_size = 2 * self.st_hidden
        self.mlp1 = nn.Sequential(
            nn.Linear(self.out_size, self.out_size // 2),
            nn.GELU(),
            nn.Linear(self.out_size // 2, self.out_size // 4),
        )
        self.out1_size = self.out_size // 4
        self.mlp2 = nn.Sequential(
            nn.Linear(self.out1_size, self.out1_size // 2),
            nn.GELU(),
            nn.Linear(self.out1_size // 2, 1),
        )

    def forward(self, x, ans):
        out, hidden = self.lstm(x)
        out = out[:, -1, :]
        out = self.mlp1(out)
        out = self.mlp2(out)
        out = out.view(x.shape[0])
        ans = torch.ones(x.shape[0], device=x.device) * ans
        loss = F.binary_cross_entropy_with_logits(out, ans, reduction='mean')
        return loss
