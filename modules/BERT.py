# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午5:09
 @FileName: BERT.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import copy
import warnings
from torch.nn import functional as F
import apex
import torch
import torch.nn as nn
from apex.contrib.multihead_attn import EncdecMultiheadAttn
from apex.mlp import MLP
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore")
layer_norm = apex.normalization.FusedLayerNorm
gradient_checkpoint = False


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = EncdecMultiheadAttn(d_model, nhead, dropout=dropout, impl='fast')
        self.feed_forward = MLP([d_model, dim_feedforward, d_model])
        self.d_model = d_model
        self.norm1 = layer_norm(d_model)
        self.norm2 = layer_norm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=None,
                              key_padding_mask=None, is_training=self.training)[0]
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)

        src2 = self.feed_forward(src2.view(-1, self.d_model)).view(src.size())
        src = src + self.dropout2(src2)

        return src


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None):
        output = src

        for i, mod in enumerate(self.layers):
            if gradient_checkpoint:
                output = checkpoint(mod, output)
            else:
                output = mod(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(n_hidden, n_head, n_hidden * 4, dropout)
        self.encoder = TransformerEncoder(encoder_layer, n_layer, None)
        self.output_ln = layer_norm(n_hidden)

    def forward(self, representations, mask=None):
        representations = representations.transpose(0, 1).contiguous()
        representations = self.encoder(representations, mask)
        return self.output_ln(representations.transpose(0, 1))


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, n_embedding, n_hidden, max_pos_size=1024):
        super().__init__()
        self.pos_size = max_pos_size
        self.n_embedding = n_embedding
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim=n_embedding, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_pos_size, embedding_dim=n_embedding)
        self.embedding_project = nn.Sequential(
            layer_norm(n_embedding),
            nn.Dropout(0.1),
            nn.Linear(n_embedding, n_hidden),
        )

    def forward(self, seq):
        word_embedding = self.word_embedding(seq)
        length = seq.size(1)
        pos = torch.arange(length).cuda()
        pos %= self.pos_size
        pos = pos.expand_as(seq)
        pos_embedding = self.pos_embedding(pos)
        return self.embedding_project(word_embedding + pos_embedding)


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer, n_head):
        super().__init__()
        pad_vocab_size = (2 + vocab_size // 8) * 8
        self.pad_vocab_size = pad_vocab_size
        self.embedding = Embedding(pad_vocab_size, n_embedding, n_hidden)
        self.n_hidden = n_hidden
        self.attention = SelfAttention(n_hidden, n_layer, n_head=n_head)

    def get_hidden_representations(self, seq):
        encoder_representations = self.embedding(seq)
        encoder_representations = self.attention(encoder_representations)
        return encoder_representations


class Bert(Transformer):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer, n_head):
        super().__init__(vocab_size, n_embedding, n_hidden, n_layer, n_head)
        self.prediction = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Tanh(),
            nn.Linear(n_hidden // 2, 1),
        )
        self.vocab_size = vocab_size
        self.n_embedding = n_embedding

    def forward(self, inputs):
        [seq, label] = inputs
        representations = self.get_hidden_representations(seq)
        mask_idx = torch.eq(seq, self.vocab_size)
        hidden = representations.masked_select(mask_idx.unsqueeze(2).expand_as(representations)).view(
            -1, self.n_hidden)
        answer_logit = self.prediction(hidden).squeeze(1)
        if label is None:
            return torch.sigmoid(answer_logit)
        return F.binary_cross_entropy_with_logits(answer_logit, label)
