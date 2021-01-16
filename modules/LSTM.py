# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/16 上午10:46
 @FileName: LSTM.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size+1, embedding_dim=n_embedding, padding_idx=0)
        self.encoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden // 2, bidirectional=True,
                               num_layers=n_layer,
                               batch_first=True)
        self.prediction = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Tanh(),
            nn.Linear(n_hidden // 2, 1),
        )
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden

    def forward(self, inputs):
        [seq, label] = inputs
        representations, _ = self.encoder(self.word_embedding(seq))
        mask_idx = torch.eq(seq, self.vocab_size)
        hidden = representations.masked_select(mask_idx.unsqueeze(2).expand_as(representations)).view(
            -1, self.n_hidden)
        answer_logit = self.prediction(hidden).squeeze(1)
        if label is None:
            return torch.sigmoid(answer_logit)
        return F.binary_cross_entropy_with_logits(answer_logit, label)
