# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午5:21
 @FileName: OfficialBERT.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn import functional as F

from utils import load_file


class OfficialBERT(nn.Module):
    def __init__(self, indicator=80):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-chinese",
                                                 mirror='https://mirrors.bfsu.edu.cn/hugging-face-models/')
        self.n_hidden = self.encoder.config.hidden_size
        self.prediction = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden // 2),
            nn.Tanh(),
            nn.Linear(self.n_hidden // 2, 1),
        )
        self.node_indicator = indicator

    def forward(self, inputs):
        [seq, label] = inputs
        representations = self.encoder(seq)[0]
        mask_idx = torch.eq(seq, self.node_indicator)
        hidden = representations.masked_select(mask_idx.unsqueeze(2).expand_as(representations)).view(
            -1, self.n_hidden)
        answer_logit = self.prediction(hidden).squeeze(1)
        if label is None:
            return torch.sigmoid(answer_logit)
        return F.binary_cross_entropy_with_logits(answer_logit, label)
