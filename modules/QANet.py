# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午5:21
 @FileName: QANet.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        x = x.transpose(2, 1)
        return F.gelu(self.pointwise_conv(F.relu(self.depthwise_conv(x)))).transpose(2, 1)


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, n_embedding, n_hidden, max_pos_size=1024):
        super().__init__()
        self.pos_size = max_pos_size
        self.n_embedding = n_embedding
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim=n_embedding, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_pos_size, embedding_dim=n_embedding)
        self.embedding_project = nn.Sequential(
            nn.LayerNorm(n_embedding),
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


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, n_hidden, k, n_head):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(n_hidden, n_hidden, k) for _ in range(conv_num)])
        self.lns = nn.ModuleList([nn.LayerNorm(n_hidden) for _ in range(conv_num)])
        self.self_att = nn.MultiheadAttention(n_hidden, n_head, 0.1)
        self.fc = nn.Linear(n_hidden, n_hidden, bias=True)
        self.normb = nn.LayerNorm(n_hidden)
        self.norme = nn.LayerNorm(n_hidden)
        self.L = conv_num

    def forward(self, x, mask):
        res = x
        out = self.normb(x)
        for ln, conv in zip(self.lns, self.convs):
            out = conv(out)
            out = ln(out)
        out = out.transpose(0, 1)
        out = self.self_att(out, out, out, mask)[0]
        out = out.transpose(0, 1)
        out = out + res
        out = F.dropout(out, p=0.1, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=0.1, training=self.training)
        return out


class CQAttention(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.w = nn.Linear(n_hidden, n_hidden)

    def forward(self, Q, C):
        # C: b_size*l_doc*h
        # Q: b_size*l_question*h
        Q = F.gelu(self.w(Q))
        logits = torch.einsum('bdh,bqh->bdq', C, Q)
        scores = F.softmax(logits, -1)
        Q_hidden_4_D = torch.matmul(scores, Q)
        return Q_hidden_4_D + C


class QANet(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer, n_head):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.embedding = Embedding(vocab_size + 1, n_embedding, n_hidden)
        self.context_conv = DepthwiseSeparableConv(n_hidden, n_hidden, 5)
        self.question_conv = DepthwiseSeparableConv(n_hidden, n_hidden, 5)
        self.c_emb_enc = EncoderBlock(conv_num=n_layer, n_hidden=n_hidden, k=7, n_head=n_head)
        self.q_emb_enc = EncoderBlock(conv_num=n_layer, n_hidden=n_hidden, k=7, n_head=n_head)
        self.cq_att = CQAttention(n_hidden)
        self.aggregation = EncoderBlock(conv_num=n_layer, n_hidden=n_hidden, k=7, n_head=n_head)
        self.prediction = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Tanh(),
            nn.Linear(n_hidden // 2, 1),
        )

    def forward(self, question, context, label):
        Q = self.embedding(question)
        C = self.embedding(context)
        C = self.context_conv(C)
        Q = self.question_conv(Q)
        Ce = self.c_emb_enc(C, None)
        Qe = self.q_emb_enc(Q, None)
        X = self.cq_att(Qe, Ce)
        representations = self.aggregation(X, None)
        mask_idx = torch.eq(context, self.vocab_size)
        hidden = representations.masked_select(mask_idx.unsqueeze(2).expand_as(representations)).view(
            -1, self.n_hidden)
        answer_logit = self.prediction(hidden).squeeze(1)
        if label is None:
            return torch.sigmoid(answer_logit)
        return F.binary_cross_entropy_with_logits(answer_logit, label)
