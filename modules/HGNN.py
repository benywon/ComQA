# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/14 下午5:21
 @FileName: HGNN.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

from modules.BERT import *


class HGNN(Transformer):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer, n_head, aggregation='fused', M=2):
        super().__init__(vocab_size, n_embedding, n_hidden, n_layer, n_head)
        self.aggregation = aggregation
        if aggregation == 'fused':
            self.gnn_projections = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(M)])
            self.gnn_projections_ln = nn.ModuleList([layer_norm(n_hidden) for _ in range(M)])
        else:  # 'pipeline'
            self.gnn_projections_intra = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(M)])
            self.gnn_projections_intra_ln = nn.ModuleList([layer_norm(n_hidden) for _ in range(M)])
            self.gnn_projections_inter = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(M)])
            self.gnn_projections_inter_ln = nn.ModuleList([layer_norm(n_hidden) for _ in range(M)])
            self.gnn_projections_global = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(M)])
            self.gnn_projections_global_ln = nn.ModuleList([layer_norm(n_hidden) for _ in range(M)])

        self.prediction = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Tanh(),
            nn.Linear(n_hidden // 2, 1),
        )
        self.vocab_size = vocab_size
        self.n_embedding = n_embedding

    @staticmethod
    def normalize_adj(adj, add_self_loop=True):
        if add_self_loop:
            idx = torch.arange(adj.size(1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

    def message_passing(self, hidden, function, adj):
        out = function(hidden)
        adj = self.normalize_adj(adj.float())
        return F.gelu(torch.matmul(adj.type_as(out), out))

    def fused_aggregation(self, hidden, A_intra, A_inter, A_global):
        A_fused = torch.logical_or(torch.logical_or(A_intra, A_inter), A_global)
        for gnn_projection_ln, gnn_projection in zip(self.gnn_projections_ln, self.gnn_projections):
            hidden = self.message_passing(hidden, gnn_projection, A_fused)
            hidden = gnn_projection_ln(hidden)
        return hidden

    def pipeline_aggregation(self, hidden, A_intra, A_inter, A_global):
        for gnn_projection_intra_ln, gnn_projection_inter_ln, gnn_projection_global_ln, gnn_projection_intra, gnn_projection_inter, gnn_projection_global in zip(
                self.gnn_projections_intra_ln,
                self.gnn_projections_inter_ln,
                self.gnn_projections_global_ln,
                self.gnn_projections_intra,
                self.gnn_projections_inter,
                self.gnn_projections_global):
            h_intra = self.message_passing(hidden, gnn_projection_intra, A_intra)
            h_intra = gnn_projection_intra_ln(h_intra)
            h_inter = self.message_passing(h_intra, gnn_projection_inter, A_inter)
            h_inter = gnn_projection_inter_ln(h_inter)
            hidden = self.message_passing(h_inter, gnn_projection_global, A_global)
            hidden = gnn_projection_global_ln(hidden)
        return hidden

    def forward(self, inputs):
        [seq, label, A_intra, A_inter, A_global] = inputs
        representations = self.get_hidden_representations(seq)
        if self.aggregation == 'fused':
            representations = self.fused_aggregation(representations, A_intra, A_inter, A_global)
        else:
            representations = self.pipeline_aggregation(representations, A_intra, A_inter, A_global)
        mask_idx = torch.eq(seq, self.vocab_size)
        hidden = representations.masked_select(mask_idx.unsqueeze(2).expand_as(representations)).view(
            -1, self.n_hidden)
        answer_logit = self.prediction(hidden).squeeze(1)
        if label is None:
            return torch.sigmoid(answer_logit)
        return F.binary_cross_entropy_with_logits(answer_logit, label)
