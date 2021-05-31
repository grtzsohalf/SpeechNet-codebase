import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from src.util import init_weights, init_gate
from functools import lru_cache

import IPython
import pdb

class SAP(nn.Module):
    ''' VERIFI model, including AudioEncoder'''

    def __init__(self, out_dim):
        super(SAP, self).__init__()

        # Setup
        self.linear = nn.Linear(out_dim,out_dim)
        self.act_fn = nn.Tanh()
        self.sap_layer = SelfAttentionPooling(out_dim)
    
    def forward(self, feature, att_mask):

        ''' 
        Arguments
            audio_feature - [BxTxD]   Acoustic feature with shape 
            feature_len   - [BxD]     Length of each sample in a batch
        '''
        #Encode
        feature = self.linear(feature)
        feature = self.act_fn(feature)
        
        # if torch.isnan(feature).any() or torch.isinf(feature).any():
        #     print("invalid value in feature line 41", feature)

        sap_vec = self.sap_layer(feature, att_mask)

        return sap_vec, att_mask

class GE2E(nn.Module):
    """Implementation of the GE2E loss in https://arxiv.org/abs/1710.10467 [1]
    Accepts an input of size (N, M, D)
        where N is the number of speakers in the batch,
        M is the number of utterances per speaker,
        and D is the dimensionality of the embedding vector (e.g. d-vector)
    Args:
        - init_w (float): the initial value of w in Equation (5) of [1]
        - init_b (float): the initial value of b in Equation (5) of [1]
    """

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        super(GE2E, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def cosine_similarity(self, dvecs):
        """Calculate cosine similarity matrix of shape (N, M, N)."""
        n_spkr, n_uttr, d_embd = dvecs.size()

        dvec_expns = dvecs.unsqueeze(-1).expand(n_spkr, n_uttr, d_embd, n_spkr)
        dvec_expns = dvec_expns.transpose(2, 3)

        ctrds = dvecs.mean(dim=1).to(dvecs.device)
        ctrd_expns = ctrds.unsqueeze(0).expand(n_spkr * n_uttr, n_spkr, d_embd)
        ctrd_expns = ctrd_expns.reshape(-1, d_embd)

        dvec_rolls = torch.cat([dvecs[:, 1:, :], dvecs[:, :-1, :]], dim=1)
        dvec_excls = dvec_rolls.unfold(1, n_uttr-1, 1)
        mean_excls = dvec_excls.mean(dim=-1).reshape(-1, d_embd)

        indices = _indices_to_replace(n_spkr, n_uttr).to(dvecs.device)
        ctrd_excls = ctrd_expns.index_copy(0, indices, mean_excls)
        ctrd_excls = ctrd_excls.view_as(dvec_expns)

        return F.cosine_similarity(dvec_expns, ctrd_excls, 3, 1e-9)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """Calculate the loss on each embedding by taking softmax."""
        n_spkr, n_uttr, _ = dvecs.size()
        indices = _indices_to_replace(n_spkr, n_uttr).to(dvecs.device)
        losses = -F.log_softmax(cos_sim_matrix, 2)
        return losses.flatten().index_select(0, indices).view(n_spkr, n_uttr)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """Calculate the loss on each embedding by contrast loss."""
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j, i, j]) +
                             torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        """Calculate the GE2E loss for an input of dimensions (N, M, D)."""
        cos_sim_matrix = self.cosine_similarity(dvecs)
        torch.clamp(self.w, 1e-9)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()


@lru_cache(maxsize=5)
def _indices_to_replace(n_spkr, n_uttr):
    indices = [(s * n_uttr + u) * n_spkr + s
               for s in range(n_spkr) for u in range(n_uttr)]
    return torch.LongTensor(indices)



class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (N, T, 1)
        
        return:
        utter_rep: size (N, H)
        """
        seq_len = batch_rep.shape[1]
        att_logits = self.W(batch_rep).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep