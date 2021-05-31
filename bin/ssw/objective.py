import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy
from scipy.signal.windows import hann as hanning
from torch import Tensor
from functools import partial
from utils import *
from asteroid.losses.sdr import SingleSrcNegSDR
from asteroid.losses.stoi import NegSTOILoss
from asteroid.losses.pmsqe import SingleSrcPMSQE

from itertools import permutations

class SISDR(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)
        src = predicted.exp().pow(0.5) * stft_length_masks.unsqueeze(-1)
        tar = linear_tar.exp().pow(0.5) * stft_length_masks.unsqueeze(-1)

        src = src.flatten(start_dim=1).contiguous()
        tar = tar.flatten(start_dim=1).contiguous()

        alpha = torch.sum(src * tar, dim=1) / (torch.sum(tar * tar, dim=1) + self.eps)
        ay = alpha.unsqueeze(1) * tar
        norm = torch.sum((ay - src) * (ay - src), dim=1) + self.eps
        loss = -10 * torch.log10(torch.sum(ay * ay, dim=1) / norm + self.eps)

        return loss.mean(), {}


class L1(nn.Module):
    def __init__(self, eps=1e-10, **kwargs):
        super().__init__()
        self.eps = eps
        self.fn = torch.nn.L1Loss()

    def forward(self, predicted, padded_srcs_feats, **kwargs):
        # predicted: (batch_size, n_src, channel, max_time)
        # padded_srcs_feats: (batch_size, n_src, channel, max_time)
        batch_size, n_src = predicted.size()[:2]
        perms = list(permutations(range(n_src)))
        loss_set = torch.stack([torch.stack([self.fn(predicted[i, perm], padded_srcs_feats[i]) for perm in perms]) for i in range(batch_size)])
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)
        return min_loss.mean(), [perms[idx] for idx in min_loss_idx]
        
