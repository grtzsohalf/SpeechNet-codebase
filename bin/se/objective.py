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

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)

        src = predicted.masked_select(stft_length_masks.unsqueeze(-1).bool())
        tar = linear_tar.masked_select(stft_length_masks.unsqueeze(-1).bool())

        l1 = self.fn(src, tar)
        return l1, {}
        

class MSE(nn.Module):
    def __init__(self, eps=1e-10, **kwargs):
        super().__init__()
        self.eps = eps
        self.fn = torch.nn.MSELoss()

    def forward(self, predicted, target, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)

        src = predicted.masked_select(stft_length_masks.unsqueeze(-1).bool())
        tar = target.masked_select(stft_length_masks.unsqueeze(-1).bool())

        mse = self.fn(src, tar)
        return mse, {}
        
