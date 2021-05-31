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



class Linear(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Linear, self).__init__()
        
        self.model = nn.Linear(input_dim, output_dim)
    
    def forward(self, utterance_vector):
        predicted = self.model(utterance_vector)

        return predicted