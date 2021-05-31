import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, **kwargs):
        predicted = self.linear(features)
        return predicted, {}

class Separator(nn.Module):
    def __init__(self, input_dim, output_dim, module_cls, n_src, **kwargs):
        super().__init__()
        self.dim = input_dim
        self.submodules = nn.ModuleList([module_cls(input_dim, output_dim) for i in range(n_src)])

    def forward(self, features, **kwargs):
        outputs = []
        for module in self.submodules:
            outputs.append(module(features))
        return outputs
