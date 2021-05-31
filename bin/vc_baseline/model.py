import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, features, **kwargs):
        predicted = self.linear_2(self.relu(self.linear_1(features)))
        return predicted, {}
