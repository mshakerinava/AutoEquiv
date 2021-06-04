import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


def create_colored_matrix(input_generators, output_generators):
    def dfs(i, j, colors, color_idx, input_generators, output_generators):
        colors[(i, j)] = color_idx
        for k in range(len(input_generators)):
            i_ = input_generators[k][i]
            j_ = output_generators[k][j]
            if (i_, j_) not in colors:
                dfs(i_, j_, colors, color_idx, input_generators, output_generators)

    assert len(input_generators) == len(output_generators)
    assert len(input_generators) > 0
    color_idx = 0
    colors = {}
    for i in range(len(input_generators[0])):
        for j in range(len(output_generators[0])):
            if (i, j) not in colors:
                dfs(i, j, colors, color_idx, input_generators, output_generators)
                color_idx += 1
    return colors


def create_colored_vector(output_generators):
    def dfs(i, colors, color_idx, output_generators):
        colors[i] = color_idx
        for k in range(len(output_generators)):
            i_ = output_generators[k][i]
            if i_ not in colors:
                dfs(i_, colors, color_idx, output_generators)

    assert len(output_generators) > 0
    color_idx = 0
    colors = {}
    for i in range(len(output_generators[0])):
        if i not in colors:
            dfs(i, colors, color_idx, output_generators)
            color_idx += 1
    return colors


# adapted from the PyTorch implementation
def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', fan=None):
    if fan is None:
        fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class LinearEquiv(nn.Module):
    def __init__(self, in_generators, out_generators, in_channels, out_channels, bias=True):
        super(LinearEquiv, self).__init__()
        self.in_features = len(in_generators[0])
        self.out_features = len(out_generators[0])
        self.in_generators = deepcopy(in_generators)
        self.out_generators = deepcopy(out_generators)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.colors_W = create_colored_matrix(in_generators, out_generators)
        self.colors_b = create_colored_vector(out_generators)
        self.num_colors_W = len(set(self.colors_W.values()))
        self.num_colors_b = len(set(self.colors_b.values()))
        self.num_weights_W = self.num_colors_W * in_channels * out_channels
        self.num_weights_b = self.num_colors_b * out_channels
        self.num_weights = self.num_weights_W + (self.num_weights_b if bias else 0)

        self.weight = nn.Parameter(torch.Tensor(self.num_weights_W))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_weights_b))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        idx_weight = np.zeros((self.out_features * out_channels, self.in_features * in_channels), dtype=int)
        for i in range(out_channels):
            row_base = i * self.out_features
            for j in range(in_channels):
                col_base = j * self.in_features
                v_base = (i * in_channels + j) * self.num_colors_W
                for k, v in self.colors_W.items():
                    idx_weight[row_base + k[1], col_base + k[0]] = v_base + v
        self.register_buffer('idx_weight', torch.tensor(idx_weight, dtype=torch.long))

        idx_bias = np.zeros((self.out_features * out_channels,), dtype=int)
        for i in range(out_channels):
            row_base = i * self.out_features
            v_base = i * self.num_colors_b
            for k, v in self.colors_b.items():
                idx_bias[row_base + k] = v_base + v
        self.register_buffer('idx_bias', torch.tensor(idx_bias, dtype=torch.long))

    def reset_parameters(self):
        fan_in = self.in_features * self.in_channels
        kaiming_uniform_(self.weight, a=math.sqrt(5), mode='fan_in', fan=fan_in)
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        W = self.weight[self.idx_weight]
        assert W.shape == (self.out_features * self.out_channels, self.in_features * self.in_channels)

        if self.bias is not None:
            b = self.bias[self.idx_bias]
            assert b.shape == (self.out_features * self.out_channels,)

        assert x.shape[-1] == self.in_features
        assert x.shape[-2] == self.in_channels
        x = x.view(*x.shape[:-2], self.in_features * self.in_channels)
        x = F.linear(x, W, b)
        assert x.shape[-1] == self.out_features * self.out_channels
        x = x.view(*x.shape[:-1], self.out_channels, self.out_features)
        return x

    def __repr__(self):
        return 'LinearEquiv(in_generators=%s, out_generators=%s, in_channels=%d, out_channels=%d, bias=%r)' % (
            str(self.in_generators), str(self.out_generators), self.in_channels, self.out_channels, (self.bias is not None))
