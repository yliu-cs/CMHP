import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionNetwork(nn.Module):
    def __init__(self
        , d_input: int
        , d_output: int
        , bias: bool = True
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        self.weight = nn.Parameter(torch.FloatTensor(d_input, d_output))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(d_output))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self
        , seq: torch.Tensor
        , adj_mat: torch.Tensor
    ):
        hidden = torch.matmul(seq, self.weight)
        denom = torch.sum(adj_mat, dim=2, keepdim=True) + 1
        output = torch.matmul(adj_mat, hidden.float()) / denom

        if self.bias is not None:
            output += self.bias

        return F.relu(output)


class GCN(nn.Module):
    def __init__(self
        , d_input: int
        , d_output: int
        , bias: bool = True
        , num_layers: int = 2
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            GraphConvolutionNetwork(
                d_input=d_input
                , d_output=d_output
                , bias=bias
            ) for _ in range(num_layers)
        ])
    
    def forward(self
        , seq: torch.Tensor
        , adj_mat: torch.Tensor
    ):
        for layer_i, layer in enumerate(self.layers):
            x = layer(seq=seq, adj_mat=adj_mat)
        
        beta = torch.matmul(seq, x.transpose(1, 2).contiguous())
        alpha = F.softmax(beta.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, x).squeeze(1)
        
        return x