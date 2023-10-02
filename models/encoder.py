import math
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


class Linear(nn.Module):
    def __init__(self
        , d_input: int
        , d_output: int
        , dropout: float = 0.1
        , relu: bool = True
        , layer_norm: bool = True
        , layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.relu = relu
        self.layer_norm = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(
                normalized_shape=d_input
                , eps=layer_norm_eps
            )
        self.linear = nn.Sequential([
            nn.Dropout(dropout)
            , nn.Linear(d_input, d_output)
        ])
    
    def forward(self
        , x: torch.Tensor
    ):
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = self.linear(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class SelfAttention(nn.Module):
    def __init__(self
        , d_embed: int
        , num_heads: int
        , dropout: float = 0.1
    ):
        super().__init__()
        self.d_embed, self.num_heads = d_embed, num_heads
        self.d_head = int(self.d_embed // num_heads)
        if self.d_head * self.num_heads != self.d_embed:
            raise ValueError(f"The hidden size {self.d_embed} is not multiple of the number of attention heads {self.d_head}")

        self.query = nn.Linear(self.d_embed, self.d_embed)
        self.key = nn.Linear(self.d_embed, self.d_embed)
        self.value = nn.Linear(self.d_embed, self.d_embed)

        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self
        , x: torch.Tensor
    ):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.d_head)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self
        , hidden_states: torch.Tensor
        , attn_mask: Optional[torch.FloatTensor] = None
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.d_head)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context_layer = torch.matmul(attn_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.d_embed,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer, attn_probs)


class ResidualLinear(nn.Module):
    def __init__(self
        , d_input: int
        , d_output: int
        , dropout:float = 0.1
        , layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_output, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self
        , hidden_states: torch.Tensor
        , input_tensor: torch.Tensor
    ):
        hidden_states = self.dropout(self.linear(hidden_states))
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states
    

class Attention(nn.Module):
    def __init__(self
        , d_embed: int
        , num_heads: int
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            d_embed=d_embed
            , num_heads=num_heads
            , dropout=dropout
        )
        self.res_linear = ResidualLinear(
            d_input=d_embed
            , d_output=d_embed
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
    
    def forward(self
        , input_tensor: torch.Tensor
        , attn_mask: Optional[torch.FloatTensor] = None
    ):
        attn_output, attn_probs = self.self_attn(
            hidden_states=input_tensor
            , attn_mask=attn_mask
        )
        attn_output = self.res_linear(
            hidden_states=attn_output
            , input_tensor=input_tensor
        )
        return (attn_output, attn_probs)


class ActLinear(nn.Module):
    def __init__(self
        , d_input: int
        , d_output: int
        , activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]
    ):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output)
        if isinstance(activation, str):
            self.act_fn = {
                "gelu": gelu
                , "gelu_new": gelu_new
                , "relu": F.relu
                , "swish": swish
            }[activation]
        else:
            self.act_fn = activation
        
    def forward(self
        , hidden_states: torch.Tensor
    ):
        return self.act_fn(self.linear(hidden_states))


class EncoderLayer(nn.Module):
    def __init__(self
        , d_model: int
        , nhead: int
        , d_hidden: int
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
        , activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu"
    ):
        super().__init__()
        self.attn = Attention(
            d_embed=d_model
            , num_heads=nhead
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
        self.act_linear = ActLinear(
            d_input=d_model
            , d_output=d_hidden
            , activation=activation
        )
        self.res_linear = ResidualLinear(
            d_input=d_hidden
            , d_output=d_model
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
    
    def forward(self
        , hidden_states: torch.Tensor
        , attn_mask: Optional[torch.Tensor] = None
    ):
        attn_output, layer_attn = self.attn(
            input_tensor=hidden_states
            , attn_mask=attn_mask
        )
        layer_output = self.res_linear(
            hidden_states=self.act_linear(
                hidden_states=attn_output
            )
            , input_tensor=attn_output
        )
        return (layer_output, layer_attn)


class Pooler(nn.Module):
    def __init__(self
        , d_model: int
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
    
    def forward(self
        , hidden_states: torch.Tensor
    ):
        return self.tanh(self.linear(hidden_states[:, 0]))


class Encoder(nn.Module):
    def __init__(self
        , d_model: int = 768
        , d_hidden: int = 3072
        , nhead: int = 12
        , num_layers: int = 6
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
        , activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu"
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model
                , nhead=nhead
                , d_hidden=d_hidden
                , dropout=dropout
                , layer_norm_eps=layer_norm_eps
                , activation=activation
            ) for _ in range(num_layers)
        ])

    def forward(self
        , hidden_states: torch.Tensor
        , attn_mask: Optional[torch.Tensor] = None
    ):
        ext_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        ext_attn_mask = (1. - ext_attn_mask) * -1e5

        all_hidden_states, all_attn = [hidden_states], []
        for layer in self.layers:
            hidden_states, layer_attn = layer(
                hidden_states=hidden_states
                , attn_mask=ext_attn_mask
            )
            all_hidden_states.append(hidden_states.detach())
            all_attn.append(layer_attn.detach())
        all_attn = torch.mean(
            torch.mean(
                torch.stack(all_attn, dim=0)
                , dim=0
            )
            , dim=1
        ).detach().cpu().numpy()
        
        return hidden_states, all_hidden_states, all_attn