import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import GCN


class SeqMLP(nn.Module):
    def __init__(self
        , d_input: int
        , d_embed: int
        , num_layers: int
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.dropout = dropout

        self.embedding = nn.Linear(d_input, d_embed)
        self.layer_norm = nn.LayerNorm(
            normalized_shape=d_embed
            , eps=layer_norm_eps
        )

        self.layers = nn.ModuleList([
            nn.Linear(d_embed, d_embed) for _ in range(num_layers)
        ])
    
    def forward(self
        , seq: torch.Tensor
    ):
        seq = F.dropout(F.relu(self.layer_norm(self.embedding(seq))), p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            seq = layer(seq)
            if i != len(self.layers) - 1:
                seq = F.dropout(F.relu(seq), p=self.dropout, training=self.training)
        
        return seq


class TextMLP(nn.Module):
    def __init__(self
        , vocab_size: int
        , d_embed: int
        , num_layers: int
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size
            , embedding_dim=d_embed
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=d_embed
            , eps=layer_norm_eps
        )

        self.layers = nn.ModuleList([
            nn.Linear(d_embed, d_embed) for _ in range(num_layers)
        ])

    def forward(self
        , indices: torch.Tensor
    ):
        emb = F.dropout(F.relu(self.layer_norm(self.embedding(indices))), p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            emb = layer(emb)
            if i != len(self.layers) - 1:
                emb = F.dropout(F.relu(emb), p=self.dropout, training=self.training)
        
        return emb


class WideMLP(nn.Module):
    def __init__(self
        , vocab_size: int
        , d_embed: int
        , num_layers: int
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.dropout = dropout

        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size
            , embedding_dim=d_embed
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=d_embed
            , eps=layer_norm_eps
        )

        self.layers = nn.ModuleList([
            nn.Linear(d_embed, d_embed) for _ in range(num_layers)
        ])
    
    def forward(self
        , indices: torch.Tensor
    ):
        emb = F.dropout(F.relu(self.layer_norm(self.embedding(indices))), p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            emb = layer(emb)
            if i != len(self.layers) - 1:
                emb = F.dropout(F.relu(emb), p=self.dropout, training=self.training)
        
        return emb


class CommentEmbedding(nn.Module):
    def __init__(self
        , vocab_size: int
        , d_embed: int
        , mlp_num_layers: int
        , gcn_num_layers: int
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        self.embedding = WideMLP(
            vocab_size=vocab_size
            , d_embed=d_embed
            , num_layers=mlp_num_layers
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
        self.gcn = GCN(
            d_input=d_embed
            , d_output=d_embed
            , num_layers=gcn_num_layers
        )
    
    def forward(self
        , comment: torch.Tensor
        , comment_graph: torch.Tensor
    ):
        batch_size, num_block, num_comment, seq_len = comment.size()
        comment = comment.view(batch_size * num_block * num_comment, -1).contiguous()
        comment = self.embedding(comment)
        comment = comment.view(batch_size * num_block, num_comment, -1).contiguous()

        batch_size, num_block, height, width = comment_graph.size()
        comment_graph = comment_graph.view(batch_size * num_block, height, width).contiguous()
        comment = self.gcn(
            seq=comment
            , adj_mat=comment_graph
        )
        comment = comment.view(batch_size, num_block, -1).contiguous()

        return comment


class PositionEmbedding(nn.Module):
    def __init__(self
        , d_embed: int
        , max_seq_len: int = 1000
        , batch_first: bool = True
    ):
        super().__init__()
        self.batch_first = batch_first
        
        pe = torch.zeros(max_seq_len, d_embed)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self
        , x: torch.Tensor
    ):
        if self.batch_first:
            x = x.permute(1, 0, 2).contiguous()
        x + self.pe[:x.size(0), :]
        if self.batch_first:
            x = x.permute(1, 0, 2).contiguous()
        return x


class Embedding(nn.Module):
    def __init__(self
        , d_embed: int
        , vocab_size: int
        , mlp_num_layers: int
        , gcn_num_layers: int
        , dropout: float = 0.1
        , layer_norm_eps: float = 1e-5
        , max_token_type: int = 5
    ):
        super().__init__()

        self.video_embedding = SeqMLP(
            d_input=2048
            , d_embed=d_embed
            , num_layers=mlp_num_layers
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
        self.audio_embedding = SeqMLP(
            d_input=3200
            , d_embed=d_embed
            , num_layers=mlp_num_layers
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
        self.title_embedding = WideMLP(
            vocab_size=vocab_size
            , d_embed=d_embed
            , num_layers=mlp_num_layers
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
        self.comment_embedding = CommentEmbedding(
            vocab_size=vocab_size
            , d_embed=d_embed
            , mlp_num_layers=mlp_num_layers
            , gcn_num_layers=gcn_num_layers
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
        self.special_token_embedding = TextMLP(
            vocab_size=vocab_size
            , d_embed=d_embed
            , num_layers=mlp_num_layers
            , dropout=dropout
            , layer_norm_eps=layer_norm_eps
        )
        self.pos_embedding = PositionEmbedding(d_embed=d_embed)
        self.type_embedding = nn.Embedding(
            num_embeddings=max_token_type
            , embedding_dim=d_embed
        )
    
    def forward(self
        , video: torch.Tensor
        , audio: torch.Tensor
        , title: torch.Tensor
        , comment: torch.Tensor
        , comment_graph: torch.Tensor
        , special_token: torch.Tensor
    ):
        video_emb = self.pos_embedding(self.video_embedding(video))
        audio_emb = self.pos_embedding(self.audio_embedding(audio))
        title_emb = self.title_embedding(title).unsqueeze(1)
        comment_emb = self.pos_embedding(self.comment_embedding(comment, comment_graph))
        special_token_emb = self.special_token_embedding(special_token)

        token = torch.cat([
                special_token_emb[:, 0, :].unsqueeze(1)
                , video_emb
                , special_token_emb[:, 1, :].unsqueeze(1)
                , audio_emb
                , special_token_emb[:, 2, :].unsqueeze(1)
                , title_emb
                , special_token_emb[:, 3, :].unsqueeze(1)
                , comment_emb
                , special_token_emb[:, 4, :].unsqueeze(1)
            ], dim=1
        )

        token += self.type_embedding(torch.tensor(
            [0] + [1] * video_emb.size(1) + [0]
            + [2] * audio_emb.size(1) + [0]
            + [3] * title_emb.size(1) + [0]
            + [4] * comment_emb.size(1) + [0]
            , device=token.device
        ))

        return token