import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import Embedding
from .encoder import Encoder, Pooler


class Model(nn.Module):
    def __init__(self
        , config: object
        , mode: str = "finetune"
    ):
        super().__init__()

        self.config = config
        self.mode = mode

        if self.mode != "pretrain" and self.config.task != "VHD":
            raise ValueError(f"{self.__class__.__name__}: {self.mode=} {self.config.task=}")

        self.embedding = Embedding(
            d_embed=self.config.d_embed
            , vocab_size=self.config.vocab_size
            , mlp_num_layers=self.config.mlp_num_layers
            , gcn_num_layers=self.config.gcn_num_layers
            , dropout=self.config.dropout
            , layer_norm_eps=self.config.layer_norm_eps
            , max_token_type=self.config.max_token_type
        )
        self.encoder = Encoder(
            d_model=self.config.d_embed
            , d_hidden=self.config.d_hidden
            , nhead=self.config.nhead
            , num_layers=self.config.encoder_num_layers
            , dropout=self.config.dropout
            , layer_norm_eps=self.config.layer_norm_eps
            , activation=self.config.activation
        )
        self.pooler = Pooler(
            d_model=self.config.d_embed
        )

        if self.mode == "finetune":
            self.vhd_output = nn.Linear(self.config.d_embed, self.config.num_class)
        elif self.mode == "pretrain":
            if "SOM" in self.config.task:
                self.som_output = nn.Linear(self.config.d_embed, self.config.video_fps * self.config.max_sec)
            if "NLC" in self.config.task:
                self.nlc_output = nn.Linear(self.config.d_embed, self.config.num_nlc_class)
            if "VCM" in self.config.task:
                self.vcm_output = nn.Linear(self.config.d_embed, self.config.num_vcm_class)
    
    def forward(self
        , batch: list
    ):
        param, label = batch[0], batch[-1]
        video, audio, title, comment, comment_graph, special_token, attn_mask = list(map(lambda x: x.cuda(), param))
        label = list(map(lambda x: x.cuda(), label))

        video_seq_len, audio_seq_len, comment_seq_len = video.size(1), audio.size(1), comment.size(1)
        token = self.embedding(video, audio, title, comment, comment_graph, special_token)
        last_hidden_states, all_hidden_states, all_attn = self.encoder(token, attn_mask)
        feature = self.pooler(last_hidden_states)

        if self.mode == "finetune":
            vhd_label = label[0]
            vhd_logits = self.vhd_output(feature)
            loss = F.cross_entropy(vhd_logits, vhd_label)
            return (vhd_logits, vhd_label), loss, feature, all_attn
        elif self.mode == "pretrain":
            loss, sub_loss = 0, {}
            if "SOM" in self.config.task:
                batch_size = last_hidden_states.size(0)
                som_hidden = torch.cat([
                        last_hidden_states[:, 1:video_seq_len + 1, :]
                        , last_hidden_states[:, video_seq_len + 2:video_seq_len + audio_seq_len + 2, :]
                        , last_hidden_states[:, video_seq_len + audio_seq_len + 5:video_seq_len + audio_seq_len + comment_seq_len + 5, :]
                    ], dim=1
                ).contiguous()
                som_attn_mask = torch.cat([
                        attn_mask[:, 1:video_seq_len + 1]
                        , attn_mask[:, video_seq_len + 2:video_seq_len + audio_seq_len + 2]
                        , attn_mask[:, video_seq_len + audio_seq_len + 5:video_seq_len + audio_seq_len + comment_seq_len + 5]
                    ], dim=1
                ).contiguous()
                som_hidden = som_hidden.view(batch_size * (video_seq_len + audio_seq_len + comment_seq_len), -1).contiguous()
                som_attn_mask = som_attn_mask.view(batch_size * (video_seq_len + audio_seq_len + comment_seq_len)).contiguous()
                som_hidden = torch.index_select(
                    som_hidden
                    , dim=0
                    , index=torch.nonzero(som_attn_mask).squeeze()
                )
                som_logits = self.som_output(som_hidden).contiguous()
                som_label = label.pop(-1)
                som_label = som_label.view(som_label.size(0) * som_label.size(1)).contiguous().squeeze()
                som_label = torch.index_select(
                    som_label
                    , dim=0
                    , index=torch.nonzero(som_label).squeeze()
                ) - 1
                som_loss = F.cross_entropy(som_logits, som_label)
                loss += som_loss
                sub_loss["SOM"] = som_loss
            if "NLC" in self.config.task:
                nlc_logits = self.nlc_output(feature)
                nlc_label = label.pop(-1)
                nlc_loss = F.cross_entropy(nlc_logits, nlc_label)
                loss += nlc_loss
                sub_loss["NLC"] = nlc_loss
            if "VCM" in self.config.task:
                vcm_logits = self.vcm_output(feature)
                vcm_label = label.pop(-1)
                vcm_loss = F.cross_entropy(vcm_logits, vcm_label)
                loss += vcm_loss
                sub_loss["VCM"] = vcm_loss
            return loss, sub_loss, feature, all_attn