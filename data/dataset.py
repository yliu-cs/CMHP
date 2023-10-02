import os
import json
import pickle
import random

import torch
import numpy as np
from transformers import AutoTokenizer

from .load import text_preprocess, load_comment


class Dataset(torch.utils.data.Dataset):
    def __init__(self
        , config: object
        , mode: str = "train"
    ):
        super().__init__()

        self.config = config
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

        if self.mode != "pre" and self.config.task != "VHD":
            raise ValueError(f"{self.__class__.__name__}: {self.mode=} {self.config.task=}")
        
        if self.mode in ("train", "val", "test"):
            self.path = os.path.join(self.config.dataset_dir, "labeled", self.mode)
        elif self.mode == "pre":
            self.path = os.path.join(self.config.dataset_dir, "unlabeled")
        
        self.ids = sorted(os.listdir(self.path))
        if self.config.dataset_scale is not None:
            scale = min(self.config.dataset_scale + (random.randint(-200, 200) if self.mode == "pre" else 0), len(self.ids))
            self.ids = self.ids[:scale]
    
    def __len__(self):
        return len(self.ids)
    
    def __get_info__(self
        , video_id: str
    ):
        info = json.loads(open(os.path.join(self.path, video_id, "info.json"), "r", encoding="utf8").read())
        return info
    
    def __get_label__(self
        , video_id: str
    ):
        if self.mode == "pre" or self.config.task != "VHD":
            raise ValueError(f"Try to get label in {self.mode=} & {self.config.task=}")
        
        info = self.__get_info__(video_id=video_id)
        humor = info["humor"]

        return humor
    
    def __get_num_of_like__(self
        , video_id: str
    ):
        info = self.__get_info__(video_id=video_id)
        num_of_like = float(info["like_number"])

        return num_of_like
    
    def __video_pad__(self
        , video: np.ndarray
    ):
        attn_mask = [1.] * video.shape[0] + [0.] * (self.config.max_sec * self.config.video_fps - video.shape[0])
        video = np.pad(video, ((0, self.config.max_sec * self.config.video_fps - video.shape[0]), (0, 0)))

        if "video" not in self.config.modal:
            video = np.zeros_like(video, dtype=video.dtype)
            attn_mask = [0.] * len(attn_mask)
            
        return video, attn_mask
    
    def __audio_pad__(self
        , audio: np.ndarray
    ):
        attn_mask = [1.] * audio.shape[0] + [0.] * (self.config.max_wave - audio.shape[0])
        audio = np.pad(audio, ((0, self.config.max_wave - audio.shape[0]), (0, 0)))

        if "audio" not in self.config.modal:
            audio = np.zeros_like(audio, dtype=audio.dtype)
            attn_mask = [0.] * len(attn_mask)

        return audio, attn_mask
    
    def __comment_graph_pad__(self
        , comment_graph: np.ndarray
    ):
        comment_graph = np.pad(comment_graph, ((0, self.config.comment_per_block - comment_graph.shape[0]), (0, self.config.comment_per_block - comment_graph.shape[1])))
        return comment_graph
    
    def __comment_pad__(self
        , comment: list
        , comment_graph: list
    ):
        if len(comment) != len(comment_graph):
            raise ValueError(f"{len(comment)=} != {len(comment_graph)=}")
        
        for i in range(len(comment)):
            for j in range(len(comment[i])):
                comment[i][j] = np.array(self.tokenizer(
                    text=text_preprocess(comment[i][j])
                    , truncation=True
                    , padding="max_length"
                    , max_length=self.config.text_max_len
                    , add_special_tokens=False
                )["input_ids"])
            if len(comment[i]) < self.config.comment_per_block:
                comment[i] += [np.zeros(shape=(self.config.text_max_len), dtype=np.int64)] * (self.config.comment_per_block - len(comment[i]))
            comment[i] = np.stack(comment[i], axis=0)
            comment_graph[i] = self.__comment_graph_pad__(comment_graph=comment_graph[i])
        attn_mask = [1.] * len(comment) + [0.] * (self.config.max_comment_block - len(comment))
        comment += [np.zeros(shape=(self.config.comment_per_block, self.config.text_max_len), dtype=np.int64)] * (self.config.max_comment_block - len(comment))
        comment_graph += [np.zeros(shape=[self.config.comment_per_block, self.config.comment_per_block], dtype=np.float32)] * (self.config.max_comment_block - len(comment_graph))
        comment, comment_graph = [np.stack(_, axis=0) for _ in (comment, comment_graph)]

        if "comment" not in self.config.modal:
            comment = np.zeros_like(comment, dtype=comment.dtype)
            attn_mask = [0.] * len(attn_mask)
        
        return comment, comment_graph, attn_mask
    
    def __get_sample__(self
        , video_id: str
    ):
        video_path = os.path.join(self.path, video_id, "feature", f"resnet_{self.config.resnet_scale}-fps_{self.config.video_fps}.pkl")
        if not os.path.exists(video_path):
            raise ValueError(f"{video_path=} is not exist")
        video = pickle.load(open(video_path, "rb"))

        audio_path = os.path.join(self.path, video_id, "feature", f"audio-fps_{self.config.audio_fps}.pkl")
        if not os.path.exists(audio_path):
            raise ValueError(f"{audio_path=} is not exist")
        audio = pickle.load(open(audio_path, "rb"))

        video_seq_len, audio_seq_len = video.shape[0], audio.shape[0]
        video, video_attn_mask = self.__video_pad__(video)
        audio, audio_attn_mask = self.__audio_pad__(audio)

        info = self.__get_info__(video_id=video_id)
        title = np.array(
            self.tokenizer(
                text=text_preprocess(info["title"])
                , truncation=True
                , padding="max_length"
                , max_length=self.config.text_max_len
                , add_special_tokens=False
            )["input_ids"]
        )
        if "title" not in self.config.modal:
            title = np.zeros_like(title, dtype=title.dtype)
        comment, comment_graph = load_comment(comment_path=os.path.join(self.path, video_id, "comment.json"))
        comment_seq_len = len(comment)
        comment, comment_graph, comment_attn_mask = self.__comment_pad__(
            comment=comment
            , comment_graph=comment_graph
        )

        special_token = np.array(self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", "[SEP]", "[SEP]", "[SEP]"]))
        attn_mask = [1.]
        attn_mask += video_attn_mask + [1.]
        attn_mask += audio_attn_mask + [1.]
        attn_mask += ([1.] if "title" in self.config.modal else [0.]) + [1.]
        attn_mask += comment_attn_mask + [1.]
        attn_mask = np.array(attn_mask, dtype=np.float32)

        param = [video, audio, title, comment, comment_graph, special_token, attn_mask]
        seq_len = (video_seq_len, audio_seq_len, comment_seq_len)
        return param, seq_len
    
    def __getitem__(self
        , idx: int
    ):
        video_id = self.ids[idx]
        sample_data = self.__get_sample__(video_id)
        param, (video_seq_len, audio_seq_len, comment_seq_len) = list(map(torch.from_numpy, sample_data[0])), sample_data[-1]

        if self.mode in ("train", "val", "test"):
            label = [torch.LongTensor([1.]).squeeze() if self.__get_label__(video_id=video_id) else torch.LongTensor([0.]).squeeze()]
        elif self.mode == "pre":
            label = []
            if "VCM" in self.config.task:
                comment_match = False if random.randint(0, 1) == 0 else True
                if not comment_match:
                    rand_idx = random.randint(0, len(self.ids) - 1)
                    while rand_idx == idx:
                        rand_idx = random.randint(0, len(self.ids) - 1)
                    rand_video_id = self.ids[rand_idx]
                    rand_param, rand_seq_len = self.__get_sample__(rand_video_id)
                    param[3], param[4] = torch.from_numpy(rand_param[3]), torch.from_numpy(rand_param[4])
                    param[-1][-11:-1] = torch.from_numpy(rand_param[-1][-11:-1])
                    # param[-1] = torch.cat([param[-1][:-11], rand_param[-1][-11:-1], param[-1][-1]], dim=0)
                    comment_seq_len = rand_seq_len[-1]
                vcm_label = torch.LongTensor([1. if comment_match else 0.]).squeeze()
                label.append(vcm_label)
            if "NLC" in self.config.task:
                nlc_label = torch.LongTensor([1. if self.__get_num_of_like__(video_id=video_id) >= 10 ** 5 else 0.]).squeeze()
                label.append(nlc_label)
            if "SOM" in self.config.task:
                video_idx = torch.randperm(video_seq_len)
                audio_idx = torch.randperm(audio_seq_len)
                comment_idx = torch.randperm(comment_seq_len)
                som_label = torch.cat([video_idx, audio_idx, comment_idx], dim=0).squeeze()
                if som_label.size(0) != video_seq_len + audio_seq_len + comment_seq_len:
                    raise ValueError(f"{som_label.size(0)=} != {video_seq_len + audio_seq_len + comment_seq_len=}")
                video_idx = torch.cat([video_idx, torch.tensor(list(range(video_seq_len, param[0].shape[0])), dtype=video_idx.dtype)], dim=0).squeeze()
                audio_idx = torch.cat([audio_idx, torch.tensor(list(range(audio_seq_len, param[1].shape[0])), dtype=audio_idx.dtype)], dim=0).squeeze()
                comment_idx = torch.cat([comment_idx, torch.tensor(list(range(comment_seq_len, param[3].shape[0])), dtype=comment_idx.dtype)], dim=0).squeeze()
                param[0] = param[0][video_idx]
                param[1] = param[1][audio_idx]
                param[3] = param[3][comment_idx]
                param[4] = param[4][comment_idx]
                som_label = torch.cat([som_label + 1, torch.zeros(param[0].shape[0] + param[1].shape[0] + param[3].shape[0] - som_label.size(0))], dim=0).squeeze().long()
                label.append(som_label)

        return param, label