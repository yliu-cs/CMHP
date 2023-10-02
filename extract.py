import os
import pickle
from argparse import ArgumentParser

import torch
import torchaudio
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from utils.misc import ignore_warnings


def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset"))
    parser.add_argument("--resnet", type=int, default=152)
    parser.add_argument("--video_fps", type=int, default=5)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    return parser.parse_args()


def get_video_dir_list(
    dataset_dir: str
):
    video_dir_list = [os.path.join(dataset_dir, "unlabeled", video_id) for video_id in os.listdir(os.path.join(dataset_dir, "unlabeled"))]
    for mode in ("train", "val", "test"):
        video_dir_list += [os.path.join(dataset_dir, "labeled", mode, video_id) for video_id in os.listdir(os.path.join(dataset_dir, "labeled", mode))]
    return video_dir_list


def load_video(
    video_path: str
    , video_fps: int
    , video_width: int
    , video_height: int
):
    if video_width != video_height:
        raise ValueError(f"{video_width=} != {video_height=}")
    
    streamer = torchaudio.io.StreamReader(src=video_path)
    streamer.add_basic_video_stream(
        frames_per_chunk=video_fps
        , frame_rate=video_fps
        , width=video_width
        , height=video_height
        , format="rgb24"
    )

    video_chunk_list = []
    for video_chunk in streamer.stream():
        if video_chunk is not None:
            video_chunk_list.append(video_chunk[0].float().numpy())
    
    return video_chunk_list


def extract_resnet_feature(
    video_dir_list: list
    , resnet: int
    , fps: int
    , width: int
    , height: int
):
    if width != height:
        raise ValueError(f"{width=} != {height=}")
    backbone = nn.Sequential(
        *list(
            getattr(torchvision.models, f"resnet{resnet}")(
                weights=getattr(torchvision.models, f"ResNet{resnet}_Weights").DEFAULT
            ).children()
        )[:-2]
    )
    if torch.cuda.is_available():
        backbone = backbone.cuda()
    backbone.eval()

    for video_dir in tqdm(video_dir_list, desc="extract_resnet", ncols=100):
        if os.path.exists(os.path.join(video_dir, "feature", f"resnet_{resnet}-fps_{fps}.pkl")):
            continue
        
        video_chunk_list = load_video(
            video_path=f"{video_dir}/video.mp4"
            , video_fps=fps
            , video_width=width
            , video_height=height
        )
        
        feature = []
        for video in video_chunk_list:
            video = torch.tensor(video)
            if torch.cuda.is_available():
                video = video.cuda()

            with torch.no_grad():
                buffer = torch.mean(
                    backbone(video)
                    , dim=[-2, -1]
                ).detach().cpu().numpy()
            feature.append(buffer)
        feature = np.concatenate(feature, axis=0)

        if not os.path.exists(os.path.join(video_dir, "feature")):
            os.mkdir(os.path.join(video_dir, "feature"))
        pickle.dump(feature, open(os.path.join(video_dir, "feature", f"resnet_{resnet}-fps_{fps}.pkl"), "wb"))


def load_audio(
    video_path: str
    , audio_fps: int = 16000
):
    streamer = torchaudio.io.StreamReader(src=video_path)
    streamer.add_basic_audio_stream(
        frames_per_chunk=audio_fps
        , sample_rate=audio_fps
    )

    audio_chunk_list = []
    for audio_chunk in streamer.stream():
        if audio_chunk is not None:
            audio_chunk_list.append(audio_chunk[0].float())
    audio = torch.cat(
        audio_chunk_list
        , dim=0
    ) # [frame, channel]
    if audio.dim() == 2:
        if 1 <= audio.size(1) <= 2:
            audio = torch.mean(audio, dim=1)
            # [frame]
        else:
            raise ValueError(f"{audio.size()=}")
    
    audio = audio.numpy()
    return audio


def extract_raw_audio_feature(
    video_dir_list: list
    , fps: int
):
    for video_dir in tqdm(video_dir_list, desc="extract_audio", ncols=100):
        if os.path.exists(os.path.join(video_dir, "feature", f"audio-fps_{fps}.pkl")):
            continue
        audio = load_audio(
            video_path=os.path.join(video_dir, "video.mp4")
            , audio_fps=fps
        ).squeeze()
        audio = np.pad(audio, ((0, fps * (audio.shape[0] // fps + 1) - audio.shape[0])))
        audio = audio.reshape((-1, fps // 5))
        
        if not os.path.exists(os.path.join(video_dir, "feature")):
            os.mkdir(os.path.join(video_dir, "feature"))
        pickle.dump(audio, open(os.path.join(video_dir, "feature", f"audio-fps_{fps}.pkl"), "wb"))


def main():
    args = get_args_parser()
    video_dir_list = get_video_dir_list(args.dataset_dir)

    extract_resnet_feature(
        video_dir_list=video_dir_list
        , resnet=args.resnet
        , fps=args.video_ps
        , width=args.width
        , height=args.height
    )
    extract_raw_audio_feature(
        video_dir_list=video_dir_list
        , fps=16000
    )


if __name__ == "__main__":
    ignore_warnings()
    main()