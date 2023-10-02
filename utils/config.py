import os
import sys
import logging
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from argparse import ArgumentParser
from time import strftime, localtime

import torch


class Config(object):
    def __init__(self):
        pass

    def __repr__(self):
        return str(self.__dict__)


def from_ckpt(
    args: ArgumentParser
):
    constant_param = [
        "resnet_scale", "video_fps", "audio_fps"
        , "text_max_len", "max_comment_block", "comment_per_block", "max_sec", "max_wave"
        , "d_embed", "nhead", "mlp_num_layers", "gcn_num_layers", "encoder_num_layers"
        , "d_hidden", "dropout", "layer_norm_eps", "activation", "max_token_type"
    ]

    configs = []
    for ckpt_path in tqdm(glob(os.path.join(os.sep.join(args.ckpt_dir.split(os.sep)[:-1]), f"pretrain-{args.ckpt_name}*", "checkpoint_*.pth")), desc="load ckpt"):
        config = torch.load(
            ckpt_path
            , map_location=torch.device("cpu")
        )["config"]
        for attr in deepcopy(vars(config)):
            if attr not in constant_param:
                delattr(config, attr)
        setattr(config, "ckpt_path", ckpt_path)
        configs.append(config)
    configs = sum(list(map(lambda x: dfs_config(args, x), configs)), [])

    return configs


def dfs_config(
    args: ArgumentParser
    , config: Config = None
):
    if config is None:
        config = Config()
    
    flag = False not in [hasattr(config, arg) for arg in vars(args)]

    configs = []
    for arg in vars(args):
        if not hasattr(config, arg):
            values = getattr(args, arg)
            if isinstance(values, list):
                for value in values:
                    setattr(config, arg, value)
                    configs += dfs_config(args, config)
            else:
                setattr(config, arg, values)
                configs += dfs_config(args, config)
            delattr(config, arg)
            break
    
    return configs + ([deepcopy(config)] if flag else [])


def num_run(
    configs: list
):
    for idx, config in enumerate(configs):
        setattr(config, "run_name", f"{strftime('%Y.%m.%d-%H:%M', localtime())}-{idx + 1}")
        config.log_path += f"-{getattr(config, 'run_name')}.log"
        config.ckpt_dir += f"-{getattr(config, 'run_name')}"
    return configs


def show_config(
    config: Config
):
    logger = logging.getLogger(config.run_name)
    file_handler = logging.FileHandler(config.log_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)-5s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    logger.info("Config".center(50, "*"))
    for attr in vars(config):
        logger.info(f"{attr}={getattr(config, attr)}")
    logger.info("Config".center(50, "*"))
    
    return logger