import os
import math
from itertools import combinations
from argparse import ArgumentParser

import torch
import numpy as np
from torch.cuda import amp
from torch.utils.data import DataLoader

from models.model import Model
from data.dataset import Dataset
from utils.config import dfs_config, num_run, show_config
from utils.misc import get_model_size, ignore_warnings, seed_everything


TASK = ["SOM", "NLC", "VCM"]


def get_args():
    task = []
    for i in range(1, len(TASK) + 1):
        for comb in list(combinations(list(range(len(TASK))), i)):
            task.append("_".join(list(map(lambda x: TASK[x], sorted(list(comb))))))
    # "SOM" "NLC" "VCM" "SOM_NLC" "SOM_VCM" "NLC_VCM" "SOM_NLC_VCM"
    task = ["_".join(TASK)]
    
    parser = ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--log_path", type=str, default=os.path.join(os.getcwd(), "log", os.path.basename(__file__).split(".")[0]))
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(os.getcwd(), "checkpoint", os.path.basename(__file__).split(".")[0]))
    # data
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset"))
    parser.add_argument("--dataset_scale", nargs="+", type=int, default=[None])
    parser.add_argument("--modal", nargs="+", type=str, default=["video_audio_title_comment"])
    parser.add_argument("--resnet_scale", nargs="+", type=int, default=[152])
    parser.add_argument("--video_fps", nargs="+", type=int, default=[5])
    parser.add_argument("--audio_fps", nargs="+", type=int, default=[16000])
    parser.add_argument("--text_max_len", nargs="+", type=int, default=[16])
    parser.add_argument("--max_comment_block", nargs="+", type=int, default=[10])
    parser.add_argument("--comment_per_block", nargs="+", type=int, default=[11])
    parser.add_argument("--max_sec", nargs="+", type=int, default=[60])
    parser.add_argument("--max_wave", nargs="+", type=int, default=[300])
    parser.add_argument("--batch_size", nargs="+", type=int, default=[16])
    parser.add_argument("--num_workers", nargs="+", type=int, default=[4])
    parser.add_argument("--pin_memory", nargs="+", type=bool, default=[True])
    # model
    parser.add_argument("--d_embed", nargs="+", type=int, default=[768])
    parser.add_argument("--nhead", nargs="+", type=int, default=[12])
    parser.add_argument("--mlp_num_layers", nargs="+", type=int, default=[2])
    parser.add_argument("--gcn_num_layers", nargs="+", type=int, default=[4])
    parser.add_argument("--encoder_num_layers", nargs="+", type=int, default=[12])
    parser.add_argument("--d_hidden", nargs="+", type=int, default=[3072])
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.1])
    parser.add_argument("--layer_norm_eps", nargs="+", type=float, default=[1e-5])
    parser.add_argument("--activation", nargs="+", type=str, default=["gelu"])
    parser.add_argument("--max_token_type", nargs="+", type=int, default=[5])
    parser.add_argument("--num_nlc_class", nargs="+", type=int, default=[2])
    parser.add_argument("--num_vcm_class", nargs="+", type=int, default=[2])
    # train
    parser.add_argument("--task", nargs="+", type=str, default=task)
    parser.add_argument("--num_epoch", nargs="+", type=int, default=[300])
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--optim", nargs="+", type=str, default=["AdamW"])
    parser.add_argument("--lr", nargs="+", type=float, default=[5e-6])
    parser.add_argument("--weight_decay", nargs="+", type=float, default=[1e-5])
    parser.add_argument("--log_iter", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    return args


def get_loader(
    config: object
):
    train_loader = DataLoader(
        dataset=Dataset(
            config=config
            , mode="pre"
        )
        , batch_size=config.batch_size
        , shuffle=True
        , num_workers=config.num_workers
        , pin_memory=config.pin_memory
    )
    setattr(config, "vocab_size", len(train_loader.dataset.tokenizer))

    return train_loader


def train(
    config: object
):
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    if not os.path.exists(f"{os.sep}".join(config.log_path.split(os.sep)[:-1])):
        os.makedirs(f"{os.sep}".join(config.log_path.split(os.sep)[:-1]))
    
    setattr(config, "GPU", torch.cuda.get_device_name(config.cuda))
    logger = show_config(config=config)
    seed_everything(seed=config.seed)
    torch.cuda.set_device(config.cuda)

    train_loader = get_loader(config=config)
    if config.log_iter == 0:
        config.log_iter = int(math.ceil(len(train_loader.dataset) / config.batch_size)) * 10
        logger.info(f"log_iter={getattr(config, 'log_iter')}")
    model = Model(
        config=config
        , mode=os.path.basename(__file__).split(".")[0]
    ).cuda()
    logger.info(get_model_size(model=model))
    optimizer = getattr(torch.optim, config.optim)(
        params=filter(lambda p: p.requires_grad, model.parameters())
        , lr=config.lr
        , weight_decay=config.weight_decay
    )
    scaler = amp.GradScaler()
    
    logger.info("Pre-Train".center(50, "-"))
    iter, loss_record, sub_loss_record = 0, [], {}
    total_iter = config.num_epoch * ((len(train_loader.dataset) + config.batch_size - 1) // config.batch_size)
    for epoch in range(1, config.num_epoch + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            with amp.autocast():
                loss, sub_loss, feature, attn = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            if torch.isnan(loss):
                raise ValueError("Pre-Train loss is NAN")
            
            loss_record.append(loss.item())
            for key, value in sub_loss.items():
                if key not in sub_loss_record:
                    sub_loss_record[key] = []
                sub_loss_record[key].append(value.item())
            iter += 1
            if iter % config.log_iter == 0 or iter == total_iter:
                loss_record = loss_record[:-config.log_iter] + [np.mean(loss_record[-config.log_iter:])]
                for key in sub_loss.keys():
                    sub_loss_record[key] = sub_loss_record[key][:-config.log_iter] + [np.mean(sub_loss_record[key][-config.log_iter:])]
                log_loss = f"Loss={loss_record[-1]:.3f}"
                log_loss += "".join([f" {key}_Loss={value[-1]:.3f}" for key, value in sub_loss_record.items()])
                logger.info(f"[ITER. {iter}/{total_iter} (EPOCH {epoch}/{config.num_epoch})] {log_loss}")
                torch.save({
                    "config": config
                    , "iter": iter
                    , "model_state": model.state_dict()
                }, f"{config.ckpt_dir}/checkpoint_{iter}.pth")
    logger.info("Pre-Train".center(50, "-"))


def main():
    args = get_args()
    configs = num_run(configs=dfs_config(args=args))
    for config in configs:
        train(config)


if __name__ == "__main__":
    ignore_warnings()
    main()