import os
import math
import pickle
from itertools import combinations
from argparse import ArgumentParser

import torch
import numpy as np
from torch.cuda import amp
from torch.utils.data import DataLoader

from models.model import Model
from data.dataset import Dataset
from utils.metrics import calc_metrics
from utils.config import dfs_config, from_ckpt, num_run, show_config
from utils.misc import get_model_size, ignore_warnings, seed_everything


MODAL = ["video", "audio", "title", "comment"]


def get_args():
    modal = []
    for i in range(1, len(MODAL) + 1):
        for comb in list(combinations(list(range(len(MODAL))), i)):
            modal.append("_".join(list(map(lambda x: MODAL[x], sorted(list(comb))))))
    # "video" "audio" "title" "comment" "video_audio" "video_title" "video_comment" "audio_title" "audio_comment" "title_comment" "video_audio_title" "video_audio_comment" "video_title_comment" "audio_title_comment" "video_audio_title_comment"
    modal = ["_".join(MODAL)]

    parser = ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--from_ckpt", action="store_true")
    parser.add_argument("--log_path", type=str, default=os.path.join(os.getcwd(), "log", os.path.basename(__file__).split(".")[0]))
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(os.getcwd(), "checkpoint", os.path.basename(__file__).split(".")[0]))
    parser.add_argument("--ckpt_name", type=str, default=None)
    # data
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset"))
    parser.add_argument("--dataset_scale", nargs="+", type=int, default=[None])
    parser.add_argument("--modal", nargs="+", type=str, default=modal)
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
    parser.add_argument("--num_class", nargs="+", type=int, default=[2])
    # train
    parser.add_argument("--task", nargs="+", type=str, default=["VHD"])
    parser.add_argument("--num_epoch", nargs="+", type=int, default=[30])
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    parser.add_argument("--optim", nargs="+", type=str, default=["AdamW"])
    parser.add_argument("--lr", nargs="+", type=float, default=[5e-5])
    parser.add_argument("--weight_decay", nargs="+", type=float, default=[1e-5])
    parser.add_argument("--log_iter", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    if args.from_ckpt and len(args.modal) > 1:
        raise ValueError(f"{args.from_ckpt=} & {args.modal=}")
    if (args.from_ckpt and args.ckpt_name is None) or (not args.from_ckpt and args.ckpt_name is not None):
        raise ValueError(f"{args.from_ckpt=} & {args.ckpt_name=}")

    return args


def get_loader(
    config: object
):
    train_loader = DataLoader(
        dataset=Dataset(
            config=config
            , mode="train"
        )
        , batch_size=config.batch_size
        , shuffle=True
        , num_workers=config.num_workers
        , pin_memory=config.pin_memory
    )
    setattr(config, "vocab_size", len(train_loader.dataset.tokenizer))
    val_loader = DataLoader(
        dataset=Dataset(
            config=config
            , mode="val"
        )
        , batch_size=config.batch_size
        , shuffle=True
        , num_workers=config.num_workers
        , pin_memory=config.pin_memory
    )
    test_loader = DataLoader(
        dataset=Dataset(
            config=config
            , mode="test"
        )
        , batch_size=config.batch_size
        , shuffle=True
        , num_workers=config.num_workers
        , pin_memory=config.pin_memory
    )
    return train_loader, val_loader, test_loader


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

    train_loader, val_loader, test_loader = get_loader(config=config)
    if config.log_iter == 0:
        config.log_iter = int(math.ceil(len(train_loader.dataset) / config.batch_size))
        logger.info(f"log_iter={getattr(config, 'log_iter')}")
    model = Model(
        config=config
        , mode=os.path.basename(__file__).split(".")[0]
    ).cuda()
    if config.from_ckpt:
        if not hasattr(config, "ckpt_path"):
            raise ValueError(f"{config.from_ckpt=} {hasattr(config, 'ckpt_path')=}")
        model.load_state_dict(
            torch.load(
                config.ckpt_path
                , map_location=torch.device(f"cuda:{config.cuda}")
            )["model_state"]
            , strict=False
        )
        model.cuda()
    logger.info(get_model_size(model=model))
    optimizer = getattr(torch.optim, config.optim)(
        filter(lambda p: p.requires_grad, model.parameters())
        , lr=config.lr
        , weight_decay=config.weight_decay
    )
    scaler = amp.GradScaler()

    logger.info("Fine-Tune".center(50, "-"))
    iter, loss_record, best_iter, max_acc = 0, [], -1, float("-inf")
    total_iter = config.num_epoch * ((len(train_loader.dataset) + config.batch_size - 1) // config.batch_size)
    for epoch in range(1, config.num_epoch + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            with amp.autocast():
                _, loss, feature, attn = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            if torch.isnan(loss):
                raise ValueError("Train loss is NAN")
            
            loss_record.append(loss.item())
            iter += 1
            if iter % config.log_iter == 0 or iter == total_iter:
                with torch.no_grad():
                    model.eval()
                    preds, labels = [], []
                    for batch in val_loader:
                        with amp.autocast():
                            (logits, label), _, _, _ = model(batch)
                        preds += torch.argmax(logits, dim=-1).detach().cpu().tolist()
                        labels += label.detach().tolist()
                        torch.cuda.empty_cache()
                    metrics = calc_metrics(labels, preds)
                    if metrics["acc"] > max_acc:
                        max_acc = metrics["acc"]
                        best_iter = iter
                        torch.save({
                            "config": config
                            , "iter": iter
                            , "model_state": model.state_dict()
                        }, f"{config.ckpt_dir}/best_checkpoint.pth")
                logger.info(f"[ITER. {iter}/{total_iter} (EPOCH {epoch}/{config.num_epoch})] Train_Loss={np.mean(loss_record[-config.log_iter:]):.3f} Val_Acc={metrics['acc']:.3f}")
                model.train()
    logger.info(f"Best_Iter.={best_iter}")
    logger.info("Fine-Tune".center(50, "-"))

    if best_iter == -1:
        raise ValueError(f"{best_iter=}")
    model.load_state_dict(
        torch.load(
            os.path.join(config.ckpt_dir, "best_checkpoint.pth")
            , map_location=torch.device(f"cuda:{config.cuda}")
        )["model_state"]
    )
    
    logger.info("Test".center(50, "="))
    with torch.no_grad():
        model.eval()
        preds, labels, features, attns = [], [], [], []
        for batch in test_loader:
            with amp.autocast():
                (logits, label), _, feature, attn = model(batch)
            preds += torch.argmax(logits, dim=-1).detach().cpu().tolist()
            labels += label.detach().tolist()
            features.append(feature.detach().cpu().numpy())
            attns.append(attn)
            torch.cuda.empty_cache()
        pickle.dump(
            obj={
                "feature": np.concatenate(features, axis=0)
                , "label": labels
                , "pred": preds
            }
            , file=open(os.path.join(config.ckpt_dir, "tSNE.pkl"), "wb")
        )
        pickle.dump(
            obj=np.mean(np.concatenate(attns, axis=0), axis=0)
            , file=open(os.path.join(config.ckpt_dir, "attn.pkl"), "wb")
        )
        metrics = calc_metrics(labels, preds)
    logger.info(f"Acc={metrics['acc']:.3f}")
    logger.info(f"Mac Pre={metrics['mac']['pre']:.3f} Rec={metrics['mac']['rec']:.3f} F1={metrics['mac']['f1']:.3f}")
    logger.info(f"Wtd Pre={metrics['wtd']['pre']:.3f} Rec={metrics['wtd']['rec']:.3f} F1={metrics['wtd']['f1']:.3f}")
    logger.info("Test".center(50, "="))


def main():
    args = get_args()
    if args.from_ckpt == True:
        configs = num_run(configs=from_ckpt(args=args))
        print(f"{len(configs)=}")
    else:
        configs = num_run(configs=dfs_config(args=args))
    for config in configs:
        train(config)


if __name__ == "__main__":
    ignore_warnings()
    main()