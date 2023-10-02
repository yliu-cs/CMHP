import os
import re
import math
import shutil
import argparse
from glob import glob

from matplotlib import ticker
import matplotlib.pyplot as plt

from .scheme import SCHEME, FONT

COLOR_MAP = {
    "SOM": 0
    , "NLC": 1
    , "VCM": 2
}


def get_args():
    names = ["-".join(".".join(log_path.split(os.sep)[-1].split(".")[:-1]).split("-")[1:]) for log_path in glob(os.path.join(os.getcwd(), "log", "pretrain-*"))]

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+", type=str, default=names)
    parser.add_argument("--x_gap", type=int, default=50)
    parser.add_argument("--y_gap", type=int, default=1)
    parser.add_argument("--all_scheme", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    return args


def complete_args(
    args: argparse.Namespace
):
    args.log_path = os.path.join(os.getcwd(), "log", f"pretrain-{args.name}.log")
    args.figure_dir = os.path.join(os.getcwd(), "figure", f"pretrain-{args.name}")
    return args


def get_log_info(
    args: argparse.Namespace
):
    flag = False
    loss_record = {"iter": [], "epoch": []}
    for line in open(args.log_path, "r").readlines():
        line = line[31:].strip()
        if re.match("^task=.*", line) is not None:
            args.task = line.split("=")[-1].split("_")
            for task in args.task:
                if task not in loss_record:
                    loss_record[task] = []
        if line == ("Pre-Train".center(50, "-")):
            if flag:
                break
            else:
                flag = True
                continue
        info = line.split(" ")
        if flag:
            loss_record["iter"].append(int(info[1].split("/")[0]))
            loss_record["epoch"].append(int(info[3].split("/")[0]))
            for i in range(5, len(info)):
                loss_record[info[i].split("_")[0]].append(float(info[i].split("=")[-1]))
    return loss_record, args


def plot_loss(
    loss_record: dict
    , args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "loss(pre-training)")):
        shutil.rmtree(os.path.join(args.figure_dir, "loss(pre-training)"))
    os.makedirs(os.path.join(args.figure_dir, "loss(pre-training)"))
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        ax.stackplot(
            loss_record["epoch"]
            , [loss_record[task] for task in args.task]
            , labels=args.task
            , colors=[SCHEME[scheme][COLOR_MAP[task]] for task in args.task]
            , alpha=0.8
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        x_min, x_max = loss_record["epoch"][0], loss_record["epoch"][-1]
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_xticks([x_min] + [_ for _ in range(((x_min + args.x_gap - 1) // args.x_gap) * args.x_gap, x_max + 1, args.x_gap)])
        # ax.xaxis.set_major_locator(plt.MultipleLocator(args.x_gap))
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x // 10)))
        y_min, y_max = 0, math.ceil(sum([max(value) for value in [loss_record[task] for task in args.task]]))
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_yticks(list(range(y_min, y_max + 1, args.y_gap)))
        # ax.yaxis.set_major_locator(plt.MultipleLocator(args.y_gap))
        for spine in ["top", "right"]:
            ax.spines[spine].set_color("none")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title="Task", loc="upper right")

        plt.savefig(os.path.join(args.figure_dir, "loss(pre-training)", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "loss(pre-training)", f"{scheme}.pdf"))
        plt.close()


def main():
    args = get_args()
    for name in args.name:
        args.name = name
        args = complete_args(args=args)
        loss_record, args = get_log_info(args)
        plot_loss(
            loss_record=loss_record
            , args=args
        )


if __name__ == "__main__":
    main()