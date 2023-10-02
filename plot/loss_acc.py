import os
import re
import math
import shutil
import argparse
from glob import glob

from matplotlib import ticker
import matplotlib.pyplot as plt

from .scheme import SCHEME, FONT


def get_args():
    names = ["-".join(".".join(log_path.split(os.sep)[-1].split(".")[:-1]).split("-")[1:]) for log_path in glob(os.path.join(os.getcwd(), "log", "pretrain-*"))]

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+", type=str, default=names)
    parser.add_argument("--x_gap", type=int, default=50)
    parser.add_argument("--y_gap", type=int, default=1)
    parser.add_argument("--acc_y_gap", type=int, default=5)
    parser.add_argument("--baseline", type=float, default=74.589)
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


def get_finetune_acc_from_pretrain(
    pretrain_log_path: str
):
    pretrain_run_name = ".".join(pretrain_log_path.split(os.sep)[-1].split(".")[:-1])
    
    iter2acc = {}
    for finetune_log_path in glob(os.path.join(os.getcwd(), "log", "finetune-*.log")):
        iter, acc = -1, 0.
        for line in open(finetune_log_path, "r").readlines():
            line = line[31:].strip()
            if re.match(f"^ckpt_path=.*{pretrain_run_name}.*", line) is not None:
                iter = int(line.split(os.sep)[-1].split("_")[-1].split(".")[0])
            if iter != -1 and re.match(f"^Acc=.*", line) is not None:
                acc = float(line.split("=")[-1])
        if iter != -1 and acc != 0.:
            iter2acc[iter] = acc
    
    return iter2acc


def get_log_info(
    args: argparse.Namespace
):
    flag = False
    record = {"iter": [], "epoch": []}
    for line in open(args.log_path, "r").readlines():
        line = line[31:].strip()
        if re.match("^task=.*", line) is not None:
            args.task = line.split("=")[-1].split("_")
            for task in args.task:
                if task not in record:
                    record[task] = []
        if line == ("Pre-Train".center(50, "-")):
            if flag:
                break
            else:
                flag = True
                continue
        info = line.split(" ")
        if flag:
            record["iter"].append(int(info[1].split("/")[0]))
            record["epoch"].append(int(info[3].split("/")[0]))
            for i in range(5, len(info)):
                record[info[i].split("_")[0]].append(float(info[i].split("=")[-1]))
    record["acc"] = [0] * len(record["iter"])

    iter2acc = get_finetune_acc_from_pretrain(
        pretrain_log_path=args.log_path
    )
    for iter, acc in iter2acc.items():
        record["acc"][record["iter"].index(iter)] = acc

    return record, args


def plot_loss(
    record: dict
    , args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "loss&acc")):
        shutil.rmtree(os.path.join(args.figure_dir, "loss&acc"))
    os.makedirs(os.path.join(args.figure_dir, "loss&acc"))
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.stackplot(
            record["epoch"]
            , [record[task] for task in args.task]
            , labels=args.task
            , colors=SCHEME[scheme]
            , alpha=0.8
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        x_min, x_max = record["epoch"][0], record["epoch"][-1]
        ax1.set_xlim(left=x_min, right=x_max)
        ax1.set_xticks([x_min] + [_ for _ in range(((x_min + args.x_gap - 1) // args.x_gap) * args.x_gap, x_max + 1, args.x_gap)])
        # ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x // 10)))
        y_min, y_max = 0, math.ceil(sum([max(value) for value in [record[task] for task in args.task]]))
        ax1.set_ylim(bottom=y_min, top=y_max)
        ax1.set_yticks(list(range(y_min, y_max + 1, args.y_gap)))
        for spine in ["top"]:
            ax1.spines[spine].set_color("none")
        
        ax2.axhline(
            y=args.baseline
            , label="BSL"
            , color=SCHEME[scheme][len(args.task) + 1]
            , linewidth=.7
        )
        ax2.plot(
            record["epoch"]
            , record["acc"]
            , label="ACC"
            , linewidth=.7
            , color=SCHEME[scheme][len(args.task)]
            , marker="x"
            , markersize=2.5
        )
        ax2.annotate(
            max(record["acc"])
            , xy=(record["epoch"][record["acc"].index(max(record["acc"]))], max(record["acc"]))
            , xytext=(record["epoch"][record["acc"].index(max(record["acc"]))] + 10, max(record["acc"]) + 1)
            , arrowprops=dict(
                facecolor="black"
                , shrink=0.05
                , headwidth=2
                , headlength=2
                , width=0.05
            ),
        )
        ax2.set_ylabel("Acc", rotation=-90, labelpad=12)
        y_min, y_max = math.floor(min(record["acc"])), math.ceil(max(record["acc"]))
        y_min, y_max = int(((y_min + 4) // args.acc_y_gap) * args.acc_y_gap) - args.acc_y_gap, int(((y_max + 4) // args.acc_y_gap) * args.acc_y_gap)
        ax2.set_ylim(bottom=y_min, top=y_max)
        ax2.set_yticks([y_min] + [_ for _ in range(((y_min + args.acc_y_gap - 1) // args.acc_y_gap) * args.acc_y_gap, y_max + 1, args.acc_y_gap)])
        # ax2.yaxis.set_major_locator(plt.MultipleLocator(5))
        for spine in ["top"]:
            ax2.spines[spine].set_color("none")
        
        ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
        ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
        ax1.legend(
            ax1_handles + ax2_handles
            , ax1_labels + ax2_labels
            # , ncol=2
            , loc="upper right"
        )

        plt.savefig(os.path.join(args.figure_dir, "loss&acc", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "loss&acc", f"{scheme}.pdf"))
        plt.close()


def main():
    args = get_args()
    for name in args.name:
        args.name = name
        args = complete_args(args=args)
        record, args = get_log_info(args)
        plot_loss(
            record=record
            , args=args
        )


if __name__ == "__main__":
    main()