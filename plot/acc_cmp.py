import os
import re
import math
import shutil
import argparse

from matplotlib import ticker
import matplotlib.pyplot as plt

from .scheme import SCHEME, FONT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+", type=str, required=True)
    parser.add_argument("--label", nargs="+", type=str, required=True)
    parser.add_argument("--x_gap", type=int, default=30)
    parser.add_argument("--y_gap", type=int, default=5)
    parser.add_argument("--all_scheme", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    if len(args.name) != 2:
        raise ValueError(f"{len(args.name)=} != 2")
    if len(args.name) != len(args.label):
        raise ValueError(f"{len(args.name)=} != {len(args.label)}")
    
    return args


def complete_args(
    args: argparse.Namespace
):
    args.log_path = [os.path.join(os.getcwd(), "log", f"finetune-{name}.log") for name in args.name]
    args.figure_dir = [os.path.join(os.getcwd(), "figure", f"finetune-{name}") for name in args.name]

    for log_path in args.log_path:
        if not os.path.exists(log_path):
            raise ValueError(f"{log_path=} not exists")
    
    return args


def get_log_info(
    log_path: str
):
    flag = False
    record = {"iter": [], "epoch": [], "acc": []}
    for line in open(log_path, "r").readlines():
        line = line[31:].strip()
        if line == ("Fine-Tune".center(50, "-")):
            flag = True
            continue
        if re.match("^Best_Iter.=.*", line) is not None:
            break
        info = line.split(" ")
        if flag:
            record["iter"].append(int(info[1].split("/")[0]))
            record["epoch"].append(int(info[3].split("/")[0]))
            record["acc"].append(float(info[5].split("=")[-1]))
    return record


def plot_acc_cmp(
    record: list
    , args: argparse.Namespace
):
    for figure_dir in args.figure_dir:
        if os.path.exists(os.path.join(figure_dir, "acc_cmp")):
            shutil.rmtree(os.path.join(figure_dir, "acc_cmp"))
        os.makedirs(os.path.join(figure_dir, "acc_cmp"))
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        for i in range(len(record)):
            ax.plot(
                record[i]["iter"]
                , record[i]["acc"]
                , label=args.label[i]
                , color=SCHEME[scheme][i]
            )

        ax.set_xlabel("Iter.")
        ax.set_ylabel("Acc")
        x_min, x_max = record[0]["iter"][0], record[0]["iter"][-1]
        ax.set_xlim(left=x_min, right=x_max)
        xticks = [x_min] + [_ for _ in range(((x_min + args.x_gap - 1) // args.x_gap) * args.x_gap, x_max + 1, args.x_gap)]
        if xticks[-1] != x_max:
            xticks += [x_max]
        ax.set_xticks(xticks)
        y_min, y_max = math.floor(min([min(record[i]["acc"]) for i in range(len(record))])), math.ceil(max([max(record[i]["acc"]) for i in range(len(record))]))
        y_min, y_max = int((y_min - 1) // args.y_gap) * args.y_gap, int((y_max + args.y_gap - 1) // args.y_gap) * args.y_gap
        ax.set_ylim(bottom=y_min, top=y_max)
        yticks = [y_min] + list(range(y_min, y_max + 1, args.y_gap))
        if yticks[-1] != y_max:
            yticks += [y_max]
        ax.set_yticks(yticks)
        for spine in ["top", "right"]:
            ax.spines[spine].set_color("none")
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="lower right")

        for figure_dir in args.figure_dir:
            plt.savefig(os.path.join(figure_dir, "acc_cmp", f"{scheme}.png"), dpi=600)
            if args.pdf:
                plt.savefig(os.path.join(figure_dir, "acc_cmp", f"{scheme}.pdf"))
        plt.close()


def main():
    args = complete_args(
        args=get_args()
    )
    record = [get_log_info(log_path) for log_path in args.log_path]
    plot_acc_cmp(
        record=record
        , args=args
    )


if __name__ == "__main__":
    main()