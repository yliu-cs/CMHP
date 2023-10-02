import os
import json
import shutil
import argparse
from glob import glob
from collections import Counter

from matplotlib import ticker
import matplotlib.pyplot as plt

from .scheme import SCHEME, FONT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure", "dataset"))
    parser.add_argument("--all_scheme", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    return args


def digital_number(
    info_path: int
    , key: str
):
    return len(str(json.loads(open(info_path, "r", encoding="utf8").read())[key]))


def plot_like_number(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "like_number")):
        shutil.rmtree(os.path.join(args.figure_dir, "like_number"))
    os.makedirs(os.path.join(args.figure_dir, "like_number"))

    like_num = {"labeled": [], "unlabeled": []}
    like_num["unlabeled"] = [Counter(list(map(lambda x: digital_number(info_path=x, key="like_number"), glob(os.path.join(args.dataset_dir, "unlabeled", "*", "info.json")))))[i] for i in range(3, 8)]
    like_num["unlabeled"][1] = sum(like_num["unlabeled"][:2])
    like_num["unlabeled"].pop(0)
    like_num["labeled"] = [Counter(list(map(lambda x: digital_number(info_path=x, key="like_number"), glob(os.path.join(args.dataset_dir, "labeled", "*", "*", "info.json")))))[i] for i in range(3, 8)]
    like_num["labeled"][1] = sum(like_num["labeled"][:2])
    like_num["labeled"].pop(0)

    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        
        width = 0.3
        ax.bar(
            list(map(lambda x: x + 0.5 - width / 2, list(range(4))))
            , like_num["labeled"]
            , width=width
            , color=SCHEME[scheme][0]
            , zorder=100
            , label="labeled"
        )
        ax.bar(
            list(map(lambda x: x + 0.5 + width / 2, list(range(4))))
            , like_num["unlabeled"]
            , width=width
            , color=SCHEME[scheme][1]
            , zorder=100
            , label="unlabeled"
        )

        def like_xaxis_fmt(
            x: int
            , pos: int
        ):
            if x == 0:
                return "0.5k"
            elif 1 <= x <= 2:
                return f"{int(10 ** x)}k"
            else:
                return f"{int(10 ** (x - 3))}M"

        ax.set_xlabel("Number of Likes")
        ax.set_ylabel("Count")
        ax.set_xlim(left=0, right=4)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(like_xaxis_fmt))
        y_max = max(max(like_num["labeled"]), max(like_num["unlabeled"]))
        ax.set_ylim(bottom=0, top=int((y_max + 499) // 500) * 500)
        ax.yaxis.set_major_locator(plt.MultipleLocator(500))
        ax.grid(
            axis="y"
            , linestyle=(0, (5, 10))
            , linewidth=0.25
            , color="#4E616C"
            , zorder=0
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_color("none")
        ax.legend(loc="upper right")

        plt.savefig(os.path.join(args.figure_dir, "like_number", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "like_number", f"{scheme}.pdf"))
        plt.close()


def plot_comment_number(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "comment_number")):
        shutil.rmtree(os.path.join(args.figure_dir, "comment_number"))
    os.makedirs(os.path.join(args.figure_dir, "comment_number"))

    comment_number = {"labeled": [], "unlabeled": []}
    comment_number["unlabeled"] = [Counter(list(map(lambda x: digital_number(info_path=x, key="comment_number"), glob(os.path.join(args.dataset_dir, "unlabeled", "*", "info.json")))))[i] for i in range(1, 8)]
    comment_number["unlabeled"][2] = sum(comment_number["unlabeled"][:3])
    comment_number["unlabeled"][-2] = sum(comment_number["unlabeled"][-2:])
    comment_number["unlabeled"] = comment_number["unlabeled"][2:-1]
    comment_number["labeled"] = [Counter(list(map(lambda x: digital_number(info_path=x, key="comment_number"), glob(os.path.join(args.dataset_dir, "labeled", "*", "*", "info.json")))))[i] for i in range(3, 8)]
    comment_number["labeled"][-2] = sum(comment_number["labeled"][-2:])
    comment_number["labeled"].pop()

    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        
        width = 0.3
        ax.bar(
            list(map(lambda x: x + 0.5 - width / 2, list(range(4))))
            , comment_number["labeled"]
            , width=width
            , color=SCHEME[scheme][0]
            , zorder=100
            , label="labeled"
        )
        ax.bar(
            list(map(lambda x: x + 0.5 + width / 2, list(range(4))))
            , comment_number["unlabeled"]
            , width=width
            , color=SCHEME[scheme][1]
            , zorder=100
            , label="unlabeled"
        )

        def like_xaxis_fmt(
            x: int
            , pos: int
        ):
            if x == 0:
                return "0.1k"
            elif 1 <= x <= 3:
                return f"{int(10 ** (x - 1))}k"
            else:
                return f"{int(10 ** (x - 4))}M"

        ax.set_xlabel("Number of Comments")
        ax.set_ylabel("Count")
        ax.set_xlim(left=0, right=4)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(like_xaxis_fmt))
        y_max = max(max(comment_number["labeled"]), max(comment_number["unlabeled"]))
        ax.set_ylim(bottom=0, top=int((y_max + 499) // 500) * 500)
        ax.yaxis.set_major_locator(plt.MultipleLocator(500))
        ax.grid(
            axis="y"
            , linestyle=(0, (5, 10))
            , linewidth=0.25
            , color="#4E616C"
            , zorder=0
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_color("none")
        ax.legend(loc="upper right")

        plt.savefig(os.path.join(args.figure_dir, "comment_number", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "comment_number", f"{scheme}.pdf"))
        plt.close()


def main():
    args = get_args()
    plot_like_number(args=args)
    plot_comment_number(args=args)


if __name__ == "__main__":
    main()