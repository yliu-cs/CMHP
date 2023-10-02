import os
import json
import shutil
import argparse
from glob import glob

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


def get_comment_number(
    comment_path: str
):
    comment = json.loads(open(comment_path, "r", encoding="utf8").read())
    return len(comment) + sum(list(map(lambda x: len(x["reply"]), comment)))


def plot_comment_statistic(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "comment")):
        shutil.rmtree(os.path.join(args.figure_dir, "comment"))
    os.makedirs(os.path.join(args.figure_dir, "comment"))

    comment_number = []
    comment_number.append(list(map(get_comment_number, glob(os.path.join(args.dataset_dir, "unlabeled", "*", "comment.json")))))
    comment_number.append(list(map(get_comment_number, glob(os.path.join(args.dataset_dir, "labeled", "*", "*", "comment.json")))))
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()

        bplot = ax.boxplot(
            comment_number
            , positions=[2, 4]
            , widths=0.8
            , patch_artist=True
            , showmeans=False
            , showfliers=False
            , medianprops={
                "color": SCHEME[scheme][4]
                , "linewidth": 1.5
            }
            , boxprops={
                "edgecolor": SCHEME[scheme][3]
                , "linewidth": 1.5
            }
            , whiskerprops={
                "color": SCHEME[scheme][3]
                , "linewidth": 1.5
            }
            , capprops={
                "color": SCHEME[scheme][3]
                , "linewidth": 1.5
            }
            , labels=["unlabeled", "labeled"]
        )
        for patch, color in zip(bplot["boxes"], SCHEME[scheme]):
            patch.set_facecolor(color)

        ax.set_ylabel("Count")
        ax.set_xlim(left=0, right=6)
        ax.set_ylim(bottom=0, top=120)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax.grid(
            axis="y"
            , linestyle=(0, (5, 10))
            , linewidth=0.25
            , color="#4E616C"
            , zorder=0
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_color("none")

        plt.savefig(os.path.join(args.figure_dir, "comment", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "comment", f"{scheme}.pdf"))
        plt.close()


def plot_labeled_comment_statistic(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "labeled_comment")):
        shutil.rmtree(os.path.join(args.figure_dir, "labeled_comment"))
    os.makedirs(os.path.join(args.figure_dir, "labeled_comment"))

    comment_number = []
    for mode in ["train", "val", "test"]:
        comment_number.append(list(map(get_comment_number, glob(os.path.join(args.dataset_dir, "labeled", mode, "*", "comment.json")))))
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()

        bplot = ax.boxplot(
            comment_number
            , positions=[1, 3, 5]
            , widths=0.8
            , patch_artist=True
            , showmeans=False
            , showfliers=False
            , medianprops={
                "color": SCHEME[scheme][4]
                , "linewidth": 1.5
            }
            , boxprops={
                "edgecolor": SCHEME[scheme][3]
                , "linewidth": 1.5
            }
            , whiskerprops={
                "color": SCHEME[scheme][3]
                , "linewidth": 1.5
            }
            , capprops={
                "color": SCHEME[scheme][3]
                , "linewidth": 1.5
            }
            , labels=["train", "val", "test"]
        )
        for patch, color in zip(bplot["boxes"], SCHEME[scheme]):
            patch.set_facecolor(color)

        ax.set_ylabel("Count")
        ax.set_xlim(left=0, right=6)
        ax.set_ylim(bottom=0, top=120)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax.grid(
            axis="y"
            , linestyle=(0, (5, 10))
            , linewidth=0.25
            , color="#4E616C"
            , zorder=0
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_color("none")

        plt.savefig(os.path.join(args.figure_dir, "labeled_comment", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "labeled_comment", f"{scheme}.pdf"))
        plt.close()


def main():
    args = get_args()
    plot_comment_statistic(args=args)
    plot_labeled_comment_statistic(args=args)


if __name__ == "__main__":
    main()