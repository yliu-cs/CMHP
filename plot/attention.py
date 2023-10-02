import os
import pickle
import argparse
import warnings
from glob import glob

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from .scheme import ATTENTION_CMAP


def get_args():
    names = ["-".join(ckpt_dir.split("-")[1:]) for ckpt_dir in glob(os.path.join(os.getcwd(), "checkpoint", "finetune-*"))]

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+", type=str, default=names)
    parser.add_argument("--all_scheme", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    return args


def complete_args(
    args: argparse.Namespace
):
    args.pkl_path = os.path.join(os.getcwd(), "checkpoint", f"finetune-{args.name}", "attn.pkl")
    args.figure_dir = os.path.join(os.getcwd(), "figure", f"finetune-{args.name}")
    return args


def plot_attention(
    attention: np.ndarray
    , args: argparse.Namespace
):
    for scheme in ATTENTION_CMAP if args.all_scheme else ATTENTION_CMAP[:1]:
        plt.rc("font", family="Times New Roman")
        fig, ax = plt.subplots()

        viridis = matplotlib.colormaps[scheme].resampled(256)
        newcolors = viridis(np.linspace(0.5, 1., 256))
        cmap = matplotlib.colors.ListedColormap(newcolors)
        im = ax.imshow(
            X=attention
            , cmap=cmap
            , vmin=0.
            , vmax=0.025
        )
        ax.set_xticks([0, 301, 602, 604, 615])
        ax.set_yticks([0, 301, 602, 604, 615])
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([])
        # cbar.set_ticks([_ * 0.005 for _ in range(0, 4)])
        fig.tight_layout()

        if not os.path.exists(os.path.join(args.figure_dir, "attn")):
            os.makedirs(os.path.join(args.figure_dir, "attn"))
        plt.savefig(os.path.join(args.figure_dir, "attn", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "attn", f"{scheme}.pdf"))
        plt.close()


def main():
    args = get_args()
    for name in args.name:
        args.name = name
        args = complete_args(args=args)
        plot_attention(
            attention=pickle.load(open(args.pkl_path, "rb"))
            , args=args
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()