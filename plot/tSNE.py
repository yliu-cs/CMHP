import os
import pickle
import argparse
import warnings
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .scheme import SCHEME


def get_args():
    names = ["-".join(ckpt_dir.split("-")[1:]) for ckpt_dir in glob(os.path.join(os.getcwd(), "checkpoint", "finetune-*"))]

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+", type=str, default=names)
    parser.add_argument("--highlight_correct", action="store_true")
    parser.add_argument("--all_scheme", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    return args


def complete_args(
    args: argparse.Namespace
):
    args.pkl_path = os.path.join(os.getcwd(), "checkpoint", f"finetune-{args.name}", "tSNE.pkl")
    args.figure_dir = os.path.join(os.getcwd(), "figure", f"finetune-{args.name}")
    return args


def plot_tSNE(
    feature: np.ndarray
    , label: list
    , pred: list
    , args: argparse.Namespace
):
    if len(label) != len(pred):
        raise ValueError(f"{len(label)=} != {len(pred)=}")

    x = TSNE(
        n_components=2
        , random_state=42
    ).fit_transform(feature)

    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", family="Times New Roman")
        fig, ax = plt.subplots()
        if args.highlight_correct:
            colors = [SCHEME[scheme][label[i]] if label[i] == pred[i] else "#EAEAEA" for i in range(len(label))]
        else:
            colors = [SCHEME[scheme][label[i]] for i in range(len(label))]
        ax.scatter(
            x[:, 0]
            , x[:, 1]
            , s=10
            , c=colors
        )

        if not os.path.exists(os.path.join(args.figure_dir, "tSNE")):
            os.makedirs(os.path.join(args.figure_dir, "tSNE"))
        plt.savefig(os.path.join(args.figure_dir, "tSNE", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "tSNE", f"{scheme}.pdf"))
        plt.close()


def main():
    args = get_args()
    for name in args.name:
        args.name = name
        args = complete_args(args=args)
        pkl_data = pickle.load(open(args.pkl_path, "rb"))
        plot_tSNE(
            feature=pkl_data["feature"]
            , label=pkl_data["label"]
            , pred=pkl_data["pred"]
            , args=args
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()