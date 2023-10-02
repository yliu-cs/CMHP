import os
import json
import shutil
import argparse
from glob import glob
from collections import Counter

import cv2
import numpy as np
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


def get_video_duration(
    video_file_path: str
):
    video_cap = cv2.VideoCapture(video_file_path)
    if video_cap.isOpened():
        fps, frame_count = video_cap.get(cv2.CAP_PROP_FPS), video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        return duration
    else:
        raise ValueError(f"{video_file_path=}.isOpened()={video_cap.isOpened()}")


def plot_unlabeled_duration(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "duration", "unlabeled")):
        shutil.rmtree(os.path.join(args.figure_dir, "duration", "unlabeled"))
    os.makedirs(os.path.join(args.figure_dir, "duration", "unlabeled"))
    
    duration = [Counter(list(map(lambda x: int(x // 10), list(map(get_video_duration, glob(os.path.join(args.dataset_dir, "unlabeled", "*", "video.mp4")))))))[i] for i in range(6)]
    duration = duration[:3] + [sum(duration[3:])]
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        ax.pie(
            duration
            , autopct="%.2f%%"
            , labels=["5~10s"] + [f"{i * 10}~{(i + 1) * 10}s" for i in range(1, 3)] + ["30~60s"]
            , colors=SCHEME[scheme]
        )

        plt.savefig(os.path.join(args.figure_dir, "duration", "unlabeled", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "duration", "unlabeled", f"{scheme}.pdf"))
        plt.close()


def plot_labeled_duration(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "duration", "labeled")):
        shutil.rmtree(os.path.join(args.figure_dir, "duration", "labeled"))
    os.makedirs(os.path.join(args.figure_dir, "duration", "labeled"))

    duration = {"humor": [], "non-humor": []}
    for video_file in glob(os.path.join(args.dataset_dir, "labeled", "*", "*", "video.mp4")):
        if json.loads(open(os.sep.join(video_file.split(os.sep)[:-1] + ["info.json"]), "r", encoding="utf8").read())["humor"]:
            duration["humor"].append(video_file)
        else:
            duration["non-humor"].append(video_file)
    for key in duration.keys():
        duration[key] = [Counter(list(map(lambda x: int(x // 10), list(map(get_video_duration, duration[key])))))[i] for i in range(6)]
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        
        width = 0.5
        ax.bar(
            list(map(lambda x: x + 0.5, list(range(6))))
            , duration["humor"]
            , width=width
            , color=SCHEME[scheme][0]
            , zorder=100
            , label="humor"
        )
        ax.bar(
            list(map(lambda x: x + 0.5, list(range(6))))
            , duration["non-humor"]
            , bottom=duration["humor"]
            , width=width
            , color=SCHEME[scheme][1]
            , zorder=100
            , label="non-humor"
        )

        ax.set_xlabel("Duration")
        ax.set_ylabel("Count")
        ax.set_xlim(left=0, right=6)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(5 if x == 0 else x * 10)}s"))
        ax.set_ylim(bottom=0, top=int((max([duration["humor"][i] + duration["non-humor"][i] for i in range(6)]) + 49) // 50) * 50)
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
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

        plt.savefig(os.path.join(args.figure_dir, "duration", "labeled", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "duration", "labeled", f"{scheme}.pdf"))
        plt.close()


def plot_test_duration(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "duration", "test")):
        shutil.rmtree(os.path.join(args.figure_dir, "duration", "test"))
    os.makedirs(os.path.join(args.figure_dir, "duration", "test"))

    duration = {"humor": [], "non-humor": []}
    for video_file in glob(os.path.join(args.dataset_dir, "labeled", "test", "*", "video.mp4")):
        if json.loads(open(os.sep.join(video_file.split(os.sep)[:-1] + ["info.json"]), "r", encoding="utf8").read())["humor"]:
            duration["humor"].append(video_file)
        else:
            duration["non-humor"].append(video_file)
    for key in duration.keys():
        duration[key] = [Counter(list(map(lambda x: int(x // 10), list(map(get_video_duration, duration[key])))))[i] for i in range(6)]
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        
        width = 0.5
        ax.bar(
            list(map(lambda x: x + 0.5, list(range(6))))
            , duration["humor"]
            , width=width
            , color=SCHEME[scheme][0]
            , zorder=100
            , label="test set humor"
        )
        ax.bar(
            list(map(lambda x: x + 0.5, list(range(6))))
            , duration["non-humor"]
            , bottom=duration["humor"]
            , width=width
            , color=SCHEME[scheme][1]
            , zorder=100
            , label="test set non-humor"
        )

        ax.set_xlabel("Duration")
        ax.set_ylabel("Count")
        ax.set_xlim(left=0, right=6)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(5 if x == 0 else x * 10)}s"))
        ax.set_ylim(bottom=0, top=int((max([duration["humor"][i] + duration["non-humor"][i] for i in range(6)]) + 49) // 50) * 50)
        ax.yaxis.set_major_locator(plt.MultipleLocator(50))
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

        plt.savefig(os.path.join(args.figure_dir, "duration", "test", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "duration", "test", f"{scheme}.pdf"))
        plt.close()


def plot_train_val_duration(
    args: argparse.Namespace
):
    if os.path.exists(os.path.join(args.figure_dir, "duration", "train_val")):
        shutil.rmtree(os.path.join(args.figure_dir, "duration", "train_val"))
    os.makedirs(os.path.join(args.figure_dir, "duration", "train_val"))

    train_duration = {"humor": [], "non-humor": []}
    for video_file in glob(os.path.join(args.dataset_dir, "labeled", "train", "*", "video.mp4")):
        if json.loads(open(os.sep.join(video_file.split(os.sep)[:-1] + ["info.json"]), "r", encoding="utf8").read())["humor"]:
            train_duration["humor"].append(video_file)
        else:
            train_duration["non-humor"].append(video_file)
    for key in train_duration.keys():
        train_duration[key] = [Counter(list(map(lambda x: int(x // 10), list(map(get_video_duration, train_duration[key])))))[i] for i in range(6)]

    val_duration = {"humor": [], "non-humor": []}
    for video_file in glob(os.path.join(args.dataset_dir, "labeled", "val", "*", "video.mp4")):
        if json.loads(open(os.sep.join(video_file.split(os.sep)[:-1] + ["info.json"]), "r", encoding="utf8").read())["humor"]:
            val_duration["humor"].append(video_file)
        else:
            val_duration["non-humor"].append(video_file)
    for key in val_duration.keys():
        val_duration[key] = [Counter(list(map(lambda x: int(x // 10), list(map(get_video_duration, val_duration[key])))))[i] for i in range(6)]
    
    for scheme in list(SCHEME.keys()) if args.all_scheme else list(SCHEME.keys())[:1]:
        plt.rc("font", **FONT)
        fig, ax = plt.subplots()
        
        width = 0.3
        ax.bar(
            list(map(lambda x: x + 0.5 - width / 2, list(range(6))))
            , train_duration["humor"]
            , width=width
            , color=SCHEME[scheme][0]
            , zorder=100
            , label="train set humor"
        )
        ax.bar(
            list(map(lambda x: x + 0.5 - width / 2, list(range(6))))
            , train_duration["non-humor"]
            , bottom=train_duration["humor"]
            , width=width
            , color=SCHEME[scheme][1]
            , zorder=100
            , label="train set non-humor"
        )
        ax.bar(
            list(map(lambda x: x + 0.5 + width / 2, list(range(6))))
            , val_duration["humor"]
            , width=width
            , color=SCHEME[scheme][2]
            , zorder=100
            , label="val set humor"
        )
        ax.bar(
            list(map(lambda x: x + 0.5 + width / 2, list(range(6))))
            , val_duration["non-humor"]
            , bottom=val_duration["humor"]
            , width=width
            , color=SCHEME[scheme][3]
            , zorder=100
            , label="val set non-humor"
        )

        ax.set_xlabel("Duration")
        ax.set_ylabel("Count")
        ax.set_xlim(left=0, right=6)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(5 if x == 0 else x * 10)}s"))
        y_max = max([train_duration["humor"][i] + train_duration["non-humor"][i] for i in range(6)] + [val_duration["humor"][i] + val_duration["non-humor"][i] for i in range(6)])
        ax.set_ylim(bottom=0, top=int((y_max + 4) // 5) * 5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
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

        plt.savefig(os.path.join(args.figure_dir, "duration", "train_val", f"{scheme}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(args.figure_dir, "duration", "train_val", f"{scheme}.pdf"))
        plt.close()


def main():
    args = get_args()
    plot_unlabeled_duration(args=args)
    plot_labeled_duration(args=args)
    plot_test_duration(args=args)
    plot_train_val_duration(args=args)


if __name__ == "__main__":
    main()