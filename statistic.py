import os
from glob import glob
from argparse import ArgumentParser

from utils.misc import ignore_warnings


def get_args():

    parser = ArgumentParser()
    parser.add_argument("--log_path", type=str, default=os.path.join(os.getcwd(), "log", "finetune-*"))
    parser.add_argument("--sort_dep", type=str, default="acc", help="acc iter")
    args = parser.parse_args()
    
    return args


def ckpt_test(
    args
):
    ckpt_acc = []
    for log_path in glob(args.log_path):
        test_flag, ckpt_path, acc = False, None, None
        for line in open(log_path, "r").readlines():
            line = line.strip()[line.index("INFO : ") + len("INFO : "):]
            if line.find("ckpt_path") != -1:
                ckpt_path = line[line.find("=") + 1:]
            elif line == ("Test".center(50, "=")):
                test_flag = True
            elif test_flag and line.find("Acc=") != -1:
                acc = float(line[line.index("Acc=") + len("Acc="):])
        if ckpt_path is not None and acc is not None:
            ckpt_acc.append([log_path.split(os.sep)[-1], ckpt_path, acc])
    if args.sort_dep == "acc":
        ckpt_acc = sorted(ckpt_acc, key=lambda x: x[2])
    elif args.sort_dep == "iter":
        ckpt_acc = sorted(
            ckpt_acc
            , key=lambda x: (
                int(x[1].split(os.sep)[-2].split("-")[-3].split(".")[-1])
                , int(x[1].split(os.sep)[-2].split("-")[-1])
                , int(x[1].split("_")[-1].split(".")[0])
            )
        )
    for name, ckpt, acc in ckpt_acc:
        print(f"{name=} {ckpt=} {acc=}")


def main():
    args = get_args()
    ckpt_test(args=args)


if __name__ == "__main__":
    ignore_warnings()
    main()