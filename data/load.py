import re
import json

import numpy as np


def text_preprocess(
    text: str
):
    text = text.replace("作者赞过", "").replace("作者回复过", "").replace("置顶", "").replace("#搞笑", "").replace("#搞笑视频", "")
    text_seq = text.strip().split(" ")
    text = []
    for seg in text_seq:
        if re.match("^@.*", seg) is not None:
            seg = "@"
        if "@" in seg:
            seg = seg[:min(seg.index("@") + 1, len(seg))]
        if re.match("^#.*", seg) is not None or "#" in seg:
            seg = seg.replace("#", "")
        if seg != "":
            text.append(seg.strip())
    text = "".join(text)
    return text


def build_comment_graph(
    comment_block: dict
):
    comment_list = [text_preprocess(comment_block["content"])]
    last_user_idx = {
        comment_block["user"]: 0
    }
    n = len(comment_block["reply"]) + 1
    comment_graph = np.zeros(shape=[n, n], dtype=np.float32)
    for i, reply in enumerate(comment_block["reply"]):
        u = len(comment_list)
        comment_list.append(text_preprocess(reply["content"]))
        v = last_user_idx.get(reply["to"], 0)
        comment_graph[u][v] = 1.
        comment_graph[v][u] = 1.
        last_user_idx[reply["user"]] = u
    return comment_list, comment_graph


def load_comment(
    comment_path: str
):
    comment, adjacency = [], []
    for comment_block in sorted(json.loads(open(comment_path, "r", encoding="utf8").read()), key=lambda x: x["like_num"]):
        cmt, adj = build_comment_graph(comment_block=comment_block)
        comment.append(cmt)
        adjacency.append(adj)
    return comment, adjacency