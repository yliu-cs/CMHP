import os
import shutil
from glob import glob

import cv2
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt


LBL_DIR = "/home/yliu/work/yliu/VHD/dataset/labeled"
IMG_DIR = "/home/yliu/work/yliu/VHD/baseline/VideoBERT/src/I3D/img"

# VIDS = ['7088983837123955979', '7106834654946184456', '7112638109447179528', '7113356757715537192', '7114120596954320161', '7119056529885580578', '7122374966804319491', '7122568370322443559']
# VIDS = ['7122568370322443559']
VIDS = ['7124902667377233189', '7118643213153340676', '7110843105930349838', '7096638544826993934', '7073774189831802149', '7118973493349256487', '6972482233902976259', '7123883201600179463', '7121315476885589262', '7124576432646262020', '7097561847255043369', '7104934280916225292', '7115014558007282955', '7120808520076872971']


def extract_image_from_video(
    video_path: str
    , store_dir: str
):
    video_cap = cv2.VideoCapture(video_path)
    if video_cap.isOpened():
        cnt = 0
        rval, frame = video_cap.read()
        while rval:
            cnt += 1
            rval, frame = video_cap.read()
            if rval:
                cv2.imwrite(os.path.join(store_dir, f"{cnt}.png"), frame)
                cv2.waitKey(1)
            else:
                break
        video_cap.release()
    else:
        raise ValueError("VideoCapture Erorr!")


def load_audio(
    video_path: str
    , sample_rate: int = 16000
):
    streamer = torchaudio.io.StreamReader(src=video_path)
    streamer.add_basic_audio_stream(
        frames_per_chunk=sample_rate
        , sample_rate=sample_rate
    )

    audio_chunk_list = []
    for audio_chunk in streamer.stream():
        if audio_chunk is not None:
            audio_chunk_list.append(audio_chunk[0].float())
    audio = torch.cat(
        audio_chunk_list
        , dim=0
    ) # [frame, channel]
    if audio.dim() == 2:
        if 1 <= audio.size(1) <= 2:
            audio = torch.mean(audio, dim=1)
            # [frame]
        else:
            raise ValueError(f"{audio.size()=}")
    
    audio = audio.numpy()
    return audio


def plot_waveform(
    video_path: str
    , store_dir: str
    , sample_rate: int = 16000
):
    waveform = load_audio(video_path=video_path, sample_rate=sample_rate)

    num_frames = waveform.shape[0]
    time_axis = torch.arange(0, num_frames) / sample_rate

    fig, ax = plt.subplots()
    ax.patch.set_facecolor("#F7FBFD")
    # ax.patch.set_facecolor("#FFFFFF")
    ax.set_aspect(0.7)
    ax.plot(time_axis, waveform, linewidth=1)
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_color("none")
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()

    plt.savefig(os.path.join(store_dir, "waveform.png"), dpi=600)
    plt.close()


def main():
    if not os.path.exists(os.path.join(LBL_DIR, "intro")):
        os.mkdir(os.path.join(LBL_DIR, "intro"))
    for vid in tqdm(VIDS, ncols=100):
        raw_path = list(filter(lambda x: "intro" not in x, glob(os.path.join(LBL_DIR, "*", vid))))[0]

        if not os.path.exists(os.path.join(LBL_DIR, "intro", vid)):
            os.mkdir(os.path.join(LBL_DIR, "intro", vid))
        
        if not os.path.exists(os.path.join(LBL_DIR, "intro", vid, "video_img")):
            os.mkdir(os.path.join(LBL_DIR, "intro", vid, "video_img"))
            extract_image_from_video(os.path.join(raw_path, "video.mp4"), os.path.join(LBL_DIR, "intro", vid, "video_img"))
        #     shutil.copytree(
        #         os.path.join(IMG_DIR, vid)
        #         , os.path.join(LBL_DIR, "intro", vid, "video_img")
        #     )

        plot_waveform(os.path.join(raw_path, "video.mp4"), os.path.join(LBL_DIR, "intro", vid))

        for f in ("info.json", "comment.json", "video.mp4"):
            if not os.path.exists(os.path.join(LBL_DIR, "intro", vid, f)):
                shutil.copy(
                    os.path.join(raw_path, f)
                    , os.path.join(LBL_DIR, "intro", vid, f)
                )


if __name__ == "__main__":
    main()