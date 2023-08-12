#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-08-04T19:23:49-04:00

@author: nate
"""
import os
import re
import json
import logging.config
import warnings
import random
import datetime
from collections import namedtuple


# magic = the python bindings for the library behind the `file` commmand.
import magic
import ffmpeg
import cv2
import numpy as np
import matplotlib.pyplot as plt


from pymediainfo import MediaInfo


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "formatter_main": {
            "format": "[%(levelname)8s][%(name)s][%(filename)s:%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "formatter_main",
            "stream": "ext://sys.stdout"
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["stdout"]
    }
}
logging.config.dictConfig(logging_config)

logger = logging.getLogger(__name__)

VidClass = namedtuple("Video", "path, width, height, num_frames, duration_sec")


metadata_cache = {}
def get_metadata(path):
    global metadata_cache

    if path in metadata_cache:
        return metadata_cache[path]

    try:
        metadata = ffmpeg.probe(path)
    except Exception:
        metadata = None

    metadata_cache[path] = metadata
    return metadata


def get_resolution(path):
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return (width, height)


def get_fps(path):
    cap = cv2.VideoCapture(path)
    ret = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return ret


def get_num_frames(path):
    cap = cv2.VideoCapture(path)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total


# unused
def is_video_mime_ffprobe(path):
    # Use magic to infer via magic number (fast)
    with magic.Magic(flags=magic.MAGIC_MIME_TYPE) as mag:
        mime = mag.id_filename(path)

    if not mime.startswith("video"):
        return False

    # Verify with ffmpeg (slow)
    metadata = get_metadata(path)
    if not metadata:
        return False

    for i, stream in enumerate(metadata['streams']):
        if 'codec_type' not in stream:
            continue
        if stream['codec_type'] != "video":
            continue

        # The only format I've encountered that looks enough like
        # video to get it this far, yet still not be widely supported
        # enough to cause problems, are VOB files
        #
        # https://en.wikipedia.org/wiki/VOB
        #
        # It's a shame because it seems like these can contain good
        # footage.
        return True

    return False

def is_video_mediainfo(path):
    info = MediaInfo.parse(path, mediainfo_options={"File_TestContinuousFileNames": '0'})
    if len(info.video_tracks) == 0:
        return False

    # All other checks are intended to skip IFO files (usually called
    # VIDEO_TS.IFO, basically these handle menus on DVDs) and their
    # associated '.BUP' backup files.

    # This is the only format that isn't a usable video yet looks enough
    # like one to get this far.

    if len(info.general_tracks) != 1:
        return False

    if info.general_tracks[0].format_profile in ["Menu", "Program"]:
        return False

    return True

def augment_test(loader1, loader2):
    '''
    Example of how to display multiple pictures at once in matplotlib
    '''
    num_cols_figure = 5
    _, figs = plt.subplots(2, num_cols_figure, figsize=(15, 15))
    for i, (fig_top, fig_bottom) in enumerate(zip(figs[0], figs[1])):
        x1, y1 = loader1[i]
        x2, y2 = loader2[i]
        fig_top.imshow(x1)
        ax = fig_top.axes
        ax.set_title("Original")
        ax.title.set_fontsize(14)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        to_show = mx.nd.transpose(x2, (1, 2, 0))
        fig_bottom.imshow(to_show.asnumpy())
        ax = fig_bottom.axes
        ax.set_title("Modified")
        ax.title.set_fontsize(14)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def get_video_stats(path):
    info = MediaInfo.parse(path)
    track = info.video_tracks[0].to_data()
    width = track["width"]
    height = track["height"]
    num_frames = None

    vidname = os.path.basename(path)
    logger.debug("====================================")
    logger.debug(f"{vidname}")

    ##################################################################
    # Get the number of frames
    ##################################################################

    if "duration" in track and "frame_rate" in track:
        logger.debug("Computing number of frames via frame_rate * duration...")
        num_frames = float(track["frame_rate"]) * float(track["duration"])
        num_frames = int(num_frames)
    else:
        logger.debug("Using cv2 to compute number of frames...")
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            num_frames = get_num_frames(item)
        '''
        num_frames = get_num_frames(path)

    if num_frames <= 0:
        logger.debug("Could not compute the number of frames.")
        return None

    num_frames = int(num_frames)

    ##################################################################
    # Get the duration (via num_frames * FPS, because CV2 allows us to
    # get the number of frames and the FPS but not the duration
    # directly.
    ##################################################################

    duration_sec = None
    if "duration" in track:
        duration_sec = float(track["duration"]) / 1000
    else:
        logger.debug("Computing duration via CV2...")
        # If this fails I woulnd't be surprised
        fps = get_fps(path)
        logger.debug(f"FPS: {fps}")
        duration_sec = (1/fps) * num_frames

    ts = datetime.timedelta(seconds=duration_sec)
    logger.debug(f"Duration: {ts}")

    vid = VidClass(path=path,
                   width=int(width),
                   height=int(height),
                   num_frames=num_frames,
                   duration_sec=duration_sec)
    return vid


def crawl_folder(path, is_video_callback):
    '''
    Recursively search `path` for video files, return a list of
    absolute paths.
    '''
    video_files = []
    for root, dirs, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not is_video_callback(fpath):
                continue
            video_files.append(fpath)
    return sorted(video_files)


def sample_frame_uniform(data):
    frame_counts = map(lambda x: int(x.num_frames), data)
    video = random.sample(data, 1, counts=frame_counts)[0]
    ret = (video, random.randint(0, video.num_frames))
    return ret


def export_frame(vid: VidClass, frame_num: int):
    cap = cv2.VideoCapture(vid.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        logger.debug("Couldn't get frame via cv2.CAP_PROP_POS_FRAMES.")
    else:
        return frame

    time_ms = (frame_num * (vid.num_frames / vid.duration_sec)) * 1000

    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ret, frame = cap.read()

    if not ret:
        breakpoint()
        logger.debug("Couldn't get frame via cv2.CAP_PROP_POS_MSEC.")
        return None

    return frame

if __name__ == "__main__":

    path = "/mnt/2TBSSD/01_nate_datasets/movies"

    logger.debug(f"Crawling {path} using `is_video_mediainfo`...")
    vids = crawl_folder(path, is_video_mediainfo)
    data = []
    for i, item in enumerate(vids):
        # If here, item is a video file that VLC could probably play.
        # The prom is that not all video formats store the total
        # number of frames in a header, and if they don't we cannot
        # always compute the total number frames using the strategies
        # that are currently implemented.
        name = os.path.split(item)[-1]
        res = get_video_stats(item)
        if res:
            data.append(res)

    os.makedirs("./output", exist_ok=True)

    for i in range(0, 64):
        (video, frame_num) = sample_frame_uniform(data)
        vidname = os.path.basename(video.path)
        logging.debug(f"{i:5}: {vidname}")
        logging.debug(f"Frame: {frame_num}/{video.num_frames}")
        frame = export_frame(video, frame_num)



    breakpoint()
    print("done")

    # It'd be preferable to have equal probability per frame rather
    # than unit time.
