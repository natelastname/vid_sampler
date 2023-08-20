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
import subprocess
from collections import namedtuple
import shutil

# magic = the python bindings for the library behind the `file` commmand.
from pymediainfo import MediaInfo
import magic
import ffmpeg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import geojson
from osgeo import gdal
from osgeo import osr

from typing import List

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
    # Check whether the file is partially downloaded or not
    ##################################################################

    if path.endswith(".part"):
        # This is not something that will likely be put there for ,
        # so for now we can go with a file extension check.
        logger.debug("Skipping, file download was not complete.")
        breakpoint()
        return None

    ##################################################################
    # Get the number of frames
    ##################################################################

    if "duration" in track and "frame_rate" in track:
        logger.debug("Computing number of frames via frame_rate * duration...")
        num_frames = float(track["frame_rate"]) * float(track["duration"])
        num_frames = int(num_frames)
    else:
        logger.debug("Using cv2 to compute number of frames...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            num_frames = get_num_frames(path)

    if num_frames <= 0:
        logger.debug("Could not compute the number of frames.")
        return None

    num_frames = int(num_frames)

    ##################################################################
    # Get the duration via num_frames * FPS, because CV2 allows us to
    # get the number of frames and the FPS but not the duration.
    ##################################################################

    duration_sec = None
    if "duration" in track:
        duration_sec = float(track["duration"]) / 1000
    else:
        logger.debug("Computing duration via CV2...")
        # If this fails I woulnd't be surprised
        fps = get_fps(path)
        duration_sec = (1/fps) * num_frames
        logger.debug(f"        FPS: {fps}")

    ts = datetime.timedelta(seconds=duration_sec)
    logger.debug(f"   Duration: {ts}")
    logger.debug(f"num. frames: {num_frames}")
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


def is_png_valid(path : str):
    """
    Use PIL to check if the given file is a valid PNG.
    """
    try:
        im = PIL.Image.open(path)
        im.verify()
        im.close()
    except Exception as ex:
        logger.warn(f"Produced exception '{type(ex)}'.")
        raise ex
        return False

    return True


def interval_intersection(interval1, interval2):
    new_min = max(interval1[0], interval2[0])
    new_max = min(interval1[1], interval2[1])
    return [new_min, new_max] if new_min <= new_max else None


def get_frame_numpy(vid: VidClass, frame_num: int):
    '''
    Load a frame as a numpy array.
    '''
    time_sec = (frame_num / vid.num_frames) * vid.duration_sec
    time_ms = round(time_sec * 1000)

    out, _ = (
        ffmpeg
        .input(vid.path, ss=f"{time_ms}ms")
        .output('pipe:',
                vframes=1,
                format='rawvideo',
                loglevel="quiet",
                pix_fmt='rgb24',
                **{'qscale:v': 1})
        .run(capture_stdout=True)
    )

    frame = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, vid.height, vid.width, 3])
    )

    if frame.shape[0] != 1:
        breakpoint()
        raise Exception("Somehow more than one frame was loaded.")
    if len(frame.shape) != 4:
        breakpoint()
        raise Exception("Unexpected number of dimensions.")
    if frame.shape[2] == vid.width and frame.shape[1] == vid.height:
        frame = np.transpose(frame, [0, 2, 1, 3])
    if frame.shape[1] == vid.width and frame.shape[2] == vid.height:
        return frame
    else:
        breakpoint()
        raise Exception("Frame was not reshaped correctly.")


def export_frame_png(vid: VidClass, frame_num: int, outfile: str):
    time_sec = (frame_num / vid.num_frames) * vid.duration_sec
    time_ms = round(time_sec * 1000)
    cmd = f'ffmpeg -y -ss {time_ms}ms -i "{vid.path}" -vframes 1 -qscale:v 1 "{outfile}" >/dev/null 2>&1'
    subprocess.call(cmd, shell=True)
    if not is_png_valid(outfile):
        breakpoint()
        logger.debug("Produced and invalid PNG.")
    return True


def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

