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

# magic = the python bindings for the library behind the `file` commmand.
import magic
import ffmpeg
import cv2

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
            "level": "INFO",
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

def crawl_folder(path, is_video_callback):
    '''
    Recursively search `path` for video files, return a list of absolute
    paths.
    '''
    video_files = []
    for root, dirs, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not is_video_callback(fpath):
                continue
            video_files.append(fpath)

    return sorted(video_files)


if __name__ == "__main__":
    path = "/mnt/2TBSSD/01_nate_datasets/movies/"

    #print("Crawling with is_video_mime_ffprobe...")
    #vids1 = crawl_folder(path, is_video_mime_ffprobe)

    print("Crawling with is_video_mediainfo...")
    vids = crawl_folder(path, is_video_mediainfo)

    for i, item in enumerate(vids):
        name = os.path.split(item)[-1]
        print("========================================")
        print(f"{i:5}: {name}")
        print(item)
        info = MediaInfo.parse(item)
        if len(info.video_tracks) != 1:
            logger.info("Cannot handle file with {len(info.video_tracks)} video tracks")
            continue

        track = info.video_tracks[0]

        track.format
        track.duration
        track.frame_rate
        track.bit_depth
        track.width
        track.height


        print("=====")
                print(f"    Format: {track.format}")
                print(f"  Duration: {track.duration}")
                print(get_num_frames(item)/float(track.frame_rate))

                print(f"Frame rate: {track.frame_rate}")
                print(f" Bit depth: {track.bit_depth}")
                print(f"Resolution: {track.width} x {track.height}")


    print("done")
    # Probability should be weighted by number of frames in the file
    # so that every frame is equally likely
