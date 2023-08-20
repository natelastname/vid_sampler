#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-08-19T18:54:33-04:00

@author: nate
"""
import os
import re
import sys
import json
import logging.config
import warnings
import random
import datetime
import subprocess
from collections import namedtuple
import shutil
import argparse
import pathlib
from typing import List

# magic = the python bindings for the library behind the `file` commmand.
from pymediainfo import MediaInfo
import magic
import ffmpeg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from osgeo import gdal
from osgeo import osr
import tqdm

import vid_sampler as vs

desc = ("Recursively discover videos in a directory and randomly"
        "sample frames from them.")
parser = argparse.ArgumentParser(prog="vid_sampler", description=desc)

######################################################################
# Always required
######################################################################

parser.add_argument("basedir",
                    help="Directory containing videos to sample from.",
                    type=pathlib.Path)

# We can treat this as mandatory because it has a default value.
parser.add_argument("--num-frames",
                    help="Number of frames to sample.",
                    default="1",
                    type=int)

######################################################################
# Required when writing a Geotiff
######################################################################

parser.add_argument("--output-geotiff",
                    help="Generate a geotiff collage.",
                    type=argparse.FileType("w+"))

gt_help = ("JSON array with 6 floating point elements specifying the"
           "geotransform. See:"
           "https://gdal.org/tutorials/geotransforms_tut.html")

parser.add_argument("--geotransform",
                    help=gt_help,
                    type=str)

######################################################################
# Required when outputting frames
######################################################################

parser.add_argument("--output-frames",
                    help="Directory to write the output images to.",
                    type=pathlib.Path)

######################################################################


def has_attr_not_none(namespace, attr: str):
    """
    Return true if namespace has attribute attr and its
    value is not None. Otherwise, return false.
    """
    if hasattr(namespace, attr) and getattr(namespace, attr) != None:
        return True
    return False


def attrs_has_all_or_none(namespace, attrs: List[str]):
    """
    Return True if namespace has all atributes in list attrs.
    Otherwise, return false.
    """
    if len(attrs) == 0:
        return True

    first = has_attr_not_none(namespace, attrs[0])
    for attr in attrs:
        if has_attr_not_none(namespace, attr) != first:
            return False

    return True


def args_has_all_or_none(namespace, attrs: List[str]):
    """
    Returns True if it has all, False if it has none.
    Otherwise (i.e., if namespace contains some of the attributes), an
    exception is raised.
    """
    if attrs_has_all_or_none(namespace, attrs) == False:
        raise Exception(f"Must provide all of arguments: {attrs}")

    return has_attr_not_none(namespace, attrs[0])



args = parser.parse_args()
write_frames = args_has_all_or_none(args, ["output_frames"])
write_geotiff = args_has_all_or_none(args, ["output_geotiff", "geotransform"])


######################################################################
# Set up logging
######################################################################

# Although it's good that PIL properly uses the logging module,
# I still want to turn it off.
logging.getLogger('PIL').setLevel(logging.WARNING)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "formatter_main": {
            "format": ("[%(levelname)8s][%(name)s][%(filename)s:%(lineno)d] "
                       "%(message)s")
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

######################################################################
# Discover videos
######################################################################

#path = "/mnt/2TBSSD/01_nate_datasets/movies"
#path = "/mnt/2TBSSD/01_nate_datasets/movies_small"
# TODO: Don't hard code
output_dir = "/home/nate/spyder_projects/vid_sampler/output/"

logger.info(f"Discovering videos in {args.basedir}...")

vids = vs.crawl_folder(args.basedir, vs.is_video_mediainfo)
usable_vids = []

logger.info("Discarding videos that cannot be sampled from...")
for i, item in enumerate(vids):
    # If here, item is a video file that VLC could probably play.
    # The problem is that not all video formats store the total
    # number of frames in a header, and if they don't we cannot
    # always compute the total number frames using the strategies
    # that are currently implemented.
    name = os.path.split(item)[-1]
    res = vs.get_video_stats(item)
    if res:
        usable_vids.append(res)


logger.info(f"Done. (Kept {len(usable_vids)}/{len(vids)} files)")

if len(usable_vids) == 0:
    sys.exit(0)

logger.info("Sampling frames...")
os.makedirs(output_dir, exist_ok=True)

for i in range(0, args.num_frames):
    (video, frame_num) = vs.sample_frame_uniform(usable_vids)
    vidname = os.path.basename(video.path)
    logging.debug(f"{i:5}: {vidname}")
    logging.debug(f"Frame: {frame_num}/{video.num_frames}")
    outfile = os.path.join(output_dir, f"{i}.png")
    frame = vs.export_frame_png(video, frame_num, outfile)

if write_geotiff == False:
    sys.exit(0)

logger.info("Generating geotiff collage...")

x_res = 4096*3
y_res = x_res
upper_left_x = -16
upper_left_y = 16
width_x = 32
width_y = 32
img_params = vs.simple_geotransform(x_res, y_res, upper_left_x, upper_left_y, width_x, width_y)

vs.geotiff_collage(args.output_geotiff.name, img_params, usable_vids, args.num_frames)

print("done")
