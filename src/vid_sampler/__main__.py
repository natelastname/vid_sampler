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


def args_has_all_not_some(namespace, attrs: List[str]):
    """
    Returns True if it has all, False if it has none. Otherwise (i.e.,
    if namespace contains some of the attributes), an exception is
    raised.
    """
    if attrs_has_all_or_none(namespace, attrs) == False:
        raise Exception(f"Must provide all of arguments: {attrs}")

    return has_attr_not_none(namespace, attrs[0])

######################################################################
# Set up ArgumentParser
######################################################################


desc = ("Recursively discover videos in a directory and randomly"
        "sample frames from them.")
parser = argparse.ArgumentParser(prog="vid_sampler", description=desc)

######################################################################
# Always required
######################################################################

parser.add_argument("--input-dir",
                    help="Directory containing videos to sample from.",
                    type=pathlib.Path)

parser.add_argument("--output-dir",
                    help="Directory where output files are written.",
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
                    default=None,
                    action='store_true')

gt_help = ("JSON array with 6 floating point elements specifying the"
           "geotransform. See:"
           "https://gdal.org/tutorials/geotransforms_tut.html")

parser.add_argument("--geotransform",
                    help=gt_help,
                    type=str)

parser.add_argument("--lon-res-px",
                    help="x resolution of the geotiff.",
                    type=int)
parser.add_argument("--lat-res-px",
                    help="y resolution of the geotiff.",
                    type=int)


######################################################################
# Required when outputting frames
######################################################################

parser.add_argument("--output-frames",
                    help="Directory to write the output images to.",
                    default=None,
                    action="store_true")

######################################################################
# A utility feature
######################################################################


# EXAMPLE:
#    vid_sampler --simple-geotransform "[12288, 12288, -16, 16, 32, 32]"

help_str = ("Convert json array of the form [x_res_px, y_res_px,"
            "upper_left_lon, upper_left_lat, width_lon, width_lat]"
            "to a geotransform.")

parser.add_argument("--simple-geotransform",
                    help=help_str,
                    type=str)

######################################################################


args = parser.parse_args()

gen_simple_gt = args_has_all_not_some(args, ["simple_geotransform"])

write_frames = args_has_all_not_some(args, ["output_frames"])

write_geotiff = args_has_all_not_some(args, ["output_geotiff",
                                             "geotransform",
                                             "lon_res_px",
                                             "lat_res_px"])

if write_frames or write_geotiff:
    args_has_all_not_some(args, ["input_dir", "output_dir"])


if gen_simple_gt:
    simple_gt = json.loads(args.simple_geotransform)
    if isinstance(simple_gt, list) == False or len(simple_gt) != 6:
        print("--simple-geotransform expects JSON array of length 6.")
        sys.exit(1)

    gt = vs.simple_geotransform(*simple_gt)
    print(json.dumps(gt.gt))
    sys.exit(0)


# Perform any argument validation/parsing here so that it fails quickly.
img_params = None
if write_geotiff:
    gt = json.loads(args.geotransform)
    img_params = vs.GeotiffParams(args.lon_res_px, args.lat_res_px, gt)

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

logger.info(f"Discovering videos in {args.input_dir}...")

vids = vs.crawl_folder(args.input_dir, vs.is_video_mediainfo)
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
    logger.info("No usable videos found.")
    sys.exit(0)



######################################################################
# Sample frames
######################################################################


logger.info("Sampling frames...")
os.makedirs(args.output_dir, exist_ok=True)

samples = []
for i in range(0, args.num_frames):
    frame = vs.sample_frame_uniform(usable_vids)
    samples.append(frame)

    vidname = os.path.basename(frame.vid.path)
    logging.debug(f"{i:5}: {vidname}")
    logging.debug(f"Frame: {frame.frame_num}/{frame.vid.num_frames}")

    if write_frames == False:
        continue

    outfile = os.path.join(args.output_dir, f"{i}.png")
    vs.export_frame_png(frame.vid, frame.frame_num, outfile)



if write_geotiff == False:
    sys.exit(0)


logger.info("Generating geotiff collage...")

# TODO: Option to only place over pixels that have no data (maybe)
# TODO: Use same images for geotiff and frame output
# TODO: Options for outputting point log as geojson
# TODO: implement log levels
# TODO: Implement --geotransform option


vs.geotiff_collage(os.path.join(args.output_dir, "out.tif"),
                   img_params,
                   usable_vids,
                   args.num_frames,
                   samples=samples)

print("done")
