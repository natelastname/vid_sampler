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
from osgeo import gdal
from osgeo import osr


# Although it's good that PIL properly uses the logging module,
# I still want to turn it off.
logging.getLogger('PIL').setLevel(logging.WARNING)

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
        logger.debug(f"Produced exception '{type(ex)}'.")
        raise ex
        return False

    return True


def get_frame_numpy(vid: VidClass, frame_num: int, outfile: str):
    '''
    Load a frame as a numpy array.
    '''
    time_sec = (frame_num / vid.num_frames) * vid.duration_sec
    time_ms = round(time_sec * 1000)

    out, _ = (
        ffmpeg
        .input(vid.path, ss=f"{time_ms}ms")
        .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24', **{'qscale:v': 1})
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

    return True


def export_frame_png(vid: VidClass, frame_num: int, outfile: str):
    time_sec = (frame_num / vid.num_frames) * vid.duration_sec
    time_ms = round(time_sec * 1000)
    cmd = f'ffmpeg -y -ss {time_ms}ms -i "{vid.path}" -vframes 1 -qscale:v 1 "{outfile}" >/dev/null 2>&1'
    subprocess.call(cmd, shell=True)
    if not is_png_valid(outfile):
        breakpoint()
        logger.debug("Produced and invalid PNG.")

    return True

if __name__ == "__main__":

    #path = "/mnt/2TBSSD/01_nate_datasets/movies"
    path = "/mnt/2TBSSD/01_nate_datasets/movies_small"
    output_dir = "/home/nate/spyder_projects/vid_sampler/output/"

    logger.info(f"Crawling {path} using `is_video_mediainfo`...")
    vids = crawl_folder(path, is_video_mediainfo)
    data = []

    logger.info("Compiling metadata...")
    for i, item in enumerate(vids):
        # If here, item is a video file that VLC could probably play.
        # The problem is that not all video formats store the total
        # number of frames in a header, and if they don't we cannot
        # always compute the total number frames using the strategies
        # that are currently implemented.
        name = os.path.split(item)[-1]
        res = get_video_stats(item)
        if res:
            data.append(res)

    logger.info("Sampling frames and writing PNGs...")

    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, 8):
        (video, frame_num) = sample_frame_uniform(data)
        vidname = os.path.basename(video.path)
        logging.debug(f"{i:5}: {vidname}")
        logging.debug(f"Frame: {frame_num}/{video.num_frames}")
        outfile = os.path.join(output_dir, f"{i}.png")
        frame = export_frame_png(video, frame_num, outfile)

    ##################################################################
    # Sparse matrices are an option for Geotiff generation:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    # https://stackoverflow.com/questions/14855177/write-data-to-geotiff-using-gdal-without-creating-data-array?rq=3
    # ImageMagick can open geotiff:
    # https://gis.stackexchange.com/questions/268961/looking-for-an-openev-replacement-i-e-a-fast-geotiff-viewer
    # Gdal documentation:
    # https://gdal.org/api/python/osgeo.gdal.html
    ##################################################################
    # Ideas:
    # - No matter what, it has to have one edge touching another edge?
    #     - No, we want a tweakable amount of intersections (maybe even set the distribution?)
    # - Why not just start with uniform cloud of points (defined by num points per unit area in px)
    #     - This would be a good place to start.
    ##################################################################

    x_res = 4096
    y_res = 4096

    output_fname = 'test.tif'
    driver = gdal.GetDriverByName('GTiff')

    # This writes the file.
    num_bands = 1
    dst_ds = driver.Create(output_fname,
                           x_res,
                           y_res,
                           num_bands,
                           gdal.GDT_Float32)

    # Set raster's projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(srs.ExportToWkt())
    #print(dst_ds.GetProjection())
    #breakpoint()

    ##################################################################
    # Parameters of geotransform
    # NOTE: the upper left of the map is -x, +y
    # See: https://gdal.org/tutorials/geotransforms_tut.html
    ##################################################################
    upper_left_x = -16
    upper_left_y = 16
    width_x = 32
    width_y = 32
    ##################################################################
    pixel_width_ns = width_x / x_res
    pixel_width_we = width_y / y_res
    geotransform = [upper_left_x,
                    pixel_width_we,
                    0.0,
                    upper_left_y,
                    0.0,
                    -1*pixel_width_ns]

    dst_ds.SetGeoTransform(geotransform)
    xpos = 0
    ypos = 0
    for i in range(0, 8):
        # generate random data
        arr = np.ones((512, 512))
        #mult = np.random.uniform()
        mult = 1.0
        arr = arr * mult
        # write data to band 1 ("band" = "channel" in geo-speak.)
        # (Basically allows us to copy and paste an ndarray onto a geotiff.)
        xpos = i*512
        ypos = i*512
        dst_ds.GetRasterBand(1).WriteArray(arr, xoff=xpos, yoff=ypos)

    dst_ds.GetRasterBand(1).SetNoDataValue(0)
    dst_ds.FlushCache()


    print("done")
