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
    for i in range(0, 0):
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
    # - Apply a circular transparency gradient to the middle of each frame before pasting. "The dollar bill effect."
    ##################################################################
    # Params of the raster layer
    ##################################################################
    x_res = 512
    y_res = 512

    ##################################################################
    # Parameters of geotransform. Places the geotiff on the map.
    # See: https://gdal.org/tutorials/geotransforms_tut.html
    # NOTE: the upper left of the world map is -x, +y.
    ##################################################################
    upper_left_x = -16
    upper_left_y = 16
    width_x = 32
    width_y = 32
    ##################################################################

    output_fname = 'test.tif'
    driver = gdal.GetDriverByName('GTiff')

    # Bands = channels in geo-lingo
    num_bands = 3
    dst_ds = driver.Create(output_fname,
                           x_res,
                           y_res,
                           num_bands,
                           gdal.GDT_Float32)

    # Set raster's projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(srs.ExportToWkt())

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

    # The points that we want to log
    PointLog = namedtuple("PointLog", "upper_left, upper_left_actual, bottom_right, bottom_right_actual")

    def pixel_to_coords(GT, pixel_coords):
        X_pixel = pixel_coords[0]
        Y_line = pixel_coords[1]
        X_geo = GT[0] + X_pixel * GT[1] + Y_line * GT[2]
        Y_geo = GT[3] + X_pixel * GT[4] + Y_line * GT[5]
        return [X_geo, Y_geo]


    def to_geojson(data : PointLog, gt):
        # Apply geotransform to each of the lists of points
        point_lists = []
        for geom in data:
            xformed = [pixel_to_coords(gt, coords) for coords in geom]
            point_lists.append(xformed)

        # Convert lists of points to MultiPoint features
        features = []
        for (label, points) in zip(data._fields, point_lists):
            multipoint = geojson.MultiPoint(points)
            feature = geojson.Feature(geometry=multipoint, properties={
                "label": label
            })
            features.append(feature)
        # Convert array of MultiPoint features to feature collection
        feature_collection = geojson.FeatureCollection(features)

        return geojson.dumps(feature_collection, indent=2)


    point_log = PointLog([], [], [], [])

    for i in range(0, 1):
        # Sample a frame
        (vid, frame_num) = sample_frame_uniform(data)
        frame = get_frame_numpy(vid, frame_num)
        ##############################################################
        #frame = np.ones((1, x_res, y_res, 3))*(i / 8)
        ##############################################################

        # width, height, num_channels
        frame = frame[0, :, :, :]

        if frame.shape[2] != 3:
            breakpoint()

        # Dimensions of the geotiff
        geotiff_dim = np.array([x_res, y_res], dtype='int64')
        # Dimensions of the frame
        frame_dim = np.array([frame.shape[0], frame.shape[1]], dtype='int64')
        # Possible positions of the upper left corner of the frame
        extents = geotiff_dim + frame_dim
        # Coords of the upper left corner of the frame
        corner = np.random.randint(0, extents)


        ##############################################################
        # This code was a pain in the ass
        ##############################################################

        upper_left = corner - frame_dim
        bottom_right = corner

        frame_col_interval = [upper_left[0], bottom_right[0]]
        frame_row_interval = [upper_left[1], bottom_right[1]]

        img_col_interval = [0, x_res]
        img_row_interval = [0, y_res]

        submat_col_interval = interval_intersection(frame_col_interval,
                                                    img_col_interval)
        submat_row_interval = interval_intersection(frame_row_interval,
                                                    img_row_interval)
        # Two positions on the geotiff that define the footprint of
        # where we paste the frame

        row1 = min(submat_col_interval)

        index1 = [min(submat_col_interval), min(submat_row_interval)]
        index2 = [max(submat_col_interval), max(submat_row_interval)]

        # TODO: Set this to the actual submat of the frame
        mask = np.ones((index2[0]-index1[0], index2[1]-index1[1]))

        ##############################################################
        # We have the submatrix of the geotiff, get the submatrix of
        # the frame.
        ##############################################################

        X1 = min(index1[0], index2[0])
        X2 = max(index1[0], index2[0])
        Y1 = min(index1[1], index2[1])
        Y2 = max(index1[1], index2[1])
        assert(index1 == [X1, Y1])
        assert(index2 == [X2, Y2])

        upper_left - np.array(index1)

        A1 = np.array(index1) - upper_left
        A2 = np.array(index2) - upper_left

        B1 = [min(A1[0], A2[0]), max(A1[0], A2[0])]
        B2 = [min(A1[1], A2[1]), max(A1[1], A2[1])]

        frame1 = np.copy(frame)
        shape1 = frame.shape
        frame = frame[B1[0]:B1[1], B2[0]:B2[1], :]
        #frame = frame[Y1:Y2, X1:X2, :]

        assert(shape1[2] == frame.shape[2])

        ##############################################################
        '''
        print("#######################################################")
        print(f" mask.shape: {mask.shape}")
        print(f" Upper left: {upper_left}")
        print(f" X Interval: {(X1, X2)}")
        print(f"  num. kept: {X2 - X1}")
        print(f" Y Interval: {(Y1, Y2)}")
        print(f"  num. kept: {Y2 - Y1}")
        print(f"     corner: {corner}")
        '''

        point_log.upper_left_actual.append([upper_left[0], upper_left[1]])
        point_log.upper_left.append([X1, Y1])
        # Where the corner actually is
        point_log.bottom_right.append([X1 + frame.shape[0], Y1 + frame.shape[1]])
        # Where the corner would have been had it not been cut off
        point_log.bottom_right_actual.append([upper_left[0] +frame_dim[0],
                                       upper_left[1]+frame_dim[1]])
        print("#######################################################")

        frame = frame.astype(dtype="float32")

        # Why do I need this
        frame = frame.transpose((1, 0, 2))

        #breakpoint()
        dst_ds.GetRasterBand(1).WriteArray(frame[:, :, 0],
                                           xoff=int(X1),
                                           yoff=int(Y1))

        dst_ds.GetRasterBand(2).WriteArray(frame[:, :, 1],
                                           xoff=int(X1),
                                           yoff=int(Y1))

        dst_ds.GetRasterBand(3).WriteArray(frame[:, :, 2],
                                           xoff=int(X1),
                                           yoff=int(Y1))



    geojson = to_geojson(point_log, geotransform)
    with open("point_log.geojson", "w+") as fp:
        fp.write(geojson)

    dst_ds.GetRasterBand(1).SetNoDataValue(0)
    dst_ds.GetRasterBand(2).SetNoDataValue(0)
    dst_ds.GetRasterBand(3).SetNoDataValue(0)
    dst_ds.FlushCache()


    print("done")
