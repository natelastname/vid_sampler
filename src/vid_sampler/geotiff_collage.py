#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-08-20T11:39:53-04:00

@author: nate
"""
from collections import namedtuple

import numpy as np
from osgeo import gdal
from osgeo import osr
import geojson
import tqdm

from . import functions as vs


def pixel_to_coords(GT, pixel_coords):
    X_pixel = pixel_coords[0]
    Y_line = pixel_coords[1]
    X_geo = GT[0] + X_pixel * GT[1] + Y_line * GT[2]
    Y_geo = GT[3] + X_pixel * GT[4] + Y_line * GT[5]
    return [X_geo, Y_geo]


PointLog = namedtuple("PointLog", "upper_left, upper_left_actual, bottom_right, bottom_right_actual")
GeotiffParams = namedtuple("GeotiffParams", "x_res, y_res, gt")


def to_geojson(data: PointLog, gt):
    """
    Apply geotransform to each of the lists of points in data, apply
    the geotransform gt, return a geojson string.

    Primarily used for debugging.
    """
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


def simple_geotransform(x_res, y_res, upper_left_lon, upper_left_lat, width_x, width_y):
    """
    Convenience function to construct a simple geotransform with 0
    row / column rotation.
    """

    ##################################################################
    # Parameters of geotransform. Places the geotiff on the map.
    # See: https://gdal.org/tutorials/geotransforms_tut.html
    # NOTE: the upper left of the world map is -x, +y.
    ##################################################################
    pixel_width_ns = width_x / x_res
    pixel_width_we = width_y / y_res
    geotransform = [upper_left_lon,
                    pixel_width_we,
                    0.0,
                    upper_left_lat,
                    0.0,
                    -1*pixel_width_ns]

    return GeotiffParams(x_res, y_res, geotransform)


def geotiff_collage(output_fname,
                    img_params: GeotiffParams,
                    vids,
                    num_frames,
                    samples=None):
    '''
    `samples` is a list of FrameClass objects.
    '''
    # Bands = channels in geo-lingo
    num_bands = 3

    driver = gdal.GetDriverByName('GTiff')

    dst_ds = driver.Create(output_fname,
                           img_params.x_res,
                           img_params.y_res,
                           num_bands,
                           gdal.GDT_Float32)

    if dst_ds == None:
        size = vs.human_readable_size(img_params.x_res * img_params.y_res * 24)
        raise Exception(f"Couldn't create {size} geotiff.")

    # Set raster's projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(srs.ExportToWkt())

    dst_ds.SetGeoTransform(img_params.gt)

    # There is no reason not to save the exact image footprints,
    # plus its convenient for debugging
    point_log = PointLog([], [], [], [])

    # Use TQDM but not the god awful progress bar
    pbar = tqdm.tqdm(total=num_frames, ncols=0)

    if samples != None:
        num_frames = len(samples)

    for i in range(0, num_frames):
        pbar.update()
        # Sample a frame

        if samples != None:
            frame = samples[i]
        else:
            frame = vs.sample_frame_uniform(vids)

        frame = vs.get_frame_numpy(frame.vid, frame.frame_num)

        # width, height, num_channels
        frame = frame[0, :, :, :]

        if frame.shape[2] != 3:
            breakpoint()

        # Dimensions of the geotiff
        geotiff_dim = np.array([img_params.x_res, img_params.y_res], dtype='int64')
        # Dimensions of the frame
        frame_dim = np.array([frame.shape[0], frame.shape[1]], dtype='int64')
        # Possible positions of the upper left corner of the frame
        extents = geotiff_dim + frame_dim
        # Coords of the upper left corner of the frame
        corner = np.random.randint(0, extents)

        ##############################################################
        # This for loop was a pain in the ass.
        ##############################################################

        upper_left = corner - frame_dim
        bottom_right = corner

        frame_col_interval = [upper_left[0], bottom_right[0]]
        frame_row_interval = [upper_left[1], bottom_right[1]]

        img_col_interval = [0, img_params.x_res]
        img_row_interval = [0, img_params.y_res]

        submat_col_interval = vs.interval_intersection(frame_col_interval,
                                                       img_col_interval)
        submat_row_interval = vs.interval_intersection(frame_row_interval,
                                                       img_row_interval)
        # Two positions on the geotiff that define the footprint of
        # where we paste the frame

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

        A1 = np.array(index1) - upper_left
        A2 = np.array(index2) - upper_left

        B1 = [min(A1[0], A2[0]), max(A1[0], A2[0])]
        B2 = [min(A1[1], A2[1]), max(A1[1], A2[1])]

        shape1 = frame.shape
        frame = frame[B1[0]:B1[1], B2[0]:B2[1], :]
        assert shape1[2] == frame.shape[2]

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
        print("#######################################################")
        '''

        point_log.upper_left_actual.append([upper_left[0], upper_left[1]])
        point_log.upper_left.append([X1, Y1])
        # Where the corner actually is
        point_log.bottom_right.append([X1 + frame.shape[0], Y1 + frame.shape[1]])
        # Where the corner would have been had it not been cut off
        point_log.bottom_right_actual.append([upper_left[0] + frame_dim[0],
                                              upper_left[1] + frame_dim[1]])

        frame = frame.astype(dtype="float32")

        # I don't know why this is needed
        frame = frame.transpose((1, 0, 2))
        dst_ds.GetRasterBand(1).WriteArray(frame[:, :, 0],
                                           xoff=int(X1),
                                           yoff=int(Y1))
        dst_ds.GetRasterBand(2).WriteArray(frame[:, :, 1],
                                           xoff=int(X1),
                                           yoff=int(Y1))
        dst_ds.GetRasterBand(3).WriteArray(frame[:, :, 2],
                                           xoff=int(X1),
                                           yoff=int(Y1))

    pbar.close()

    dst_ds.GetRasterBand(1).SetNoDataValue(0)
    dst_ds.GetRasterBand(2).SetNoDataValue(0)
    dst_ds.GetRasterBand(3).SetNoDataValue(0)
    dst_ds.FlushCache()
    '''
    geojson = to_geojson(point_log, geotransform)
    with open("point_log.geojson", "w+") as fp:
        fp.write(geojson)
    '''
