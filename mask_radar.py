import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import utm
from PIL import Image, ImagePalette

import time
import pickle
from pyproj import Proj
from shapely.geometry import Point
import matplotlib.pyplot as plt

import rasterio
from rasterio import Affine as A
from rasterio.mask import mask
from rasterio import transform
from rasterio.warp import reproject, Resampling

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
DB_DIR = os.path.join(BASE_DIR, 'code/hydat_db')
PROJECT_DIR = os.path.abspath('')
IMG_DIR = os.path.join(PROJECT_DIR, 'data/radar_img')

stn_df = pd.read_csv(DB_DIR + '/WSC_Stations_Master.csv')

radar_sites = {'CASAG': {'lat_lon': [49.0580516, -122.470667], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Aldergrove',},                       
               'CASPG': {'lat_lon': [53.916943, -122.749443], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Prince George',}, 
               'CASSS': {'lat_lon': [50.271790, -119.276505], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Silver Star',}, 
               'CASSI': {'lat_lon': [48.407326, -123.329773], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Victoria',}, 
              }

wsc_stn = '08MH006'

tif_path = wsc_stn + '/tif/'

tif_files = [f for f in os.listdir(os.path.join(IMG_DIR, tif_path))]

TIF_PATH = os.path.join(IMG_DIR, tif_path)

# import catchment boundary

def get_polygon(stn):
    gdb_path = DB_DIR + '/WSC_Basins.gdb.zip'
    data = gpd.read_file(gdb_path, driver='FileGDB', layer='EC_{}_1'.format(stn))    
    data = data.to_crs('epsg:4326')
    return data.geometry


def mask_radar(stn):
    basin_geometry = get_polygon(stn)
    basin_bbox = basin_geometry.bounds
    for f in tif_files:
        fpath = os.path.join(TIF_PATH, f)
        with rasterio.open(fpath) as src:
            out_image, out_transform = mask(src, basin_geometry, crop=True)
            out_meta = src.meta

            out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

        with rasterio.open("masked.tif", "w", **out_meta) as dest:
            dest.write(out_image)
        

        print(break_here)

mask_radar(wsc_stn)