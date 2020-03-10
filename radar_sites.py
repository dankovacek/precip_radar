import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import utm
from PIL import Image, ImagePalette

# import gdal
# import osr

import time
import pickle
from pyproj import Proj
from shapely.geometry import Point
import matplotlib.pyplot as plt

import rasterio
from rasterio import Affine as A
from rasterio import transform
from rasterio.warp import reproject, Resampling

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
DB_DIR = os.path.join(BASE_DIR, 'code/hydat_db')
PROJECT_DIR = os.path.abspath('')
IMG_DIR = os.path.join(PROJECT_DIR, 'data/radar_img')

stn_df = pd.read_csv(DB_DIR + '/WSC_Stations_Master.csv')

radar_sites = {'CASAG': {'lat_lon': [49.0580516, -122.470667], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Aldergrove',
                        }, # km/pixel                       
               'CASPG': {'lat_lon': [53.916943, -122.749443], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Prince George',}, # km/pixel}, # km/pixel
               'CASSS': {'lat_lon': [50.271790, -119.276505], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Silver Star',}, # km/pixel}, # km/pixel
               'CASSI': {'lat_lon': [48.407326, -123.329773], # radar location code, lat/lon
                       'scale': 1,
                      'alt_name': 'Victoria',}, # km/pixel}, # km/pixel
               'CASSM': {'lat_lon': [51.206092, -113.399426],
                        'scale': 1,
                        'alt_name': 'Strathmore'},
              }


def find_closest_radar_stn(wsc_stn):
    """
    To retrieve radar images, we need to find the closest radar location
    to the station of interest.  
    Input the station number,
    returns the location code of the nearest radar station.
    """
    stn_data = stn_df[stn_df['Station Number'] == wsc_stn]
    s1 = [stn_data['Latitude'].values[0], stn_data['Longitude'].values[0]]
    min_dist = 1E6
    closest_stn = None
    for site in radar_sites.keys():

        s2 = [*radar_sites[site]['lat_lon']]        

        this_dist = np.sqrt((s2[0] - s1[0])**2 + (s2[1] - s1[1])**2)

        if this_dist < min_dist:
            min_dist = this_dist
            closest_stn = site
        
    return closest_stn

def load_img_rgb_values(path, img):
    gif_img = Image.open(os.path.join(path, img))

    with rasterio.open(os.path.join(path, img)) as gif_rs:
        p = gif_rs.profile
        p['driver'] = 'GTiff'
        # print(p)

    label = gif_img.convert('RGB')

    img_array = np.asarray(label)
    radar_img_array = np.asarray(img_array)[:,:480]
    color_bar_array = img_array[144:340, 515:535]
    # color_bar_img = Image.fromarray(color_bar_array, mode='RGB')
    # color_bar_img.save('colorbar.png')
    # colors = color_bar_img.getcolors()
    return radar_img_array, color_bar_array

def modify_raster(coords, path, img):
    with rasterio.Env():

        # As source: a 480 x 480 raster centered on coords[1] degrees E and coords[0]
        # degrees N, each pixel covering 15".
        rows, cols = src_shape = (480, 480)
        dpp = 1.0/240 # decimal degrees per pixel
        dpp = 1 / 111.32 # roughly 1000m per pixel
        # The following is equivalent to
        # west, south, east, north = -cols*dpp/2, -rows*dpp/2, cols*dpp/2, rows*dpp/2
        # src_transform = transform.from_bounds(west, south, east, north, cols, rows)
        # src_transform = A(dpp, coords[1], -cols*dpp/2, coords[0], -dpp, rows*dpp/2)
        src_transform = transform.from_origin(coords[1], coords[0], 111.32, 111.32)
        # src_transform = A.translation(-cols*d/2, rows*d/2) * A.scale(d, -d)
        src_crs = {'init': 'EPSG:4326'}
    
        img_src, color_bar_img = load_img_rgb_values(path, img)
        mod_src = img_src.reshape((480*480, 3))
        r_chan = np.asarray([e[0] for e in mod_src]).reshape((480, 480))
        b_chan = np.asarray([e[1] for e in mod_src]).reshape((480, 480))
        g_chan = np.asarray([e[2] for e in mod_src]).reshape((480, 480))
        src = np.asarray([r_chan, b_chan, g_chan])

        dest_folder = path + '/tif/'
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
       
        with rasterio.open(
            dest_folder + '{}.tif'.format(img.split('.')[0]),
            'w',
            driver='GTiff',
            height=480,
            width=480,
            count=3,
            dtype=np.uint8,
            nodata=0,
            crs=src_crs, 
            photometric='RGB',
            transform=src_transform,
        ) as dst:
            # dst.write(r_chan, 1)
            dst.write(src)
            # dst.write(img_src.T, 3)
            

radar_image_folders = [e for e in os.listdir(IMG_DIR)]
print(radar_image_folders)

folders = ['08MH006']

for f in folders:
    closest_radar_stn = find_closest_radar_stn(f)
    coords = radar_sites[closest_radar_stn]['lat_lon']
    print(closest_radar_stn, coords)

    # print(os.path.join(IMG_DIR, f))
    img_filepath = os.path.join(IMG_DIR, f)
    img_files = [img for img in os.listdir(img_filepath) if os.path.isfile(os.path.join(img_filepath, img))]
    for i in img_files:
        modify_raster(coords, img_filepath, i)
        # print(asdf)


