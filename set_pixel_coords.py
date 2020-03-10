import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import utm
from PIL import Image
import time
import pickle
from pyproj import Proj
from shapely.geometry import Point
import matplotlib.pyplot as plt

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

# what is the full png dimension?
#480x580
# what is radar image dimension?
#480x480

# note that the radar location "centre" does not correspond to
# the image pixel dimension centre because of the toolbar at right
# find the centre pixel coordinates (corresponds to centre of radar img)
img_center = (240, 240)
# radar_px[img_center] = stn_coords_utm

# traverse the matrix and apply a cumulative subtraction
# corresponding to the pixel distance from centre,
# exploiting the fact that the image projection resolution
# is 1:1000, i.e. 1 pixel = 1000x1000m
# the cumulative subtraction should be a tuple
# such that the coordinate for each pixel represents 
# the centre of the 1kmx1km square

def assign_latlon_to_pixel_matrix(coords):
    px = np.zeros((480, 480, 2))
    df = pd.DataFrame()
    dpp = 1 / 111.32 # roughly 1000m per pixel
    for j in range(480): # columns
        df[j] = [tuple((coords[0] + dpp * (i - 240), coords[1] + dpp * (j - 240))) for i in range(480)]
    return df


def encode_coordinate_files(radar_stn_names):
    for stn in radar_stn_names:
        stn_coords = radar_sites[stn]['lat_lon']
        # stn_coords_utm = utm.from_latlon(*stn_coords)
        radar_coord_df = assign_latlon_to_pixel_matrix(stn_coords)
        fname = PROJECT_DIR + '/data/radar_img_pixel_coords/{}_latlon_coords.json'.format(stn)
        radar_coord_df.to_json(fname)


encode_coordinate_files(list(radar_sites.keys()))

def load_df(fname):
    fpath = PROJECT_DIR + '/data/radar_img_pixel_coords/' + fname
    return pd.read_json(fpath)

fnames = [f for f in os.listdir(PROJECT_DIR + '/data/radar_img_pixel_coords')]
print(fnames[0])
# coord_pairs = load_df(fnames[0]).to_numpy().flatten()
df = load_df(fnames[0])
print(df.head())
def map_point(pt):
    return Point(pt[0], pt[1])

# encode_coordinate_files(list(radar_sites.keys()))

# coords = [map_point(p) for p in coord_pairs]

# geo_df = gpd.GeoDataFrame(geometry=coords, crs='EPSG:32633')

# print(geo_df.head())

# geo_df = geo_df.to_crs('epsg:4326')

# print(geo_df.crs)
# print(geo_df.head())

# output = []
# for i in reshaped:
#     if Point(i).within(poly):
#         output.append(i)

# import catchment boundary

# def get_polygon(stn):
#     gdb_path = DB_DIR + '/WSC_Basins.gdb.zip'
#     data = gpd.read_file(gdb_path, driver='FileGDB', layer='EC_{}_1'.format(stn))    
#     data = data.to_crs('epsg:4326')
#     return data.geometry

# test_stn = '08HE006'

# basin_geometry = get_polygon(test_stn)

# basin_bbox = basin_geometry.bounds


