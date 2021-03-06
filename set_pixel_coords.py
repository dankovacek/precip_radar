import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import math

from PIL import Image
import time
import pickle
from pyproj import Proj, transform, Transformer
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pyproj import Geod
from pygc import great_circle as gc

from radar_station_coords import radar_sites


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
DB_DIR = os.path.join(BASE_DIR, 'code/hydat_db')
PROJECT_DIR = os.path.abspath('')
IMG_DIR = os.path.join(PROJECT_DIR, 'data/radar_img')

stn_df = pd.read_csv(DB_DIR + '/WSC_Stations_Master.csv')



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
# according to the data publisher, centre is located at 239, 239
img_center = (239, 239)

# traverse the matrix and apply a cumulative subtraction
# corresponding to the pixel distance from centre,
# exploiting the fact that the image projection resolution
# is 1:1000, i.e. 1 pixel = 1000x1000m
# the cumulative subtraction should be a tuple
# such that the coordinate for each pixel represents 
# the centre of the 1kmx1km square

def assign_latlon_to_pixel_matrix(coords, ):
    # dpp = 1 / 111.32 # roughly 1000m per pixel
    # set a step size to the image resolution
    img_scale = 1000 # 1 km / 1 pixel

    # calculate radial distance and azimuth (AZ)
        # euclidean distance based on cell position
        # AZ measured from north get from inverse tangent function
        # of v/h pixels

    t1 = time.time()

    # p_mt = Proj('EPSG:32662') # metric
    # p_WGS84 = Proj('EPSG:4326') # WGS 84
    # p_NAD83 = Proj('EPSG:4269') # NAD83
    # p_BC = Proj('EPSG:3153')
    # p_esri = Proj('esri:102001')
    # p_utm = Proj('epsg:3395')
    # p_bc2 = Proj('epsg:3005')
    # p_sc = Proj('epsg:3347')
    # p_utm11N = Proj('epsg:26911')
    # p_utm12N = Proj('epsg:26912')

    # projection = Proj('epsg:3005')

    x_centre, y_centre = coords[1], coords[0]

    gridpoints = []
    n = 0
    px = np.zeros((480, 480, 2))
    for r in range(480):
        t0 = time.time()
        for c in range(480):
            # calculate the radial distance from centre
            # 1 px = 1km
            # compute dx and dy centred at location 239, 239
            dx = c - 239
            dy = 239 - r
            R = np.sqrt(dx**2 + dy**2)

            scale = 1000

            if (dx == 0) & (dy == 0):
                xx, yy = 0, 0
            else:
                if (dx <= 0) & (dy >= 0):
                    az = math.acos(abs(dx) / R) + 3 * np.pi / 2
                    bearing = math.degrees(az) 

                    xx = scale * R * math.cos(az)
                    yy = scale * R * math.sin(az)                    
                elif (dx <= 0) & (dy <= 0):
                    az = 3 * np.pi / 2 - math.acos(abs(dx) / R)
                    bearing = math.degrees(az)
                    xx = scale * R * math.cos(az)
                    yy = scale * -1.0 * R * math.sin(az)                    
                elif (dx >= 0) & (dy <= 0):
                    az = math.asin(abs(dy) / R) + np.pi / 2
                    bearing = math.degrees(az)                    
                    xx = scale * R * math.sin(az)
                    yy = scale * -1.0 * R * math.cos(az)
                else:
                    az = math.acos(dy / R)
                    bearing = math.degrees(az)
                    xx = scale * R * math.sin(az)
                    yy = scale * R * math.cos(az) 
  
        # 
            # px[r, c] = tuple((x_centre + xx, y_centre + yy))

            g = Geod(ellps='WGS84')
            lon2, lat2, bear2 = g.fwd(x_centre, y_centre, bearing, R*scale)
            px[r, c] = tuple((lon2, lat2))
    
    transformed_pts = np.array(px).reshape(480 * 480, 2)
    x_coords = [e[0] for e in transformed_pts]
    y_coords = [e[1] for e in transformed_pts]
    
    df = pd.DataFrame({'x': x_coords, 'y': y_coords})
    df['coords'] = list(zip(df['x'], df['y']))
    # df['geometry'] = [Point(e[0], e[1], crs='epsg:4326') for e in df['coords']]
    df['geometry'] = df['coords'].apply(Point)

    df = df[['geometry']]
    print('stn_loc:')
    print(px[239, 239])
    print('image extents: {:.2f} {:.2f} {:.2f} {:.2f}'.format(np.min(x_coords), np.min(y_coords),
          np.max(x_coords), np.max(y_coords)))

    t2 = time.time()
    print('transform time = {:.2f}'.format(t2 - t1))
    print(df)
    # gridpoints = [Point(pt) for pt in transformed_pts]
    geo_df = gpd.GeoDataFrame(df, geometry='geometry', crs='epsg:4326')
    # geo_df = geo_df.to_crs('epsg:4269')
    return geo_df


def encode_coordinate_files(radar_stn_names):
    for stn in radar_stn_names:
        print('station: {}'.format(stn))
        stn_coords = radar_sites[stn]['lat_lon'] # order is y, x
        radar_coord_gdf = assign_latlon_to_pixel_matrix(stn_coords)

        fname = PROJECT_DIR + '/data/radar_img_pixel_coords/{}_coords.geojson'.format(stn)
        radar_coord_gdf.to_file(fname, driver='GeoJSON')
        print('saved...')
        print('')
        # np.save(fname, allow_pickle=True)


encode_coordinate_files(list(radar_sites.keys()))

print(breasdf)

def load_df(fname):
    fpath = PROJECT_DIR + '/data/radar_img_pixel_coords/' + fname
    return pd.read_json(fpath)

fnames = os.listdir(PROJECT_DIR + '/data/radar_img_pixel_coords')

# coord_pairs = load_df(fnames[0]).to_numpy().flatten()
df = load_df(fnames[-1])

def map_point(pt):
    return Point(pt[0], pt[1])

# encode_coordinate_files(list(radar_sites.keys()))

print(breakasdfasd)

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

test_stn = '08GA072'

basin_geometry = get_polygon(test_stn)

basin_bbox = basin_geometry.bounds


