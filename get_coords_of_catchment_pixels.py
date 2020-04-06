import os
import sys
import json
import numpy as np
import pandas as pd
import math

from PIL import Image
import time
import pickle

from pyproj import Proj, transform, Transformer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pysheds.grid import Grid
from shapely.geometry import Point
import geopandas as gpd

from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt
# from pyproj import Geod
# from pygc import great_circle as gc

from radar_station_coords import radar_sites


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
DB_DIR = os.path.join(BASE_DIR, 'code/hydat_db')
PROJECT_DIR = os.path.abspath('')
IMG_DIR = os.path.join(PROJECT_DIR, 'data/radar_img')
FP_DIR = os.path.join(DB_DIR, 'na_dem_15s_grid')
GDIR_DIR = os.path.join(DB_DIR, 'na_dir_15s_grid')

stn_df = pd.read_csv(DB_DIR + '/WSC_Stations_Master.csv')

img_center = (239, 239)

basin_geo_path = os.path.join(PROJECT_DIR, 'data/basin_geometry_data.geojson')
basin_df = gpd.read_file(basin_geo_path)

all_stations_with_geometry = basin_df['Station'].values

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


def get_pixel_coordinates(closest_stn):
    # note that the pixel coordinates are projected to NAD83
    # coordinate system and when I figure out 
    # how to re-project without 20km error I will
    # try to estimate the error accordingly.
    img_coord_path = 'data/radar_img_pixel_coords'
    img_coord_file = [f for f in os.listdir(img_coord_path) if closest_stn in f][0]
    geo_df = gpd.read_file(os.path.join(img_coord_path, img_coord_file))
#     print('what is this?')
#     print(geo_df.crs)
#     geo_df = geo_df.to_crs('EPSG:3153')
    print('checking radar pixel coordinate bounds')
    geo_df['x'] = [e.x for e in geo_df['geometry'].values]
    geo_df['y'] = [e.y for e in geo_df['geometry'].values]

    print(geo_df.crs)

    print('radar image extents: {:.2f} {:.2f} {:.2f} {:.2f}'.format(np.min(geo_df['x']), np.min(geo_df['y']),
          np.max(geo_df['x']), np.max(geo_df['y'])))
    
    return geo_df


def get_img_mask(closest_stn, basin_geom, wsc_stn):
    # take in the basin geometry and its corresponding radar station
    # return a 480x480 boolean matrix where True represents
    # pixels that fall within the basin polygon
    # note that the points are stored in (lat, lon) tuples
    # which corresponds to y, x

    basin_geom_data = basin_geom.geometry
    
    # mask_folder = 'data/wsc_stn_basin_masks'
    # mask_path = os.path.join(PROJECT_DIR, mask_folder)
    # mask_filename = wsc_stn + '.json'
    # existing_masks = os.listdir(mask_path)

    radar_pixel_coord_df = get_pixel_coordinates(closest_stn)

    # trim the radar image df to the bounds of the watershed 
    # to make the .within() function more efficient
    minx, miny, maxx, maxy = basin_geom.bounds.values[0]
    radar_pixel_coord_df = radar_pixel_coord_df[(radar_pixel_coord_df['x'] > minx) & \
                                                (radar_pixel_coord_df['x'] < maxx) & \
                                                (radar_pixel_coord_df['y'] > miny) & \
                                                (radar_pixel_coord_df['y'] < maxy) ]

    pip_mask = radar_pixel_coord_df.within(basin_geom.loc[0, 'geometry'])

    masked_df = radar_pixel_coord_df.loc[pip_mask, :]

    xs = [e.x for e in masked_df['geometry'].values]
    ys = [e.y for e in masked_df['geometry'].values]

    df = pd.DataFrame()
    df['x'] = xs
    df['y'] = ys
    df['geometry'] = masked_df['geometry'].values

    return df, pip_mask

print('there are {} basin shape files'.format(len(basin_df)))

def get_polygon(stn):
    gdb_path = os.path.join(DB_DIR, 'WSC_Basins.gdb.zip')
    data = gpd.read_file(gdb_path, driver='FileGDB', layer='EC_{}_1'.format(stn))
    return data

def get_polygon1(stn):
    gdb_path = os.path.join(PROJECT_DIR, 'data/basin_geometry_data.geojson')
    data = gpd.read_file(gdb_path)
    basin_geom = data[data['Station'] == stn]
    print(basin_geom)
    print(basin_geom.crs)
    basin_geom = basin_geom.to_crs(4326)
    print(basin_geom.crs)
    print(asdfasd)
    return data


def get_basin_geometry(test_stn):
    basin_geom = get_polygon1(test_stn)
    # reproject to EPSG: 3395 (mercator) for plotting
    # or to coincide with radar image coordinates use
    # original WSC basin polygon is EPSG: 4269 (NAD83)
    # WGS 84 is EPSG: 4326
    basin_geom = basin_geom.to_crs(4326)
    return basin_geom


def load_dem_data(basin_geom):
    basin_bounds = basin_geom.bounds
    basin_bbox = tuple((basin_bounds['minx'].values[0] - 0.001,
                basin_bounds['miny'].values[0] - 0.001,
                basin_bounds['maxx'].values[0] + 0.001,
                basin_bounds['maxy'].values[0] + 0.001))

    grid = Grid.from_raster(path=FP_DIR + '/na_dem_15s/na_dem_15s', 
                                data_name='dem', window=basin_bbox,
                                window_crs=Proj(4326))

    
    xs = [c[1] for c in grid.view('dem').coords]
    ys = [c[0] for c in grid.view('dem').coords]

    elevations = np.array(grid.view('dem')).flatten()
    dem_df = pd.DataFrame()
    dem_df['x'] = xs
    dem_df['y'] = ys
    dem_df['elevation'] = elevations
    dem_df['geometry'] = [Point(a[0], a[1]) for a in grid.view('dem').coords]
    return dem_df

i = 0
# need to get 70 - 08MC045; 101 08KH019
last_stations = ['08KH019', '08MC045']
for wsc_stn in all_stations_with_geometry[:1]:
    i += 1
    stn_info = stn_df[stn_df['Station Number'] == wsc_stn]
    print(stn_info[['Latitude', 'Longitude']])
    print(stn_info['Station Name'])

    print('{}/{} {} Starting...'.format(i, len(all_stations_with_geometry), wsc_stn))

    t0 = time.time()
    radar_stn = find_closest_radar_stn(wsc_stn)
    
    # load the shape file containing the catchment boundary
    basin_bounds_geom = get_basin_geometry(wsc_stn)
    t1 = time.time()
    print('    ...time to load radar ({}) and basin geom: {:.2f}'.format(radar_stn, t1 - t0))
    
    # get the coordinates of all of the radar image pixels
    # that fall inside the catchment boundary
    basin_coords, catchment_mask = get_img_mask(radar_stn, basin_bounds_geom, wsc_stn)    

    # load the dem and clip it to the catchment boundary bounds
    dem_data = load_dem_data(basin_bounds_geom) 

    tree = BallTree(dem_data[['x', 'y']].values, leaf_size=2)

    basin_coords['distance_nearest'], basin_coords['id_nearest'] = tree.query(
        basin_coords[['x', 'y']].values, # The input array for the query
        k=1, # The number of nearest neighbors
    )

    els = dem_data.loc[basin_coords['id_nearest'].to_numpy(), 'elevation']
    inds = catchment_mask.index[catchment_mask.values].tolist()

    # get the elevation of the nearest dem pixel
    basin_coords['elevation'] = els.to_numpy()
    basin_coords['mask_indices'] = inds

    df_save_path = os.path.join(PROJECT_DIR, 'data/basin_pixel_coords_els/{}.csv'.format(wsc_stn))

    basin_coords.to_csv(df_save_path)

    t2 = time.time()

    print('    ...time to get image mask, basin df and dem, save output.: {:.2f}'.format(t2 - t1)) 
    print('    ###############')
