import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os 
import sys
import math
import utm
import time

import json
import geopandas as gpd
import fiona
from geopy import distance

from numba import jit

from pysheds.grid import Grid
from shapely.geometry import shape, mapping, Polygon
from pyproj import Proj, transform

from PIL import Image
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

from bokeh.plotting import ColumnDataSource, output_notebook
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Spectral3
from bokeh.layouts import gridplot

import matplotlib.pyplot as plt
import matplotlib.patches as mp

from get_station_data import get_daily_runoff
from radar_station_coords import radar_sites


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
DB_DIR = os.path.join(BASE_DIR, 'code/hydat_db')
PROJECT_DIR = os.path.abspath('')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'data/AD_results')
FP_DIR = os.path.join(DB_DIR, 'na_dem_15s_grid')
GDIR_DIR = os.path.join(DB_DIR, 'na_dir_15s_grid')

results_folders = os.listdir(RESULTS_DIR)

results_dict = {}
# create a dictionary of results from all AD searches
for f in results_folders:
    folder_path = os.path.join(RESULTS_DIR, f)
    all_sites = [e.split('_')[0] for e in os.listdir(folder_path)]
    for site in all_sites:
        if site in results_dict.keys():
            old_results = results_dict[site]
            new_results = pd.read_csv(os.path.join(folder_path, site + '_results.csv'))
            results_dict[site] = pd.concat([old_results, new_results], sort=True)
        else:            
            results_dict[site] = pd.read_csv(os.path.join(folder_path, site + '_results.csv'))


def get_best_result(site):
    ad_df = pd.DataFrame(results_dict[site])
    ad_df.drop(labels='Unnamed: 0', inplace=True, axis=1)
    ad_df.sort_values('len_results', inplace=True, ascending=False)
    return ad_df.iloc[0, :]

def find_closest_radar_stn(row):
    """ 
    Input the dict of all station distances,
    Return the location code of the nearest radar station.
    """
    radar_station_distances = row['radar_stn_distance_dict']
    min_dist = min(radar_station_distances.items(), key=lambda x: x[1])
    return min_dist[0]


def find_closest_radar_stn_distance(row):
    """ 
    Input the dict of all station distances,
    Return the location code of the nearest radar station.
    """
    radar_station_distances = row['radar_stn_distance_dict']
    min_dist = min(radar_station_distances.items(), key=lambda x: x[1])
    return min_dist[1]


def calc_distance(wsc_row, station):
    wsc_stn_coords = (wsc_row['Latitude'], wsc_row['Longitude'])
    radar_coords = radar_sites[station]['lat_lon']
    return distance.distance(radar_coords, wsc_stn_coords).km

def calculate_radar_stn_distances(row):
    distance_dict = {}
    for site in radar_sites:
        distance_dict[site] = calc_distance(row, site)
    return distance_dict

def initialize_wsc_station_info_dataframe():
    # import master station list
    stations_df = pd.read_csv(DB_DIR + '/WSC_Stations_Master.csv')
    # filter for stations that have concurrent record with the historical radar record
    stations_df['RADAR_Overlap'] = stations_df['Year To'].astype(int) - 2007
    stations_filtered = stations_df[stations_df['RADAR_Overlap'] > 0]
    # filter for stations that are natural flow regimes
    stations_filtered = stations_filtered[stations_filtered['Regulation'] == 'N']
    stations_filtered.rename(columns={'Gross Drainage Area (km2)': 'DA'}, inplace=True)
    # filter for stations in Alberta and British Columbia
    stations_filtered = stations_filtered[(stations_filtered['Province'] == 'BC') | (stations_filtered['Province'] == 'AB')]
    
    # calculate distance to each radar station
    stations_filtered['radar_stn_distance_dict'] = stations_filtered.apply(lambda row: calculate_radar_stn_distances(row), axis=1)    
    stations_filtered['closest_radar_station'] = stations_filtered.apply(lambda row: find_closest_radar_stn(row), axis=1)
    stations_filtered['radar_distance_km'] = stations_filtered.apply(lambda row: find_closest_radar_stn_distance(row), axis=1)
    
    # radar range is a 240km radius from the station
    stations_filtered = stations_filtered[stations_filtered['radar_distance_km'] < 200]
    stn_df = stations_filtered[np.isfinite(stations_filtered['DA'].astype(float))]
    # filter for stations greater than 10 km^2 (too small for meaningful results)
    stn_df = stn_df[stn_df['DA'].astype(float) >= 10]
    # filter for stations smaller than 1000 km^2 (too large and complex)
    stn_df = stn_df[stn_df['DA'].astype(float) < 1000].sort_values('DA')
    df = stn_df[['Province', 'Station Number', 'Station Name', 'DA', 
                 'Elevation', 'Latitude', 'Longitude', 'RADAR_Overlap',
                'closest_radar_station', 'radar_stn_distance_dict', 'radar_distance_km']]
#     print('After filtering, there are {} candidate stations.'.format(len(stn_df)))
    df.reset_index(inplace=True)
    return df

all_sites = list(results_dict.keys())

wsc_info = initialize_wsc_station_info_dataframe()

# load all the basin data into a pandas dataframe

gdb_path = os.path.join(DB_DIR, 'WSC_Basins.gdb.zip')
all_layers = fiona.listlayers(gdb_path)
all_layer_names = [e.split('_')[1].split('_')[0] for e in all_layers]
filtered_layers = sorted(list(set(all_sites).intersection(all_layer_names)))


# # For DEM plot
# # Specify directional mapping
# #         N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
boundaries = ([0] + sorted(list(dirmap)))

def get_basin_geometry(stn):
    basin_df = gpd.GeoDataFrame()
    layer_label = 'EC_' + stn + '_1'
    s = gpd.read_file(gdb_path, driver='FileGDB', layer=layer_label)
    basin_df = basin_df.append(s, ignore_index=True)
    # original WSC basin polygon is EPSG: 4269 (NAD83)
    # WGS 84 is EPSG: 4326
    return basin_df

ascii_save_path = os.path.join(PROJECT_DIR, 'data/ascii_dem')
pickled_hypso_dict = {}
i = 1
for stn in filtered_layers[:1]:
    t0 = time.time()
    # get the basin data
    basin_data = get_basin_geometry(stn)
    basin_geom = basin_data.geometry
    basin_geom = basin_geom.to_crs(4326)
    basin_bounds = basin_geom.bounds
    stn_info = wsc_info[wsc_info['Station Number'] == stn]
    stn_da = stn_info['DA'].values[0]
    stn_el = stn_info['Elevation'].values[0]
    print('{} ({} km2) el. {} m'.format(stn, stn_da, stn_el))
    x, y = stn_info['Longitude'].values[0], stn_info['Latitude'].values[0]

    # expand the bounding box slightly
    # 0.01 decimal degrees equals approximately 1.1132 km

    basin_bbox = tuple((basin_bounds['minx'].values[0] - 0.2,
                basin_bounds['miny'].values[0] - 0.2,
                basin_bounds['maxx'].values[0] + 0.2,
                basin_bounds['maxy'].values[0] + 0.2))
    print(basin_bbox)
    # get the DEM data
    t1 = time.time()
    print('{}/{} {} basin geometry loaded in {:.2f}s'.format(i, len(all_sites), stn, t1-t0))
    grid = Grid.from_raster(path=FP_DIR + '/na_dem_15s/na_dem_15s', 
                            data_name='dem', window=basin_bbox, 
                            nodata=np.nan)
    print(grid.crs)
    # grid.read_raster(GDIR_DIR + '/na_dir_15s/na_dir_15s', 
                    # data_name='dem')#, window=basin_bbox, nodata=np.nan)#,
                    # window_crs=Proj('epsg:4326'))
    print(grid.view('dem'))
    print(grid.crs)
    # reset the nodata from -32768 to 0
    grid.catchment(data='dem', x=x, y=y, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)
    grid.clip_to('catch', pad=(1,1,1,1))
    # Get a view of the catchment
    catch = grid.view('catch', nodata=np.nan)
    
    print(catch)
    print('station da = ', stn_da)
    print(np.shape(catch))
    print(asdfasdf)

    # grid.to_ascii('dem', ascii_save_path + '/{}.asc'.format(stn))
    # pickled_hypso_dict[stn] = catch
    t2 = time.time()
    print('    ...completed raster loading and hypso curve in {:.2f}s'.format(i, len(all_sites), stn, round(t2-t1, 1)))
    i += 1



# save_path = os.path.join(PROJECT_DIR, 'data')
# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle
# with open(path + '/pickled_hypso_dict.p', 'wb') as fp:
#     pickle.dump(pickled_hypso_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)