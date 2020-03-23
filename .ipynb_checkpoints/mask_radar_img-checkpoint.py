import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
DB_DIR = os.path.join(BASE_DIR, 'code/hydat_db/')

catchments = gpd.read_file(DB_DIR + 'WSC_Catchments/HYDZ_HYD_WATERSHED_BND_POLY.geojson')

def get_mask(s):
    stn_name = s.split(' ')[0]
    print(stn_name)

    foo = catchments[catchments['SOURCE_NAME'].str.contains(stn_name)]
    print(foo['SOURCE_NAME'])
    print(len(foo['SOURCE_NAME']))
    if len(foo) == 1:
        return True, foo['SOURCE_NAME'].values[0], foo['geometry'].values[0]
    else: 
        return False, False, False
    

def get_wsc_df():
    wsc_stations = pd.read_csv(DB_DIR + 'WSC_Stations_Master.csv')
    wsc_stations = wsc_stations[wsc_stations['Province'] == 'BC']
    return wsc_stations


stn_info = get_wsc_df()
stn_selections = stn_info['Station Name'].sample(20)
good_stns = {}
for s in stn_selections:
    found, s_name, geom = get_mask(s)
    if found:
        good_stns[s_name] = geom

print(good_stns)

foo = list(good_stns.keys())[0]

gm = good_stns[foo]

