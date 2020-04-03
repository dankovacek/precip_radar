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

import codecs

from shapely.geometry import Point, shape, mapping, Polygon

from sklearn.decomposition import PCA
from sklearn import preprocessing

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from bokeh.plotting import ColumnDataSource, output_notebook
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Spectral3
from bokeh.layouts import gridplot

import matplotlib.pyplot as plt

from radar_scrape import get_radar_img_urls, request_img_files
from get_station_data import get_daily_runoff

from numpy.random import seed
import tensorflow

from keras.layers import Input, Dropout
from keras.layers.core import Dense 
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json

from radar_station_coords import radar_sites

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
DB_DIR = os.path.join(BASE_DIR, 'code/hydat_db')
PROJECT_DIR = os.path.abspath('')
# IMG_DIR = os.path.join(PROJECT_DIR, 'data/radar_img')
RADAR_IMG_DIR = os.path.join(PROJECT_DIR, 'data/sorted_radar_images')

RESULTS_DIR = os.path.join(PROJECT_DIR, 'data/AD_results')


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
    stations_filtered = stations_filtered[stations_filtered['radar_distance_km'] < 211]
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



def create_lag_df(flow_df, stn_da):

    lag_df = flow_df.copy()
    
    lag_df.rename(columns={'DAILY_FLOW': 'Q'}, inplace=True)

    num_lags = int(np.ceil(stn_da / 100) + 5)

    for i in range(1,num_lags):
        lag_df['Q{}'.format(i)] = lag_df['Q'].shift(i)

    lag_df.dropna(inplace=True)
    
    return lag_df, num_lags


def split_train_and_test_data(data, training_months, training_year):
    time_range_check = (data.index.year == training_year) & (data.index.month.isin(training_months))
    train_data = data[time_range_check]
    # the test data is the entire dataset because we want to extract
    # extreme events from the training year as well
    test_data = data
    return train_data, test_data


def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def initialize_runoff_dataframe(test_stn):
    runoff_df = get_daily_runoff(test_stn)
    runoff_df['Q'] = runoff_df['DAILY_FLOW']
    runoff_df = runoff_df[['Q']]
    
    # filter by minimum radar date
    runoff_df = runoff_df[runoff_df.index > pd.to_datetime('2007-05-31')]
  
    return runoff_df

def do_PCA(X_train, X_test, n_components):
    
    for n_components_kept in range(2, n_components + 1):

        pca = PCA(n_components=n_components_kept, svd_solver= 'full')
        X_train_PCA = pca.fit_transform(X_train)
        X_train_PCA = pd.DataFrame(X_train_PCA)
        X_train_PCA.index = X_train.index

        X_test_PCA = pca.transform(X_test)
        X_test_PCA = pd.DataFrame(X_test_PCA)
        X_test_PCA.index = X_test.index

        var_expl = 100*np.sum(pca.explained_variance_ratio_)
        if var_expl >= 90:
#             print('var > 0.9 in {} components'.format(n_components_kept))
            return X_train_PCA, X_test_PCA, var_expl, n_components_kept
#     print('var < 0.9 in {} components'.format(n_components_kept))
    return X_train_PCA, X_test_PCA, var_expl, n_components_kept


def get_start_from_annual_distribution(df):
    annual_dist = df.groupby(df.index.month).mean()
#     print(annual_dist)
    annual_dist['rank'] = annual_dist['Q'].rank(ascending=False)
    annual_dist['b'] = 1 - annual_dist['rank'] / float(len(annual_dist) - 1)
    june_val = annual_dist[annual_dist.index == 6]['b'].values[0]
    if june_val > 0.8:
        return 7
    else:
        return 6


def train_model(input_data):
    tstart = time.time()
    training_months = input_data['months']
    training_year = input_data['year']
    wsc_station_num = input_data['wsc_stn']
    training_sample_size = input_data['n_sample']
    stn_da = input_data['stn_da']
    closest_radar_stn = input_data['radar_stn']
    
    lag_df = input_data['lag_df']
    num_lags = input_data['num_lags']
    runoff_df = input_data['runoff_df']
    
    dataset_train, dataset_test = split_train_and_test_data(lag_df, training_months, training_year)
    
    training_set_len = len(dataset_train)
    
    if len(dataset_train) < 25:
#         print('exited because dataset_train is too small')
#         print(dataset_train)
        return pd.DataFrame([]), 0

    t0 = time.time()

    scaler = preprocessing.MinMaxScaler()

    X_train = pd.DataFrame(scaler.fit_transform(dataset_train), 
                                  columns=dataset_train.columns, 
                                  index=dataset_train.index)
    # Random shuffle training data
    X_train.sample(frac=1)

    X_test = pd.DataFrame(scaler.transform(dataset_test), 
                                 columns=dataset_test.columns, 
                                 index=dataset_test.index)
    t1 = time.time()
    
   
    X_train_PCA, X_test_PCA, var_expl, n_components = do_PCA(X_train, X_test, num_lags)
    
    data_train = np.array(X_train_PCA.values)
    data_test = np.array(X_test_PCA.values)
    t2 = time.time()
#     print('time to end of PCA = {:.4f}'.format(t2-tstart))

    
    def cov_matrix(data, verbose=False):
        covariance_matrix = np.cov(data, rowvar=False)
        if is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if is_pos_def(inv_covariance_matrix):
                return True, covariance_matrix, inv_covariance_matrix
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
                return False, None, None
        else:
#             print("Error: Covariance Matrix is not positive definite!")
            return False, None, None

               
    cov_test, cov_matrix, inv_cov_matrix = cov_matrix(data_train)
    
    if cov_test == False:
        return pd.DataFrame([]), 0

    mean_distr = data_train.mean(axis=0)

    dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
    dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    threshold = MD_threshold(dist_train, extreme = True)
    
    anomaly_train = pd.DataFrame()
    anomaly_train['Mob dist']= dist_train
    anomaly_train['Thresh'] = threshold
    # If Mob dist above threshold: Flag as anomaly
    anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
    anomaly_train.index = X_train_PCA.index
    anomaly = pd.DataFrame()
    anomaly['Mob dist']= dist_test
    anomaly['Thresh'] = threshold
    anomaly['num_components_kept'] = n_components
    # If Mob dist above threshold: Flag as anomaly
    anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
    anomaly.index = X_test_PCA.index
    anomaly.head()
    
    anomaly_alldata = pd.concat([anomaly_train, anomaly], sort=True)
    
    event_times = np.where(anomaly_alldata['Anomaly'].values[:-1] != anomaly_alldata['Anomaly'].values[1:])[0]
    events = pd.merge(lag_df, anomaly_alldata.iloc[event_times,:], how='inner', 
                      left_index=True, right_index=True)

    events = events.loc[~events.index.duplicated(keep='first')]
    
    if len(events) < 5:
#         print('exited because len(events) < 5')
        return pd.DataFrame([]), n_components
    elif events.iloc[0]['Anomaly'] == True:
        events = events.iloc[1:]
        
    # create a column of time difference between events in days
    events['dt_days'] = events.index.to_series().diff(1)    

    a = time.time()

    last_event_end = False

    new_events = pd.DataFrame()

    # iterate through the detected event pairs 
    for i in np.arange(0, len(events) - 1, 2):
        
        # parse a single event pair
        this_event = events.iloc[i:i+2]
        
        check_sign_switch = this_event['Anomaly'].values[0] != this_event['Anomaly'].values[1]
        taa = time.time()
        concurrent_wsc = lag_df[(lag_df.index >= this_event.index.values[0]) & (lag_df.index <= this_event.index.values[1])][['Q']]
        peak_in_middle = check_peak_in_middle(this_event, concurrent_wsc)


        if (check_sign_switch) & (peak_in_middle):
            
            
            
            # get the start date
            this_event_start = pd.to_datetime(this_event[this_event['Anomaly'] == False].index.values[0])
            # get the end date
            this_event_end = pd.to_datetime(this_event[this_event['Anomaly'] == True].index.values[0])
            tloops = time.time()
            adjusted_start_date = pd.to_datetime(adjust_edge_date(this_event_start, lag_df[['Q']], 'start', stn_da))
            adjusted_end_date = pd.to_datetime(adjust_edge_date(this_event_end, lag_df[['Q']], 'end', stn_da))
            
            lag_df_start = pd.to_datetime(lag_df.index.values[0])
            
            tin = time.time()
#             print('asd {:.3f}'.format(tin - tloops))
            
            # check if the adjusted start date predates the record
            if lag_df_start > adjusted_start_date:
                adjusted_start_date = lag_df_start
#                 print('this was adjusted')
            
            if last_event_end is not False:
                # find if the start date is on the rising limb - adjust if so

                if adjusted_start_date < last_event_end:
                    adjusted_start_date = last_event_end + pd.DateOffset(1)
                    
            new_event_start = lag_df[lag_df.index == adjusted_start_date][['Q']]
            new_event_end = lag_df[lag_df.index == adjusted_end_date][['Q']] 

            new_event_start['timing'] = 'start'
            new_event_end['timing'] = 'end'
            
            start_month_limit = get_start_from_annual_distribution(runoff_df)            
            
            if stn_da < 100:
                max_days = 4
            elif stn_da < 500:
                max_days = 6
            else:
                max_days = 14
                
            if len(new_event_start) == 0:
                new_event_start = lag_df[lag_df.index == this_event_start][['Q']]
            if len(new_event_end) == 0:
                new_event_end = lag_df[lag_df.index == this_event_end][['Q']]
            
            min_time_check = (new_event_end.index - new_event_start.index).days > 1
            max_time_check = (new_event_end.index - new_event_start.index).days <= max_days
            start_month = new_event_start.index.month
            
            end_month = new_event_end.index.month
            season_check = (start_month > 5) & (start_month < 11) & (end_month <= 11)

            if (min_time_check) & (max_time_check) & (season_check):
                new_events = new_events.append(new_event_start, sort=True)
                new_events = new_events.append(new_event_end, sort=True)

            last_event_end = pd.to_datetime(this_event_end)
            

    new_events.sort_index(inplace=True)    

    new_events['dt_days'] = new_events.index.to_series().diff(1)
    new_events['wsc_station'] = wsc_station_num
    new_events['training_year'] = training_year
    new_events['training_months'] = str(training_months)# for e in new_events]
    new_events['training_set_len'] = training_set_len
    new_events['m_threshold'] = threshold
    new_events['var_explained'] = var_expl
    new_events['n_components'] = n_components
    new_events['num_lags'] = num_lags
    new_events['radar_stn'] = closest_radar_stn
                
    return new_events, n_components


def adjust_edge_date(initial_date, data, direction, stn_da):
    """
    If the start flow is on a rising limb, adjust the start to the start of the runoff event.
    """
    initial_val = data[data.index == initial_date]['Q']

    
    if direction == 'start':
        search_criteria = (data.index >= initial_date - pd.Timedelta('7 days')) & (data.index <= initial_date)
        search_direction = 1
    elif direction == 'end':
        search_criteria = (data.index <= initial_date + pd.Timedelta('3 days')) & (data.index >= initial_date)
        search_direction = 1

        
    extended_week_vals = data[search_criteria][['Q']]
    extended_week_vals['diff'] = extended_week_vals.diff(periods=search_direction)
    extended_week_vals['pct_change'] = 100 * extended_week_vals['diff'] / extended_week_vals['Q']

    if direction == 'start':
        try:
            change_date = pd.to_datetime(extended_week_vals[['Q']].idxmin().values[0])
            change_point_date = change_date - pd.DateOffset(1)
            adjusted_date = change_point_date
            
        except ValueError as err:
            adjusted_date = initial_date

    elif direction == 'end':
        try:
            adjusted_date = pd.to_datetime(extended_week_vals[['diff']].idxmin().values[0])
        except ValueError as err:
            print('print error in adjusting event end date', err)
            adjusted_date = initial_date
            
    return adjusted_date


def check_peak_in_middle(event, data):
    """
    Ensure there is a peak between the start and end points
    so we aren't targeting a non-runoff event.
    """
    start_time = event.index.values[0] 
    end_time = event.index.values[-1]
    max_time = data[data['Q'] == data['Q'].max()].index.values[0]
    if (max_time == start_time) | (max_time == end_time):
        return False
    else:
        return True


def get_all_combinations(months, years):
    month_combos = [list(itertools.combinations(months, n)) for n in list(range(1, 13))]
    flat_combos =  [item for sublist in month_combos for item in sublist]
    return np.asarray(list(itertools.product(flat_combos, years)))

def calc_softmax(X):
    return np.exp(X) / np.sum(np.exp(X))

def filter_input_data(data):
    filtered = []
    for d in data:
        months = list(d[0])
        common_months = [m for m in months if m in [7, 8, 9, 10]]
        if (len(months) == 1) & (months[0] not in [12, 1, 2, 3]):
            filtered.append(list(d))
        elif (len(months) > 1) & (len(common_months) > 0):
            filtered.append(list(d))
    return filtered

def run_AD_training(wsc_station_num, training_sample_size=5):
    
    radar_stn = stn_df[stn_df['Station Number'] == wsc_station_num]['closest_radar_station'].values[0]
    stn_da = stn_df[stn_df['Station Number'] == wsc_station_num]['DA'].values[0]
    
    lag_df, closest_radar_stn, runoff_df, num_lags = initialize_input_data(wsc_station_num)
    
#     runoff_df = initialize_runoff_dataframe(wsc_station_num)  
    
    training_months = list(set(runoff_df.index.month))
    training_years = list(set(runoff_df.index.year))

    all_combinations = get_all_combinations(training_months, training_years)
    
    filtered_combinations = np.array(filter_input_data(all_combinations))
        
    weights = calc_softmax([(13.0 - len(c[0]))*3.5 for c in filtered_combinations])
    
    # a complete search is intractable, so sample n permutations without replacement
    rand_ints = np.random.choice(range(len(filtered_combinations)), training_sample_size, 
                                 replace=False, p=weights)
    
    initial_pop = [filtered_combinations[i] for i in rand_ints]
    

    input_array = []
    for combo in initial_pop:
        input_data = {'year': combo[1],
                     'months': combo[0],
                     'n_sample': training_sample_size,
                     'wsc_stn': wsc_station_num,
                     'stn_da': stn_da,
                     'radar_stn': radar_stn,
                      'lag_df': lag_df,
                      'num_lags': num_lags,
                      'runoff_df': runoff_df
                     }
        input_array.append(input_data)
        
    results = []
    for input_dat in input_array:
        result, n_components = train_model(input_dat)
        results.append((input_dat, n_components, result))
    
    return results


def initialize_input_data(wsc_stn_num):
        
    t0 = time.time()
    stn_df = initialize_wsc_station_info_dataframe()

    test_stn_info = stn_df[stn_df['Station Number'] == wsc_stn_num]
    stn_da = test_stn_info['DA'].values[0]
    wsc_stn_name = test_stn_info['Station Name'].values[0]
    closest_radar_stn = test_stn_info['closest_radar_station'].values[0]
#     print('{} ({}) has a DA of {} km^2'.format(wsc_stn_name, wsc_stn_num, stn_da))
    
    runoff_df = initialize_runoff_dataframe(wsc_stn_num)    
    lag_df, num_lags = create_lag_df(runoff_df, stn_da) 
    
    
    candidate_stations = stn_df['Station Number'].values
    
    return lag_df, closest_radar_stn, runoff_df, num_lags

def run_AD_from_results(test_stn):

    stn_df = initialize_wsc_station_info_dataframe()

    test_flow_df = get_daily_runoff(test_stn)
    test_flow_df = test_flow_df[test_flow_df.index.year >= 2007]
    test_flow_df.rename(columns={'DAILY_FLOW': 'Q'}, inplace=True)
    test_flow_df = test_flow_df[['Q']]
    
    radar_stn = stn_df[stn_df['Station Number'] == test_stn]['closest_radar_station'].values[0]
    stn_da = stn_df[stn_df['Station Number'] == test_stn]['DA'].values[0]
    
    lag_df, closest_radar_stn, runoff_df, num_lags = initialize_input_data(test_stn)
    
    best_training_params = get_best_result(test_stn)

    months = [m for m in best_training_params[0][1:-1].split(',') if len(m) > 0]
    train_months = [int(m) for m in months]
       
    input_data = {'year': best_training_params[1],
             'months': train_months,
             'n_sample': best_training_params[4],
             'wsc_stn': test_stn,
             'stn_da': stn_da,
             'radar_stn': radar_stn,
              'lag_df': lag_df,
              'num_lags': num_lags,
              'runoff_df': runoff_df
            }
        
    best_events, n_components = train_model(input_data)

    return best_events, test_flow_df, closest_radar_stn

def get_best_result(site):
    ad_df = pd.DataFrame(results_dict[site])
    ad_df.drop(labels='Unnamed: 0', inplace=True, axis=1)
    ad_df.sort_values('len_results', inplace=True, ascending=False)
    return ad_df.iloc[0, :]


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
