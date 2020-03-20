# Dan Kovacek
# www.dkhydrotech.com


# from urllib.request import urlopen
import requests
import os

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

import certifi
import urllib3

urllib3.disable_warnings()

from random import randint
from time import sleep


PROJECT_DIR = os.path.abspath('')
IMG_DIR = os.path.join(PROJECT_DIR, 'data/radar_img')
RADAR_IMG_DIR = os.path.join(PROJECT_DIR, 'data/sorted_radar_images')

def get_img_urls(dts):
    
    datetime = dts[0]
    site_code = dts[1]

    base_url = 'https://climate.weather.gc.ca/radar/image_e.html?'

    # use a duration of 2 hours to get 10 minute resolution data
    duration = '2' # retrieves 1 images for every 10 minutes
    # duration = '6' # retrieves 1 images for every 30 minutes
    duration = '12' # retrieves 1 images for every hour

    date_obj = pd.to_datetime(datetime)

    if date_obj < pd.to_datetime('2013-11-01'):
        image_type = 'PRECIP_RAIN'
    else:
        image_type = 'PRECIPET_RAIN'

    url_extension = 'time={}{:02}{:02}{:02}{:02}&site={}&image_type={}_WEATHEROFFICE'.format(
        date_obj.year,
        date_obj.month,
        date_obj.day,
        date_obj.hour,
        date_obj.minute,
        site_code,
        image_type
    )
            
    return base_url + url_extension


def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta


def get_images(data):
    link = data[0]
    local_folder = data[1]

    session = requests.Session()
    r = requests.get(link)

    sleep(randint(1, 10) / 1000.0)

    local_filename = link.split('&')[0].split('=')[-1]

    img = session.get(link, stream=True, verify=False)

    # img_path = os.path.join(IMG_DIR, wsc_site + '/{}.gif'.format(datetime)) 
    img_path = local_folder + local_filename + '.gif'
                
    with open(img_path, 'wb') as f:
        for chunk in img.iter_content(chunk_size=1024):
            f.write(chunk)

def get_unique_radar_images(radar_stn, dates_all):
    files_to_search = [''.join(e[0].split(' ')[0].split('-') + e[0].split(' ')[1].split(':') + ['.gif']) for e in dates_all]
    existing_files = os.listdir(os.path.join(RADAR_IMG_DIR, radar_stn))
    
    new_files = np.setdiff1d(files_to_search, existing_files)

    # convert back to datetime
    return [(e[:4] + '-' +e[4:6] + '-' + e[6:8] + ' ' + e[8:10] + ':' + e[10:12], radar_stn) for e in new_files]
    

def get_radar_img_urls(events_array, radar_stn_code):
    
    dts_all = []
    for event in events_array:
        # add 8 hours because radar image timestamps 
        # are in UTM by default (PST is -8 UTC + 1 for PDT)
        start = pd.to_datetime(event[0]) + pd.Timedelta(hours=7) - pd.Timedelta(days=1)
        end = pd.to_datetime(event[1]) + pd.Timedelta(hours=7)

        dts_all += [(dt.strftime('%Y-%m-%d %H:%M'), radar_stn_code)
               for dt in datetime_range(start, end, timedelta(hours=1))]
    
    dts_all = [e for e in dts_all if len(e) > 0]

    unique_datetimes = get_unique_radar_images(radar_stn_code, dts_all)

    # initialisation for query queuing
    proc = []
    # frames = []

    # initialize the queue for storing threads
    p = Pool()
    img_url_results = p.map(get_img_urls, unique_datetimes)
    p.close()
    p.join()
    return img_url_results

def request_img_files(img_url_results, wsc_station, radar_station):
        local_folder = 'data/sorted_radar_images/{}/'.format(radar_station)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
            
        flat_url_list = []
        for sub in img_url_results:
            flat_url_list += [(sub, local_folder)]

        proc = []
        frames = []
        p = Pool()

        print('For {} there are {} images to be requested from the {} radar station.'.format(wsc_station, len(flat_url_list), radar_station))

        # write all images to folders
        # each event gets a separate folder
        p.map(get_images, flat_url_list)

        p.close()
        p.join()

# test_stn =  '08MG001'
# best_radar_stn = 'CASAG'
# event_pairs = [['2007-07-20', '2007-07-25'], ['2007-11-02', '2007-11-06'], ['2008-06-20', '2008-06-23'], ['2008-06-27', '2008-07-04'], 
#                 ['2008-08-17', '2008-08-23'], ['2008-08-26', '2008-08-31'], ['2008-10-04', '2008-10-09'], ['2008-10-13', '2008-10-19'], 
#                 ['2009-06-21', '2009-06-28'], ['2009-08-31', '2009-09-09'], ['2009-10-14', '2009-10-22'], ['2009-10-25', '2009-10-27'], 
#                 ['2009-10-28', '2009-11-10'], ['2010-09-11', '2010-09-14'], ['2010-09-18', '2010-09-23'], ['2010-09-25', '2010-10-08'], 
#                 ['2010-10-08', '2010-10-13'], ['2010-10-23', '2010-10-28'], ['2010-10-31', '2010-11-12'], ['2010-11-14', '2010-11-18'], 
#                 ['2011-06-21', '2011-06-26'], ['2011-06-28', '2011-07-02'], ['2011-07-02', '2011-07-09'], ['2011-07-14', '2011-07-17'], 
#                 ['2011-07-20', '2011-07-23'], ['2011-09-21', '2011-09-29'], ['2011-10-09', '2011-10-20'], ['2011-11-16', '2011-11-25'], 
#                 ['2011-11-26', '2011-12-09'], ['2012-06-07', '2012-06-09'], ['2012-06-28', '2012-07-05'], ['2012-07-08', '2012-07-11'], 
#                 ['2013-06-05', '2013-06-08'], ['2013-06-15', '2013-06-22'], ['2013-06-26', '2013-06-28'], ['2013-09-21', '2013-09-25'], 
#                 ['2013-11-01', '2013-11-10'], ['2013-11-12', '2013-11-20'], ['2013-11-29', '2013-12-03']]
# new_img_urls = get_radar_img_urls(event_pairs, best_radar_stn)
# request_img_files(new_img_urls, test_stn, best_radar_stn)
