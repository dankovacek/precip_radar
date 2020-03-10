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

from multiprocessing import Process, Queue, Pool

PROJECT_DIR = os.path.abspath('')
IMG_DIR = os.path.join(PROJECT_DIR, 'data/radar_img')

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


def get_radar_img_urls(events_array, site_code):
    dts_all = []
    for event in events_array:
        # add 8 hours because radar image timestamps 
        # are in UTM by default (PST is -8 UTC + 1 for PDT)
        start = pd.to_datetime(event[0]) + pd.Timedelta(hours=7) - pd.Timedelta(days=1)
        end = pd.to_datetime(event[1]) + pd.Timedelta(hours=7)

        dts_all += [(dt.strftime('%Y-%m-%d %H:%M'), site_code)
               for dt in datetime_range(start, end, timedelta(hours=1))]
    
    dts_all = [e for e in dts_all if len(e) > 0]
    # initialisation for query queuing
    proc = []
    frames = []

    # initialize the queue for storing threads
    p = Pool()
    img_url_results = p.map(get_img_urls, dts_all)
    p.close()
    p.join()
    return img_url_results

def request_img_files(img_url_results, wsc_station):
        local_folder = 'data/radar_img/{}/'.format(wsc_station)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
            
        flat_url_list = []
        for sub in img_url_results:
            flat_url_list += [(sub, local_folder)]

        proc = []
        frames = []
        p = Pool()

        print('For {} there are {} images.'.format(wsc_station, len(flat_url_list)))

        # write all images to folders
        # each event gets a separate folder
        p.map(get_images, flat_url_list)

        p.close()
        p.join()

# test_stn =  '08GA075'
# best_radar_stn = 'CASAG'
# event_pairs = [('2010-09-09', '2010-09-10'), ('2011-09-09', '2011-09-10')]
# all_img_urls = get_radar_img_urls(event_pairs, best_radar_stn)
# request_img_files(all_img_urls, test_stn)
