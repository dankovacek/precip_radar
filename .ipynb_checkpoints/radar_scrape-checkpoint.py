# Dan Kovacek
# www.dkhydrotech.com


# from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from random import randint
from time import sleep

import requests
import os
from multiprocessing import Process, Queue, Pool

def get_img_urls(datetime, site_code):

    base_url = 'http://climate.weather.gc.ca/radar/index_e.html?'

    # use a duration of 2 hours to get 10 minute resolution data
    duration = '2' # retrieves 1 images for every 10 minutes
    duration = '6' # retrieves 1 images for every 30 minutes
    duration = '12' # retrieves 1 images for every hour

    date_obj = pd.to_datetime(datetime)
    site_code = 'CASAG'
    url_extension = 'site={}&year={}&month={}&day={}&hour={}&minute={}&duration={}&image_type=PRECIPET_RAIN_WEATHEROFFICE'.format(
        site_code,
        date_obj.year,
        date_obj.month,
        date_obj.day,
        date_obj.hour,
        date_obj.minute,
        duration
    )

    r = requests.get(base_url + url_extension)
    data = r.text

    soup = BeautifulSoup(data, 'lxml')

    # print(soup)

    session = requests.Session()

    image_links = []
    pattern = 'blobArray'
    for s in soup.find_all('script'):
        if pattern in s.text:
            script_string = s.text
            start = 'blobArray'
            end = 'index_end'
            target = script_string[script_string.find(
                start):script_string.find(end)][:-5]
            image_links = list(target.split('\''))
            image_links = [
                e for e in image_links if '/radar/image_e.html?time=' in e]
            
    print(image_links)
    return image_links


def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta


def get_images(image_links):
    
    for l in image_links:
        session = requests.Session()
        r = requests.get('http://climate.weather.gc.ca/' + l)

        sleep(randint(1, 10) / 100.0)

        local_filename = l.split(
            '/')[-1].split('?')[-1].split('&')[0].split('=')[-1]

        print(local_filename)
        local_folder = local_filename.split('+')[0]
        print(local_folder)
        print(break_here)

        if not os.path.isdir(local_folder):
            os.makedirs(local_folder)

        img = session.get('http://climate.weather.gc.ca/' +
                          l, stream=True, verify=False)
        with open(local_folder + '/' + local_filename, 'wb') as f:
            for chunk in img.iter_content(chunk_size=1024):
                f.write(chunk)

def get_radar_images(events_array, site_code):
    for event in events_array:
        start = pd.to_datetime(event[0])
        end = pd.to_datetime(event[1])

        dts = [dt.strftime('%Y-%m-%d %H:%M')
               for dt in datetime_range(start, end, timedelta(hours=2))]

        image_urls = []
        image_urls.append([get_img_urls(dt, site_code) for dt in dts])

        # initialisation for query queuing
        proc = []
        frames = []
        # initialize the queue for storing threads
        p = Pool()

        # store results in a list until all requests complete
        # before concatenating
        img_url_results = p.map(get_img_urls, dts)
        p.close()
        p.join()

        proc = []
        frames = []
        p = Pool()

        # write all images to folders
        # each event gets a separate folder
        img_scrape = p.map(get_images, img_url_results)

        p.close()
        p.join()

        
# get_radar_images('CASAG', [['2010-09-09 00:00:00', '2010-09-17 00:00:00']])