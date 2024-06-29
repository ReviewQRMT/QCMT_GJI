#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import requests
# import hashlib
# from obspy import UTCDateTime
from obspy.clients.seedlink.rmt_client import Client as SL_client
# from obspy.clients.seedlink.seedlinkexception import SeedLinkException
from os.path import exists, getmtime
from datetime import timezone
from datetime import datetime as dt
# from datetime import timedelta as td
# from urllib.request import urlopen
# from urllib.error import HTTPError, URLError
from urllib3.exceptions import ProtocolError
from requests.exceptions import HTTPError, ConnectionError, RequestException, Timeout
from http.client import RemoteDisconnected
from modules_MGMT.confidential_materials import bmkg_seedlink_vars, pgr9_seedlink_vars
from modules_CMT.extras import read_local_cache, write_local_cache


def get_seedlink_stations(seedlink_host, seedlink_port=18000, all_sta='input/rmt_all.stn',
                          out_sta='input/rmt_stations.stn', filter_channel=('BH', 'SH', 'HH')):
    sl_client = SL_client(seedlink_host, port=seedlink_port)
    list_sts = sl_client.get_info(level='channel', cache=True)

    sts = open(all_sta, 'r')
    lines = sts.readlines()
    sts.close()

    out_sts = open(out_sta, 'w')
    out_sts.write('sts     lat     lon\n')

    for line in lines:
        if not line.strip():
            continue
        items = line.split()
        sta, lat, lon = items[0:3]
        l = sta.split(':')
        network = l[0]
        sta = l[1]
        location = l[2]
        channelcode = l[3]

        if channelcode in filter_channel:
            if (network, sta, location, channelcode + "Z") in list_sts \
                    and (network, sta, location, channelcode + "N") in list_sts \
                    and (network, sta, location, channelcode + "E") in list_sts:
                out_sts.write(line)

    out_sts.close()


def initiate_seedlink_cache(cache_file, station_file, time_out=20, local=False, filter_channel=('BH', 'HH', 'SH')):
    if local:
        host, seedlink_port = pgr9_seedlink_vars()
    else:
        host, seedlink_port = bmkg_seedlink_vars()
    # host = "geof.bmkg.go.id"
    sl_client = SL_client(host, port=seedlink_port, timeout=time_out)

    if exists(cache_file):
        cache_modif_date = dt.fromtimestamp(getmtime(cache_file), timezone.utc)
        station_modif_date = dt.fromtimestamp(getmtime(station_file), timezone.utc)
        cache_age = dt.now(timezone.utc) - cache_modif_date
        if cache_age.total_seconds() / 86400 > 7 or station_modif_date > cache_modif_date:
            print(f'\nCaching SeedLink Station Info . . .')
            sts_cache = sl_client.get_info(network='*', station='*', location='*', channel='*', level='channel',
                                           cache=True)
            filtered = [x for x in sts_cache if filter_channel[0] in x[3] or
                        filter_channel[1] in x[3] or filter_channel[2] in x[3]]
            sl_client._station_cache = set(filtered)
            # print(f'{dt.now(timezone.utc).strftime("%H:%M:%S")}\n')
            write_local_cache({'st_cache': filtered, 'sl_client': sl_client._slclient,
                               'cache_level': sl_client._station_cache_level}, cache_file)
        else:
            local_cache = read_local_cache(cache_file)
            # sl_client._slclient = set(local_cache['sl_client'])
            sl_client._station_cache = set(local_cache['st_cache'])
            sl_client._station_cache_level = local_cache['cache_level']
    else:
        print(f'\nCaching SeedLink Station Info . . .')
        sts_cache = sl_client.get_info(network='*', station='*', location='*', channel='*', level='channel',
                                       cache=True)
        filtered = [x for x in sts_cache if filter_channel[0] in x[3] or
                    filter_channel[1] in x[3] or filter_channel[2] in x[3]]
        sl_client._station_cache = set(filtered)
        write_local_cache({'st_cache': filtered, 'sl_client': sl_client._slclient,
                           'cache_level': sl_client._station_cache_level}, cache_file)
    return sl_client


def initiateAPIconnection(url_api, max_retry=10):
    try:
        session = requests.Session()
        current_resp = session.get(url_api).json()
    except (Timeout, HTTPError, RemoteDisconnected, ProtocolError, ConnectionError, RequestException) as err:
        retry_num = 1
        delay_time = 5
        while True:
            time.sleep(delay_time * retry_num)
            print(f'{err}. Retrying in {delay_time * retry_num} seconds...')
            err, response = retryingAPIsession(url_api)
            if err is None:
                session, current_resp = response
                print('Reconnection successful...')
                return session, current_resp
            if retry_num == max_retry:
                sys.exit('Connection error. Please restart the program\n')
            retry_num += 1
    return session, current_resp


def retryingAPIsession(url_api):
    e = None
    try:
        session = requests.Session()
        current_resp = session.get(url_api).json()
    except (Timeout, HTTPError, RemoteDisconnected, ProtocolError, ConnectionError, RequestException) as e:
        return e, None
    return e, (session, current_resp)


def checkAPIchanges(request_session, url_api, current_response):
    t = int(dt.now().timestamp() * 1e3)
    url_api = f'{url_api}?t={t}'  # add "? ..." to prevent cache
    try:
        response = request_session.get(url_api)
    except (Timeout, HTTPError, RemoteDisconnected, ProtocolError, ConnectionError, RequestException):
        print('Error getting API data. Check Connection\n')
        return False, current_response
    else:
        new_response = response.json()
        if new_response == current_response:
            return False, new_response
        else:
            return True, new_response


# def keep_SeedLinkCon(seedlink_client):
#     t0 = UTCDateTime(dt.now(timezone.utc) - td(minutes=5))
#     t1 = UTCDateTime(dt.now(timezone.utc) - td(minutes=4))
#     try:
#         st = seedlink_client.get_waveforms(network='IA', station='AAII', location='*', channel='*',
#                                            starttime=t0, endtime=t1)
#     except (SeedLinkException, TypeError):
#         pass
