#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os.path
import pandas as pd
import numpy as np
import multiprocessing as mp
from synt_CMT.multiprocessings import seedlink_mp
from synt_CMT._pre_process import dist_mag_weight
from synt_CMT.geometry import azimuth_gap, divide_sector_edge
from pyrocko.gf import LocalEngine
from obspy.core import AttribDict
from obspy import UTCDateTime, read, Stream, read_inventory
from obspy.geodetics.base import gps2dist_azimuth, degrees2kilometers
from obspy.clients.fdsn.client import Client
from obspy.clients.seedlink.rmt_client import Client as SL_client
from obspy.clients.seedlink.seedlinkexception import SeedLinkException
from datetime import datetime as dt
# from datetime import timedelta as td
from datetime import timezone


def vars_by_mag(self):
    """
    Set several inversion parameters setting affected by magnitudo
    :return: minsnr, mindist, maxdist, covariance_noise, t_before (using noise or not)
    """
    mag = self.event['mag']
    # if float(mag) >= 5.2:
    #     covariance_noise = False
    #     t_before = 30  # s  waveforms time range for processing
    #     minsnr = 25.  # minimum SNR value for used trace
    # el
    if float(mag) >= 5:
        covariance_noise = False
        t_before = 100  # s  waveforms time range for processing
        minsnr = 2.  # minimum SNR value for used trace
        # self.seismogram = 'displacement'
        # self.seisunit = 'DISP'
        # self.unit = 'displacement [m]'
    else:
        covariance_noise = True
        t_before = None
        minsnr = 1.5  # minimum SNR value for used trace
        # self.seismogram = 'velocity'
        # self.seisunit = 'VEL'
        # self.unit = 'velocity [m/s]'

    if float(mag) <= 4.:
        mindist = 0.7  # deg  station filter by radius
        maxdist = 6  # deg
    elif float(mag) <= 4.5:
        mindist = 0.8
        maxdist = 7
    elif float(mag) <= 5:
        mindist = 1
        maxdist = 7.5
    elif float(mag) <= 5.5:
        mindist = 1.2
        maxdist = 8
    else:
        mindist = 1.4
        maxdist = 8.5

    # else:
    #     mindist = 0.8
    #     maxdist = 3.5

    # for moderate to large EQ using only near station to limit used length data
    if UTCDateTime(dt.now(timezone.utc)) - self.event['t'] < 200:
        if float(mag) > 5:
            mindist = 0.8
            maxdist = 3.5
    elif UTCDateTime(dt.now(timezone.utc)) - self.event['t'] < 250:
        if float(mag) > 5:
            mindist = 1
            maxdist = 4.5
    elif UTCDateTime(dt.now(timezone.utc)) - self.event['t'] < 300:
        if float(mag) > 5:
            mindist = 1
            maxdist = 5.5
    elif UTCDateTime(dt.now(timezone.utc)) - self.event['t'] < 350:
        if float(mag) > 5:
            mindist = 1
            maxdist = 6.5

    if self.snr_min != 'auto':
        minsnr = self.snr_min
    if self.use_cov_noise != 'auto':
        covariance_noise = self.use_cov_noise
        if covariance_noise is True:
            t_before = None
    if self.stn_max_dist != 'auto':
        maxdist = self.stn_max_dist
    if self.stn_min_dist != 'auto':
        mindist = self.stn_min_dist

    return minsnr, mindist, maxdist, covariance_noise, t_before


def set_event_info(self, ID, lat, lon, depth, mag, t, remark):
    """
    Sets event coordinates, magnitude, and time from parameters given to this function
    :param self:
    :param ID: seiscomp ID
    :param lat: event latitude
    :type lat: float
    :param lon: event longitude
    :type lon: float
    :param depth: event depth in km
    :type lat: float
    :param mag: event moment magnitude
    :type lat: float
    :param t: event origin time
    :type t: :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime` or string
    :param remark: earthquake region location
    :type lat: string, optional
    """

    if type(t) == str:
        t = UTCDateTime(t)
    self.event = {'id': ID, 'lat': lat, 'lon': lon, 'depth': depth * 1e3, 'mag': mag, 't': t,
                  'remark': remark, 'agency': self.agency}
    self.log('\nHypocenter location:\n  Agency: {agency:s}\n  Origin time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   '
             'Depth {d:4.1f} km   Mag {m:4.1f}   {r:s}'.format(t=t.strftime('%Y-%m-%d %H:%M:%S'), lat=lat, lon=lon,
                                                               d=depth, m=mag, r=remark, agency=self.agency))


def read_stations(self, filename, min_dist=0, max_dist=180, max_cluster_num=0, delimiter=' '):
    """
    Read stations information from file.
    Calculate their distances and azimuthes using haversine.
    Calculate wave arrival using pyrocko engine.
    Create data structure ``self.stations``. Sorts it according to station epicentral distance.
    :param self:
    :param filename: path to file with network coordinates
    :type filename: string
    :param min_dist: minimum distance for used station in degrees
    :param max_dist: maximum distance for used station in degrees
    :param max_cluster_num: maximum number of stations for each octdrant sector
    """
    self.logtext['network'] = s = 'Station data: ' + filename
    self.log(s)

    df = pd.read_csv(filename, skipinitialspace=True, delimiter=delimiter)
    df[['dist', 'az', 'baz']] = df.apply(lambda x: gps2dist_azimuth(float(self.event['lat']),
                                                                    float(self.event['lon']), float(x['lat']),
                                                                    float(x['lon'])), axis=1, result_type='expand')
    df.reset_index(inplace=True)
    df.drop_duplicates(subset='sts', inplace=True)
    df = df.loc[(df['dist'] >= degrees2kilometers(min_dist) * 1000) &
                (df['dist'] <= degrees2kilometers(max_dist) * 1000)]
    df = df.sort_values(by=['dist', 'index'])

    if max_cluster_num:
        df[['weight']] = df.apply(lambda x: dist_mag_weight(x['dist'], self.event['mag']), axis=1)
        list_azimuth = df['az'].tolist()
        gap, start, end = azimuth_gap(list_azimuth)
        lst_qdr_edge = divide_sector_edge(start, end, 8)

        select_sts = pd.DataFrame()
        for i in range(len(lst_qdr_edge) - 1):
            if lst_qdr_edge[i] == max(lst_qdr_edge):
                sel_sta = df.loc[(df['az'] > lst_qdr_edge[i]) | (df['az'] <= lst_qdr_edge[i + 1])]
            else:
                sel_sta = df.loc[(df['az'] > lst_qdr_edge[i]) & (df['az'] <= lst_qdr_edge[i + 1])]

            if len(sel_sta) == 0:
                continue
            elif len(sel_sta) > max_cluster_num:
                sel_sta = sel_sta.sort_values(by=['weight'], ascending=False)
                sel_sta = sel_sta.head(max_cluster_num)

            select_sts = pd.concat([select_sts, sel_sta])
        df = select_sts.sort_values(by=['dist', 'index'])

    df.reset_index(inplace=True)
    engine = LocalEngine(store_superdirs=[self.GF_store_dir])
    store = engine.get_store(self.GF_store)
    # if isinstance(self.GF_stores, list):
    for gfstore in self.GF_stores:
        stre = engine.get_store(gfstore)
        self.store_samprate = min(stre.config.sample_rate, self.store_samprate)
        self.store_mindepth = max(stre.config.source_depth_min, self.store_mindepth)
        self.store_maxdepth = min(stre.config.source_depth_max, self.store_maxdepth)
    stations = []
    list_net = ''
    list_sta = ''
    # list_loc = ''
    list_cha = ''
    for index, row in df.iterrows():
        sta = row['sts']
        l = sta.split(':')
        if l[3] == 'BH' or l[3] == 'HH' or l[3] == 'SS':
            weight = 1.
        else:
            weight = 0.8
        stn = {'code': l[1], 'lat': row['lat'], 'lon': row['lon'], 'network': l[0], 'location': l[2],
               'channelcode': l[3], 'model': self.GF_store, 'dist': row['dist'], 'az': row['az'], 'useN': False,
               'useE': False, 'useZ': False, 'weightN': weight, 'weightE': weight, 'weightZ': weight,
               'arr_time': store.t('begin', (self.event['depth'], row['dist']))}
        stations.append(stn)
        if l[0] not in list_net:
            list_net = f'{list_net},{l[0]}'
        if l[1] not in list_sta:
            list_sta = f'{list_sta},{l[1]}'
        # if l[2] not in list_loc:
        #     list_loc = f'{list_loc},{l[2]}'
        if f'{l[3]}*' not in list_cha:
            list_cha = f'{list_cha},{l[3]}*'

    # stations = sorted(stations, key=lambda stn: stn['dist'])
    self.stations = stations
    self.list_net = list_net[1:]
    self.list_sta = list_sta[1:]
    self.list_cha = list_cha[1:]
    self.check_a_station_present()
    # self.list_loc = list_loc[1:]
    self.create_station_index()
    # self.all_stations = deepcopy(stations)


def create_station_index(self):
    """
    Creates ``self.stations_index`` which serves for accesing ``self.stations`` items by the station name.
    Called from :func:`read_stations`.
    """
    self.stations_index = {}
    sts = self.stations
    self.nr = len(sts)
    for i in range(self.nr):
        self.stations_index['_'.join([sts[i]['network'], sts[i]['code'], sts[i]['location'],
                                      sts[i]['channelcode']])] = sts[i]


def check_a_station_present(self):
    """
    Checks whether at least one station is present, otherwise raises error.
    Called from :func:`load_streams_ArcLink` and :func:`load_files`.
    """
    if not len(self.stations):
        self.log('No station present. Exiting...')
        raise ValueError('No station present.')


def waveform_time_window(self, t_before, t_after, min_length=200):
    if t_after is None:
        if int(np.ceil(self.stations[-1]['dist'] / self.max_t_v)) > min_length:
            t_after = int(np.ceil(self.stations[-1]['dist'] / self.max_t_v))
        else:
            t_after = min_length
    if t_before is None:
        t_before = t_after * 1.7
    if t_before < 30:
        t_before = 30
    return t_before, t_after


def load_fsdn(self, url, user, password, t_before=None, t_after=None):
    """
    Loads waveform from fsdn for stations listed in ``self.stations``.
    :param self:
    :param url:
    :param user:
    :param password:
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    """
    self.logtext['data'] = s = 'Fetching data from FSDN.'
    self.log('\n' + s)

    fdsn_bmkg = Client(base_url=url, user=user, password=password, force_redirect=False)

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t = self.event['t']
    t0 = t - t_before
    t1 = t + t_after
    # if not self.data_are_corrected:
    #     inv = read_inventory(xml_file)
    self.inv = fdsn_bmkg.get_stations(network=self.list_net, sta=self.list_sta, loc="*", channel=self.list_cha,
                                      level="response", starttime=t0, endtime=t1)
    streams = fdsn_bmkg.get_waveforms(network=self.list_net, station=self.list_sta, location="*", channel=self.list_cha,
                                      starttime=t0, endtime=t1)

    i = 0
    data_raw = []
    data_deltas = []
    while i < len(self.stations):
        sta = self.stations[i]['code']
        net = self.stations[i]['network']
        loc = self.stations[i]['location']
        cha = self.stations[i]['channelcode']

        if len(data_raw) > 0:
            if sta == data_raw[-1][0].stats.station:
                self.stations.pop(i)
                continue

        st = streams.select(network=net, station=sta, location=loc, channel=f'{cha}*')

        if not st:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Waveform not Available. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue

        if len(st) != 3:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue
        ch = {}
        for comp in range(3):
            ch[st[comp].stats.channel[2]] = st[comp]
        if sorted(ch.keys()) != ['E', 'N', 'Z']:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Unoriented components. '
                     'Removing station from further processing.'.format(net, sta, cha))
            continue
        str_comp = self.comps_order
        st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
        data_raw.append(st)
        self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True

        if not self.data_are_corrected:
            for tr in data_raw[-1]:
                filt_resp = self.inv.select(network=tr.stats.network, station=tr.stats.station,
                                            channel=tr.stats.channel, time=self.event['t'])
                if len(filt_resp) == 0:
                    data_raw.pop()
                    self.stations.pop(i)
                    self.log(
                        'Cannot find response metadata(s) for station {0:s}:{1:s}:{2:s}. Removing station from '
                        'further processing.'.format(net, sta, cha), printcopy=True)
                    break
                response = filt_resp[0][0][0].response
                poles = response.response_stages[0].poles
                zeros = response.response_stages[0].zeros
                norm_fact = response.response_stages[0].normalization_factor
                inst_sens = response.instrument_sensitivity.value
                # stage_gain = response.response_stages[0].stage_gain

                tr.stats.paz = AttribDict({
                    'sensitivity': inst_sens,
                    'poles': poles,
                    'gain': norm_fact,
                    'zeros': zeros
                })
            else:
                i += 1  # station not removed
                if not st[0].stats.delta in data_deltas:
                    data_deltas.append(st[0].stats.delta)
        else:
            i += 1

    self.data_raw = data_raw
    self.data_deltas = data_deltas
    self.create_station_index()
    self.check_a_station_present()


def load_seedlink_mp(self, sl_client, fsdn_url, fsdn_user, fsdn_pass, t_before=None, t_after=None):
    """
    Loads waveform from seedlink for stations listed in ``self.stations``.

    :param self:
    :param sl_client:
    :param fsdn_url:
    :param fsdn_user:
    :param fsdn_pass:
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    """
    self.logtext['data'] = s = 'Fetching data from SeedLink.'
    self.log('\n' + s)

    fdsn_bmkg = Client(base_url=fsdn_url, user=fsdn_user, password=fsdn_pass,
                       force_redirect=False)
    # sl_client = SL_client(seedlink_host, port=port, timeout=timeout)

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t = self.event['t']
    t0 = t - t_before
    t1 = t + t_after
    # if not self.data_are_corrected:
    #     inv = read_inventory(xml_file)
    self.inv = fdsn_bmkg.get_stations(starttime=t0, endtime=t1, network=self.list_net, sta=self.list_sta,
                                      loc="*", channel=self.list_cha, level="response")

    data_raw = []
    data_deltas = []
    stations = []

    pool = mp.Pool(processes=self.threads)
    results = [pool.apply_async(seedlink_mp, args=(i, station, sl_client, t0, t1, self.comps_order, self.inv, t))
               for i, station in zip(range(len(self.stations)), self.stations)]
    output = [p.get() for p in results]
    output = sorted(output, key=lambda d: d['sts_n'])

    for data in output:

        if data['desc'] != 'data ok':
            self.log(data['desc'])
            continue

        stations.append(data['sts'])

        if len(data_raw) > 0:
            if data['st'][0].stats.station == data_raw[-1][0].stats.station:
                # self.stations.pop(i)
                continue
        data_raw.append(data['st'])
        if not data['st'][0].stats.delta in data_deltas:
            data_deltas.append(data['st'][0].stats.delta)

    self.stations = stations
    self.data_raw = data_raw
    self.data_deltas = data_deltas
    self.create_station_index()
    self.check_a_station_present()


def load_seedlink(self, seedlink_host, port, timeout, fsdn_url, fsdn_user, fsdn_pass, t_before=None, t_after=None):
    """
    Loads waveform from seedlink for stations listed in ``self.stations``.

    :param self:
    :param seedlink_host:
    :param port:
    :param timeout:
    :param fsdn_url:
    :param fsdn_user:
    :param fsdn_pass:
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    """
    self.logtext['data'] = s = 'Fetching data from SeedLink.'
    self.log('\n' + s)

    fdsn_bmkg = Client(base_url=fsdn_url, user=fsdn_user, password=fsdn_pass, force_redirect=False)
    sl_client = SL_client(seedlink_host, port=port, timeout=timeout)

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t = self.event['t']
    t0 = t - t_before
    t1 = t + t_after
    # if not self.data_are_corrected:
    #     inv = read_inventory(xml_file)
    self.inv = fdsn_bmkg.get_stations(network=self.list_net, sta=self.list_sta, loc="*", channel=self.list_cha,
                                      level="response", starttime=t0, endtime=t1)
    try:
        streams = sl_client.get_waveforms(network=self.list_net, station=self.list_sta, location="*",
                                          channel=self.list_cha, starttime=t0, endtime=t1)
    except SeedLinkException:
        self.log('Waveform not Available.')

    i = 0
    data_raw = []
    data_deltas = []
    while i < len(self.stations):
        sta = self.stations[i]['code']
        net = self.stations[i]['network']
        loc = self.stations[i]['location']
        cha = self.stations[i]['channelcode']

        if len(data_raw) > 0:
            if sta == data_raw[-1][0].stats.station:
                self.stations.pop(i)
                continue

        st = streams.select(network=net, station=sta, location=loc, channel=f'{cha}*')

        if not st:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Waveform not Available. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue

        if len(st) != 3:  # todo: check for components beside ZNE
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue
        ch = {}
        if st[0].stats.location != loc:  # seedlink not filter data location (obspy seedlink/server bugs?)
            self.stations[i]['location'] = st[0].stats.location
        for comp in range(3):
            ch[st[comp].stats.channel[2]] = st[comp]
        if sorted(ch.keys()) != ['E', 'N', 'Z']:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Unoriented components. '
                     'Removing station from further processing.'.format(net, sta, cha))
            continue
        str_comp = self.comps_order
        st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
        data_raw.append(st)
        self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True

        if not self.data_are_corrected:
            for tr in data_raw[-1]:
                filt_resp = self.inv.select(network=tr.stats.network, station=tr.stats.station,
                                            channel=tr.stats.channel, time=self.event['t'])
                if len(filt_resp) == 0:
                    data_raw.pop()
                    self.stations.pop(i)
                    self.log(
                        'Cannot find responses metadata(s) for station {0:s}:{1:s}:{2:s}. Removing station from '
                        'further processing.'.format(net, sta, cha), printcopy=True)
                    break
                response = filt_resp[0][0][0].response
                try:
                    poles = response.response_stages[0].poles
                    zeros = response.response_stages[0].zeros
                    norm_fact = response.response_stages[0].normalization_factor
                    inst_sens = response.instrument_sensitivity.value
                    # stage_gain = response.response_stages[0].stage_gain
                except IndexError:
                    data_raw.pop()
                    self.stations.pop(i)
                    self.log(
                        'Responses metadata(s) for station {0:s}:{1:s}:{2:s} is not complete. Removing station from '
                        'further processing.'.format(net, sta, cha), printcopy=True)
                    break

                tr.stats.paz = AttribDict({
                    'sensitivity': inst_sens,
                    'poles': poles,
                    'gain': norm_fact,
                    'zeros': zeros
                })
            else:
                i += 1  # station not removed
                if not st[0].stats.delta in data_deltas:
                    data_deltas.append(st[0].stats.delta)
        else:
            i += 1

    self.data_raw = data_raw
    self.data_deltas = data_deltas
    self.create_station_index()
    self.check_a_station_present()


def load_fsdn_loop(self, url, user, password, t_before=None, t_after=None):
    """
    Loads waveform from fsdn for stations listed in ``self.stations``.
    :param self:
    :param url:
    :param user:
    :param password:
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    """
    self.logtext['data'] = s = 'Fetching data from FSDN.'
    self.log('\n' + s)

    fdsn_bmkg = Client(base_url=url, user=user, password=password,
                       force_redirect=False)

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t = self.event['t']
    t0 = t - t_before
    t1 = t + t_after
    # if not self.data_are_corrected:
    #     inv = read_inventory(xml_file)
    self.inv = fdsn_bmkg.get_stations(starttime=t0, endtime=t1, network=self.list_net, sta=self.list_sta,
                                      loc="*", channel=self.list_cha, level="response")

    i = 0
    data_raw = []
    data_deltas = []
    while i < len(self.stations):
        sta = self.stations[i]['code']
        net = self.stations[i]['network']
        loc = self.stations[i]['location']
        cha = self.stations[i]['channelcode']

        if len(data_raw) > 0:
            if sta == data_raw[-1][0].stats.station:
                self.stations.pop(i)
                continue
        try:
            # st = fdsn_bmkg.get_waveforms(network=net, station=sta, location=loc, channel=f'{cha}*',
            #                              starttime=t0, endtime=t1, attach_response=True)
            st = fdsn_bmkg.get_waveforms(network=net, station=sta, location=loc, channel=f'{cha}*',
                                         starttime=t0, endtime=t1)
        except Exception as e:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Waveform not Available {3:s}. Removing station from further '
                     'processing.'.format(net, sta, cha, e))
            continue

        if len(st) != 3:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue
        ch = {}
        for comp in range(3):
            ch[st[comp].stats.channel[2]] = st[comp]
        if sorted(ch.keys()) != ['E', 'N', 'Z']:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Unoriented components. '
                     'Removing station from further processing.'.format(net, sta, cha))
            continue
        str_comp = self.comps_order
        st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
        data_raw.append(st)
        self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True

        if not self.data_are_corrected:
            for tr in data_raw[-1]:
                filt_resp = self.inv.select(network=tr.stats.network, station=tr.stats.station,
                                            channel=tr.stats.channel, time=self.event['t'])
                if len(filt_resp) == 0:
                    data_raw.pop()
                    self.stations.pop(i)
                    self.log(
                        'Cannot find response metadata(s) for station {0:s}:{1:s}:{2:s}. Removing station from '
                        'further processing.'.format(net, sta, cha), printcopy=True)
                    break
                response = filt_resp[0][0][0].response
                poles = response.response_stages[0].poles
                zeros = response.response_stages[0].zeros
                norm_fact = response.response_stages[0].normalization_factor
                inst_sens = response.instrument_sensitivity.value
                # stage_gain = response.response_stages[0].stage_gain

                tr.stats.paz = AttribDict({
                    'sensitivity': inst_sens,
                    'poles': poles,
                    'gain': norm_fact,
                    'zeros': zeros
                })
            else:
                i += 1  # station not removed
                if not st[0].stats.delta in data_deltas:
                    data_deltas.append(st[0].stats.delta)
        else:
            i += 1

    self.data_raw = data_raw
    self.data_deltas = data_deltas
    self.create_station_index()
    self.check_a_station_present()


def load_seedlink_loop(self, sl_client, fsdn_url, fsdn_user, fsdn_pass, t_before=None, t_after=None):
    """
    Loads waveform from seedlink for stations listed in ``self.stations``.
    :param self:
    :param sl_client:
    :param fsdn_url:
    :param fsdn_user:
    :param fsdn_pass:
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    """
    self.logtext['data'] = s = 'Fetching data from SeedLink.'
    self.log('\n' + s)

    fdsn_bmkg = Client(base_url=fsdn_url, user=fsdn_user, password=fsdn_pass,
                       force_redirect=False)
    # sl_client = SL_client(seedlink_host, port=port, timeout=timeout)

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t = self.event['t']
    t0 = t - t_before
    t1 = t + t_after
    # if not self.data_are_corrected:
    #     inv = read_inventory(xml_file)
    self.inv = fdsn_bmkg.get_stations(starttime=t0, endtime=t1, network=self.list_net, sta=self.list_sta,
                                      loc="*", channel=self.list_cha, level="response")

    i = 0
    data_raw = []
    data_deltas = []
    while i < len(self.stations):
        sta = self.stations[i]['code']
        net = self.stations[i]['network']
        loc = self.stations[i]['location']
        cha = self.stations[i]['channelcode']

        if len(data_raw) > 0:
            if sta == data_raw[-1][0].stats.station:
                self.stations.pop(i)
                continue
        try:
            st = sl_client.get_waveforms(network=net, station=sta, location=loc, channel=f'{cha}*',
                                         starttime=t0, endtime=t1)
            # st = sl_client.get_waveforms(network=net, station=sta, location=loc, channel=f'{cha}*',
            #                              starttime=t0, endtime=t1, metadata=True)
        except (SeedLinkException, TypeError):
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Waveform not Available. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue

        if len(st) != 3:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue

        if st[0].stats.location != loc:  # seedlink not filter data location (obspy seedlink/server bugs?)
            self.stations[i]['location'] = st[0].stats.location

        ch = {}
        for comp in range(3):
            ch[st[comp].stats.channel[2]] = st[comp]
        if sorted(ch.keys()) != ['E', 'N', 'Z']:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Unoriented components. '
                     'Removing station from further processing.'.format(net, sta, cha))
            continue
        str_comp = self.comps_order
        st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
        data_raw.append(st)
        self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True

        if not self.data_are_corrected:
            for tr in data_raw[-1]:
                filt_resp = self.inv.select(location=tr.stats.location, network=tr.stats.network,
                                            station=tr.stats.station, channel=tr.stats.channel, time=self.event['t'])
                if len(filt_resp) == 0:
                    data_raw.pop()
                    self.stations.pop(i)
                    self.log(
                        'Cannot find responses metadata(s) for station {0:s}:{1:s}:{2:s}. Removing station from '
                        'further processing.'.format(net, sta, cha), printcopy=True)
                    break
                response = filt_resp[0][0][0].response
                try:
                    poles = response.response_stages[0].poles
                    zeros = response.response_stages[0].zeros
                    norm_fact = response.response_stages[0].normalization_factor
                    inst_sens = response.instrument_sensitivity.value
                    # stage_gain = response.response_stages[0].stage_gain
                except IndexError:
                    data_raw.pop()
                    self.stations.pop(i)
                    self.log(
                        'Responses metadata(s) for station {0:s}:{1:s}:{2:s} is not complete. Removing station from '
                        'further processing.'.format(net, sta, cha), printcopy=True)
                    break

                tr.stats.paz = AttribDict({
                    'sensitivity': inst_sens,
                    'poles': poles,
                    'gain': norm_fact,
                    'zeros': zeros
                })
            else:
                i += 1  # station not removed
                if not st[0].stats.delta in data_deltas:
                    data_deltas.append(st[0].stats.delta)
        else:
            i += 1

    self.data_raw = data_raw
    self.data_deltas = data_deltas
    self.create_station_index()
    self.check_a_station_present()


def load_q_data(self, archive_dir, t_before=None, t_after=None, xml_file='', noise_amp=None, orig_data=False):
    """
    Loads waveform from local continuous stream for stations listed in ``self.stations``.

    :param self:
    :param archive_dir: parent directory of seiscomp archive waveforms
    :type archive_dir: string
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    :param xml_file: path of stationXML files
    :type xml_file: string
    :param noise_amp: noise amplitude for synthetic test
    :type noise_amp: int
    """
    self.logtext['data'] = s = '\nFetching data from Q archive directory.' \
                               '\n\tarchive dir: {0:s}\n\txml file:  {1:s}'.format(archive_dir, xml_file)
    self.log('\n' + s)

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t  = self.event['t']
    t0 = t - t_before
    t1 = t + t_after
    # if not self.data_are_corrected:
    # self.inv = inv = read_inventory(xml_file)
    i = 0
    data_raw = []
    data_orig = []
    data_deltas = []
    while i < len(self.stations):
        sta = self.stations[i]['code']
        net = self.stations[i]['network']
        cha = self.stations[i]['channelcode']

        # filename = os.path.join(archive_dir, str(t.year), '{:s}.mseed'.format(sta))
        if noise_amp:
            filename = os.path.join(archive_dir, 'noise{:d}%'.format(int(noise_amp)), '{:s}.{:s}.mseed'.format(net, sta))
        else:
            filename = os.path.join(archive_dir, str(t.year), '{:s}.mseed'.format(sta))
        if orig_data:
            orig_file = os.path.join(archive_dir, 'noise0%', '{:s}.{:s}.mseed'.format(net, sta))
            st_orig = read(orig_file, starttime=t0, endtime=t1)
            st_orig = st_orig.select(channel='{:s}*'.format(cha))
            ch_orig = {}
            for comp in range(3):
                ch_orig[st_orig[comp].stats.channel[2]] = st_orig[comp]
            str_comp = self.comps_order
            st_orig = Stream(traces=[ch_orig[str_comp[0]], ch_orig[str_comp[1]], ch_orig[str_comp[2]]])
            data_orig.append(st_orig)
        if not os.path.isfile(filename):
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Cannot find data file(s). Removing station from further processing.'.
                     format(net, sta, cha), printcopy=False)
            self.log('\tExpected file location: ' + filename, printcopy=False)
            continue

        st = read(filename, starttime=t0, endtime=t1)
        st = st.select(channel='{:s}*'.format(cha))
        if len(st) != 3:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further '
                     'processing.'.format(net, sta, cha))
            continue
        ch = {}
        for comp in range(3):
            ch[st[comp].stats.channel[2]] = st[comp]
        if sorted(ch.keys()) != ['E', 'N', 'Z']:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Unoriented components. Removing station from further processing.'.
                     format(net, sta, cha))
            continue
        str_comp = self.comps_order
        st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
        data_raw.append(st)
        self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True
        if not st[0].stats.delta in data_deltas:
            data_deltas.append(st[0].stats.delta)

        if not self.data_are_corrected:
            for tr in data_raw[-1]:
                filt_resp = inv.select(network=tr.stats.network, station=tr.stats.station,
                                       channel=tr.stats.channel, time=self.event['t'])
                if len(filt_resp) == 0:
                    data_raw.pop()
                    self.stations.pop(i)
                    self.log(
                        'Cannot find xml response file(s) for station {0:s}:{1:s}. Removing station from further '
                        'processing.'.format(net, sta), printcopy=True)
                    break
                response = filt_resp[0][0][0].response
                poles = response.response_stages[0].poles
                zeros = response.response_stages[0].zeros
                norm_fact = response.response_stages[0].normalization_factor
                inst_sens = response.instrument_sensitivity.value
                # stage_gain = response.response_stages[0].stage_gain

                tr.stats.paz = AttribDict({
                    'sensitivity': inst_sens,
                    'poles': poles,
                    'gain': norm_fact,
                    'zeros': zeros
                })
            else:
                i += 1  # station not removed
        else:
            i += 1

    self.data_raw = data_raw
    self.data_orig = data_orig
    self.data_deltas = data_deltas
    self.create_station_index()
    if noise_amp:
        self.data_are_corrected = True
    else:
        self.data_are_corrected = False
    self.check_a_station_present()

#
# def load_local_streams_xml(self, st_file, t_before=None, t_after=None, xml_file=''):
#     """
#     Loads waveform from local continuous stream for stations listed in ``self.stations``.
#
#     :param st_file: file of continuous mseed data
#     :param t_before: length of the record before the event origin time
#     :type t_before: float, optional
#     :param t_after: length of the record after the event origin time
#     :type t_after: float, optional
#     :param xml_file: path of stationXML files
#     :type xml_file: string, optional
#     """
#     self.logtext['data'] = s = '\nLoading data from BMKG continuous mseed file.' \
#                                '\n\tdata file: {0:s}\n\txml file:  {1:s}'.format(st_file, xml_file)
#     self.log('\n' + s)
#     t = self.event['t']
#
#     t_before, t_after = self.waveform_time_window(t_before, t_after)
#
#     st_waveforms = read(st_file, starttime=t - t_before, endtime=t + t_after)  # q_edit
#     inv = read_inventory(xml_file)
#     i = 0
#     while i < len(self.stations):
#         sta = self.stations[i]
#         try:
#             st = st_waveforms.select(sta['network'], sta['code'], sta['location'],
#                                      '{:s}*'.format(sta['channelcode']))
#         #st = client.getWaveform(sta['network'], sta['code'], sta['location'], '{:s}*'.format(sta['channelcode']),
#         #                         t - t_before, t + t_after, metadata=True)
#         # st.write('_'.join([sta['network'], sta['code'], sta['location'], sta['channelcode']]), 'MSEED') # DEBUG
#         except:
#             self.log('{0:s}:{1:s}: Record not found. Removing station from further processing.'.format(
#                 sta['network'], sta['code']))
#             self.stations.remove(sta)
#             continue
#         if st.__len__() != 3:
#             self.log('{0:s}:{1:s}: Gap in data / wrong number of components. Removing station from further '
#                      'processing.'.format(sta['network'], sta['code']))
#             self.stations.remove(sta)
#             continue
#         ch = {}
#         for comp in range(3):
#             ch[st[comp].stats.channel[2]] = st[comp]
#         if sorted(ch.keys()) != ['E', 'N', 'Z']:
#             self.log(
#                 '{0:s}:{1:s}: Unoriented components. Removing station from further processing.'.format(
#                     sta['network'],
#                     sta['code']))
#             self.stations.remove(sta)
#             continue
#         str_comp = self.comps_order
#         st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
#         self.data_raw.append(st)
#         sta['useN'] = sta['useE'] = sta['useZ'] = True
#         if not st[0].stats.delta in self.data_deltas:
#             self.data_deltas.append(st[0].stats.delta)
#
#         for tr in self.data_raw[-1]:
#             filt_resp = inv.select(network=tr.stats.network, station=tr.stats.station,
#                                    channel=tr.stats.channel, time=self.event['t'])
#             if len(filt_resp) == 0:
#                 self.stations.pop(i)
#                 self.data_raw.pop()
#                 self.log(
#                     'Cannot find xml response file(s) for station {0:s}:{1:s}. Removing station from further '
#                     'processing.'.format(sta['network'], sta['code']), printcopy=True)
#                 break
#             response = filt_resp[0][0][0].response
#             poles = response.response_stages[0].poles
#             zeros = response.response_stages[0].zeros
#             norm_fact = response.response_stages[0].normalization_factor
#             inst_sens = response.instrument_sensitivity.value
#             stage_gain = response.response_stages[0].stage_gain
#
#             tr.stats.paz = AttribDict({
#                 'sensitivity': inst_sens,
#                 'poles': poles,
#                 'gain': norm_fact,
#                 'zeros': zeros
#             })
#         else:
#             i += 1  # station not removed
#
#     self.create_station_index()
#     self.data_are_corrected = False
#     self.check_a_station_present()
#
# def load_seiscomp3_arc2(self, archive_dir, t_before=None, t_after=None, xml_file=''):
#     """
#     Loads waveform from local continuous stream for stations listed in ``self.stations``.
#
#     :param archive_dir: parent directory of seiscomp archive waveforms
#     :type archive_dir: string, optional
#     :param t_before: length of the record before the event origin time
#     :type t_before: float, optional
#     :param t_after: length of the record after the event origin time
#     :type t_after: float, optional
#     :param xml_file: path of stationXML files
#     :type xml_file: string, optional
#     """
#     self.logtext['data'] = s = '\nFetching data from BMKG seiscomp3 archive directory.' \
#                                '\n\tarchive dir: {0:s}\n\txml file:  {1:s}'.format(archive_dir, xml_file)
#     self.log('\n' + s)
#     loaded = len(self.data)+len(self.data_raw)
#     t  = self.event['t']
#
#     t_before, t_after = self.waveform_time_window(t_before, t_after)
#
#     t0 = t - t_before
#     t1 = t + t_after
#     # st_waveforms = read(archive_dir, starttime=t - t_before, endtime=t + t_after)  # q_edit
#     if not self.data_are_corrected:
#         inv = read_inventory(xml_file)
#     i = 0
#     while i < len(self.stations):
#         if i < loaded:  # the data already loaded from another source, it will be probably used rarely
#             i += 1
#             continue
#         # if i >= self.nr: # some station removed inside the cycle
#             # break
#         sta = self.stations[i]['code']
#         net = self.stations[i]['network']
#         loc = self.stations[i]['location']
#         ch  = self.stations[i]['channelcode']
#         # load data
#         files = []
#         for comp in ['Z', 'N', 'E']:
#             datadir = os.path.join(str(t.year), net, sta, ch + comp + '.D')
#             try:
#                 names = [waveform for waveform in os.listdir(os.path.join(archive_dir, datadir))
#                          if os.path.isfile(os.path.join(archive_dir, datadir, waveform))
#                             and str(t0.year) + '.'  + str(t0.julday) in waveform or
#                          os.path.isfile(os.path.join(archive_dir, datadir, waveform))
#                             and str(t1.year) + '.'  + str(t1.julday) in waveform]
#                 for name in names:
#                     files.append(os.path.join(archive_dir, datadir, name))
#                     # break
#             except WindowsError:
#                 # print('There is no data for station {:s}.{:s}.{:s}{:s}'.format(net, sta, ch, comp))
#                 break
#         if len(files) == 3:
#             self.add_SC3arc(files[0], files[1], files[2], t_before=t_before, t_after=t_after)
#         elif len(files) == 6:
#             self.add_SC3arc2(files, t_before=t_before, t_after=t_after)
#         else:
#             self.stations.pop(i)
#             self.create_station_index()
#             self.log('Cannot find data file(s) for station {0:s}:{1:s}. '
#                      'Removing station from further processing.'.format(net, sta), printcopy=True)
#             self.log('\tExpected file location: ' + os.path.join(archive_dir, datadir), printcopy=True)
#             continue
#
#         for tr in self.data_raw[-1]:
#             filt_resp = inv.select(network=tr.stats.network, station=tr.stats.station,
#                                    channel=tr.stats.channel, time=self.event['t'])
#             if len(filt_resp) == 0:
#                 self.stations.pop(i)
#                 self.data_raw.pop()
#                 self.create_station_index()
#                 self.log(
#                     'Cannot find xml response file(s) for station {0:s}:{1:s}. Removing station from further '
#                     'processing.'.format(net, sta), printcopy=True)
#                 break
#             response = filt_resp[0][0][0].response
#             poles = response.response_stages[0].poles
#             zeros = response.response_stages[0].zeros
#             norm_fact = response.response_stages[0].normalization_factor
#             inst_sens = response.instrument_sensitivity.value
#             # stage_gain = response.response_stages[0].stage_gain
#
#             tr.stats.paz = AttribDict({
#                 'sensitivity': inst_sens,
#                 'poles': poles,
#                 'gain': norm_fact,
#                 'zeros': zeros
#             })
#         else:
#             i += 1  # station not removed
#
#     self.check_a_station_present()
#


def add_SC3arc2(self, filenames=None, t_before=500, t_after=500):
    """
    Reads data from seiscomp archive file. Can read 6 continuous mseed files (3 components, from 2 respective days)
    for overlapping waveform on a station to produce three component stream.
    Append the stream to ``self.data_raw``.
    If its sampling is not contained in ``self.data_deltas``, add it there.
    """
    t = self.event['t']

    if len(filenames) == 6:
        st1 = read(filenames[0]); st2 = read(filenames[1]); st3 = read(filenames[2])
        st4 = read(filenames[3]); st5 = read(filenames[4]); st6 = read(filenames[5])
        # if st1.stats.network == st4.stats.network and st1.stats.station == st4.stats.station and \
        #         st1.stats.channel == st4.stats.channel:
        st1 += st2
        st1.trim(t-t_before, t+t_after)
        # else:
        #     sys.exit('Files sequence is [sta.A-day1, sta.A-day2, sta.B-day1, sta.B-day2, sta.C-day1, sta.C-day2]')
        # if st2.stats.network == st5.stats.network and st2.stats.station == st5.stats.station and \
        #         st2.stats.channel == st5.stats.channel:
        st3 += st4
        st3.trim(t-t_before, t+t_after)
        # else:
        #     sys.exit('Files sequence is [sta.A-day1, sta.A-day2, sta.B-day1, sta.B-day2, sta.C-day1, sta.C-day2]')
        # if st3.stats.network == st6.stats.network and st3.stats.station == st6.stats.station and \
        #         st3.stats.channel == st6.stats.channel:
        st5 += st6
        st5.trim(t-t_before, t+t_after)
        # else:
        #     sys.exit('Files sequence is [sta.A-day1, sta.A-day2, sta.B-day1, sta.B-day2, sta.C-day1, sta.C-day2]')

        st = Stream(traces=[st1[0], st3[0], st5[0]])
    else:
        raise ValueError('Read six files (Z, N, E components from 2 respective days))')

    return st


def add_SC3arc(self, filename, filename2=None, filename3=None, t_before=500, t_after=500):
    """
    Reads data from seiscomp archive file. Can read either one mseed continuous file,
    or three mseed continuous files simultaneously to produce three component stream.
    Append the stream to ``self.data_raw``.
    If its sampling is not contained in ``self.data_deltas``, add it there.
    """
    t = self.event['t']
    if filename3:
        st1 = read(filename, starttime=t - t_before, endtime=t + t_after)
        st2 = read(filename2, starttime=t - t_before, endtime=t + t_after)
        st3 = read(filename3, starttime=t - t_before, endtime=t + t_after)
        st = Stream(traces=[st1[0], st2[0], st3[0]])
    else:
        raise ValueError('Must read three files (Z, N, E components)')
    return st


def load_seiscomp3_archive(self, archive_dir, t_before=None, t_after=None, inventory_dir=''):
    """
    Loads waveform from local continuous stream for stations listed in ``self.stations``.

    :param archive_dir: parent directory of seiscomp archive waveforms
    :type archive_dir: string, optional
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    :param inventory_dir: path of stationXML directory
    :type inventory_dir: string, optional
    """
    self.logtext['data'] = s = f'\nFetching data from SeisComP3 archive directory.\n' \
                               f'\tarchive dir: {archive_dir}\n\tinventory dir: {inventory_dir}'
    self.log('\n' + s)
    t  = self.event['t']

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t0 = t - t_before
    t1 = t + t_after
    # st_waveforms = read(archive_dir, starttime=t - t_before, endtime=t + t_after)  # q_edit
    i = 0
    data_raw = []
    data_deltas = []
    while i < len(self.stations):
        sta = self.stations[i]['code']
        net = self.stations[i]['network']
        loc = self.stations[i]['location']
        ch  = self.stations[i]['channelcode']

        if len(data_raw) > 0:
            if sta == data_raw[-1][0].stats.station:
                self.stations.pop(i)
                continue

        # if not self.data_are_corrected:
        # if net == 'IA':
        inv_file = f'{net}.{sta}.xml'
        # else:
        #     inv_file = f'{net}_BMKG.xml'
        try:
            inv = read_inventory(os.path.join(inventory_dir, inv_file))
        except Exception:
            self.stations.pop(i)
            self.log(
                'Cannot find responses metadata(s) for station {0:s}:{1:s}:{2:s}. Removing station from '
                'further processing.'.format(net, sta, ch), printcopy=True)
            continue

        # load data
        files = []
        datadir = ''
        datadirs = []
        for comp in self.comps_order:
            if t0.year == t1.year:
                datadir = os.path.join(str(t.year), net, sta, ch + comp + '.D')
                try:
                    names = [waveform for waveform in os.listdir(os.path.join(archive_dir, datadir))
                             if os.path.isfile(os.path.join(archive_dir, datadir, waveform))
                             and f'{t0.year}.{t0.julday:03d}' in waveform or
                             os.path.isfile(os.path.join(archive_dir, datadir, waveform))
                             and f'{t1.year}.{t1.julday:03d}' in waveform]
                    for name in names:
                        files.append(os.path.join(archive_dir, datadir, name))
                        # break
                except Exception:
                    # print('There is no data for station {:s}.{:s}.{:s}{:s}'.format(net, sta, ch, comp))
                    break
            else:
                datadirs = [os.path.join(str(t0.year), net, sta, ch + comp + '.D'),
                            os.path.join(str(t1.year), net, sta, ch + comp + '.D')]
                for datadir in datadirs:
                    try:
                        names = [waveform for waveform in os.listdir(os.path.join(archive_dir, datadir))
                                 if os.path.isfile(os.path.join(archive_dir, datadir, waveform))
                                 and f'{t0.year}.{t0.julday:03d}' in waveform or
                                 os.path.isfile(os.path.join(archive_dir, datadir, waveform))
                                 and f'{t1.year}.{t1.julday:03d}' in waveform]
                        for name in names:
                            files.append(os.path.join(archive_dir, datadir, name))
                            # break
                    except Exception:
                        # print('There is no data for station {:s}.{:s}.{:s}{:s}'.format(net, sta, ch, comp))
                        pass
        files = sorted(files, key=lambda x: x.split('/')[-1])
        if len(files) == 3:
            try:
                st = self.add_SC3arc(files[0], files[1], files[2], t_before=t_before, t_after=t_after)
            except Exception:
                self.stations.pop(i)
                self.log('Error read archive data(s) for station {0:s}:{1:s}:{2:s}. '
                         'Removing station from further processing.'.format(net, sta, ch), printcopy=True)
                continue
        elif len(files) == 6:
            try:
                st = self.add_SC3arc2(files, t_before=t_before, t_after=t_after)
            except Exception:
                self.stations.pop(i)
                self.log('Error read archive data(s) for station {0:s}:{1:s}:{2:s}. '
                         'Removing station from further processing.'.format(net, sta, ch), printcopy=True)
                continue
        else:
            self.stations.pop(i)
            self.log('Cannot find data file(s) for station {0:s}:{1:s}:{2:s}. '
                     'Removing station from further processing.'.format(net, sta, ch), printcopy=True)
            if t0.year == t1.year:
                self.log(f'\tExpected file location: {os.path.join(archive_dir, datadir)}', printcopy=True)
            else:
                self.log(f'\tExpected file location: {os.path.join(archive_dir, datadirs[0])}\n'
                         f'\t\t\t\t\t\t\t\t\t\tand\t\t{os.path.join(archive_dir, datadirs[1])}', printcopy=True)
            continue

        if len(st) != 3:  # todo: check for components beside ZNE
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further '
                     'processing.'.format(net, sta, ch))
            continue
        if st[0].stats.location != loc:  # seedlink not filter data location (obspy seedlink/server bugs?)
            self.stations[i]['location'] = st[0].stats.location
        ch = {}
        for comp in range(3):
            ch[st[comp].stats.channel[2]] = st[comp]
        if sorted(ch.keys()) != ['E', 'N', 'Z']:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Unoriented components. '
                     'Removing station from further processing.'.format(net, sta, ch))
            continue
        str_comp = self.comps_order
        st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
        data_raw.append(st)
        self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True

        for tr in data_raw[-1]:
            filt_resp = inv.select(network=tr.stats.network, station=tr.stats.station,
                                   channel=tr.stats.channel, time=self.event['t'])
            if len(filt_resp) == 0:
                self.data_raw.pop()
                self.stations.pop(i)
                self.log(
                    'Cannot find responses metadata(s) for station {0:s}:{1:s}:{2:s}. Removing station from '
                    'further processing.'.format(net, sta, ch), printcopy=True)
                break
            response = filt_resp[0][0][0].response
            try:
                poles = response.response_stages[0].poles
                zeros = response.response_stages[0].zeros
                norm_fact = response.response_stages[0].normalization_factor
                inst_sens = response.instrument_sensitivity.value
                # stage_gain = response.response_stages[0].stage_gain
            except IndexError:
                data_raw.pop()
                self.stations.pop(i)
                self.log(
                    'Responses metadata(s) for station {0:s}:{1:s}:{2:s} is not complete. Removing station from '
                    'further processing.'.format(net, sta, ch), printcopy=True)
                break

            tr.stats.paz = AttribDict({
                'sensitivity': inst_sens,
                'poles': poles,
                'gain': norm_fact,
                'zeros': zeros
            })
        else:
            i += 1  # station not removed
            if not st[0].stats.delta in data_deltas:
                data_deltas.append(st[0].stats.delta)

        # else:
        #     i += 1  # station not removed

    self.data_raw = data_raw
    self.data_deltas = data_deltas
    self.create_station_index()
    self.check_a_station_present()


def load_sac_archive(self, archive_dir, t_before=None, t_after=None, inventory_dir=''):
    """
    Loads waveform from local sac files for stations listed in ``self.stations``.

    :param archive_dir: parent directory of seiscomp archive waveforms
    :type archive_dir: string, optional
    :param t_before: length of the record before the event origin time
    :type t_before: float, optional
    :param t_after: length of the record after the event origin time
    :type t_after: float, optional
    :param inventory_dir: path of stationXML directory
    :type inventory_dir: string, optional
    """
    self.logtext['data'] = s = f'\nFetching data from SAC directory.\n' \
                               f'\tdir: {archive_dir}\n\tinventory dir: {inventory_dir}'
    self.log('\n' + s)
    t  = self.event['t']

    dir_pref = f"{t.strftime('%y%m%d%H%M%S')}"
    datadir = [directory for directory in os.listdir(archive_dir) if directory.startswith(dir_pref)]

    if len(datadir) > 1:
        raise NameError("There is few number of data directory of this event")
    elif len(datadir) < 1:
        raise NameError("There is no event data directory")
    else:
        datadir = datadir[0]

    t_before, t_after = self.waveform_time_window(t_before, t_after)

    t0 = t - t_before
    t1 = t + t_after
    # st_waveforms = read(archive_dir, starttime=t - t_before, endtime=t + t_after)  # q_edit
    i = 0
    data_raw = []
    data_deltas = []
    while i < len(self.stations):
        sta = self.stations[i]['code']
        net = self.stations[i]['network']
        loc = self.stations[i]['location']
        ch  = self.stations[i]['channelcode']

        if len(data_raw) > 0:
            if sta == data_raw[-1][0].stats.station:
                self.stations.pop(i)
                continue

        if ch == 'SS':
            pzfile = "IGU-BD3C-5_PZ_Pole.pz"
        else:
            pzfile = ""

        # load data
        files = []
        datadirs = []
        try:
            names = [waveform for waveform in os.listdir(os.path.join(archive_dir, datadir))
                     if os.path.isfile(os.path.join(archive_dir, datadir, waveform))
                     and sta in waveform and ch in waveform]
            for name in names:
                files.append(os.path.join(archive_dir, datadir, name))
        except Exception:
            # print('There is no data for station {:s}.{:s}.{:s}{:s}'.format(net, sta, ch, comp))
            break

        files = sorted(files, key=lambda x: x.split('/')[-1])
        if len(files) == 3:
            try:
                st = self.add_SC3arc(files[0], files[1], files[2], t_before=t_before, t_after=t_after)
            except Exception:
                self.stations.pop(i)
                self.log('Error read archive data(s) for station {0:s}:{1:s}:{2:s}. '
                         'Removing station from further processing.'.format(net, sta, ch), printcopy=True)
                continue
        else:
            self.stations.pop(i)
            self.log('Cannot find data file(s) for station {0:s}:{1:s}:{2:s}. '
                     'Removing station from further processing.'.format(net, sta, ch), printcopy=True)
            if t0.year == t1.year:
                self.log(f'\tExpected file location: {os.path.join(archive_dir, datadir)}', printcopy=True)
            else:
                self.log(f'\tExpected file location: {os.path.join(archive_dir, datadirs[0])}\n'
                         f'\t\t\t\t\t\t\t\t\t\tand\t\t{os.path.join(archive_dir, datadirs[1])}', printcopy=True)
            continue

        if len(st) != 3:  # todo: check for components beside ZNE
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further '
                     'processing.'.format(net, sta, ch))
            continue
        if st[0].stats.location != loc:  # seedlink not filter data location (obspy seedlink/server bugs?)
            self.stations[i]['location'] = st[0].stats.location
        ch = {}
        for comp in range(3):
            ch[st[comp].stats.channel[2]] = st[comp]
        if sorted(ch.keys()) != ['E', 'N', 'Z']:
            self.stations.pop(i)
            self.log('{0:s}:{1:s}:{2:s}: Unoriented components. '
                     'Removing station from further processing.'.format(net, sta, ch))
            continue
        str_comp = self.comps_order
        st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
        data_raw.append(st)
        self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True

        for tr in data_raw[-1]:
            if pzfile:
                attach_PZ(tr, os.path.join(inventory_dir, pzfile))
            else:  # poles&zeros file not found
                self.data_raw.pop()
                self.stations.pop(i)
                self.log(
                    'Cannot find poles and zeros file(s) for station {0:s}:{1:s}:{2:s}. Removing station from '
                    'further processing.'.format(net, sta, ch), printcopy=True)
                break

        else:
            i += 1  # station not removed
            if not st[0].stats.delta in data_deltas:
                data_deltas.append(st[0].stats.delta)

    self.data_raw = data_raw
    self.data_deltas = data_deltas
    self.create_station_index()
    self.check_a_station_present()


def attach_PZ(tr, pz_file):
    """
    Attaches to a trace a PZ AttribDict containing poles zeros and gain from isola format or sac format.

    :param tr: Trace
    :type tr: :class:`~obspy.core.trace.Trace`
    :param pz_file: path to pzfile in ISOLA or SAC format
    :type pz_file: string
    """
    f = open(pz_file, 'r')
    pz = f.readlines()
    f.close()
    if 'A0' in pz[0] or 'a0' in pz[0]:  # ISOLA format
        f = open(pz_file, 'r')
        f.readline()  # comment line: A0
        A0 = float(f.readline())
        f.readline()  # comment line: count-->m/sec
        count2ms = float(f.readline())
        f.readline()  # comment line: zeros
        n_zeros = int(f.readline())
        zeros = []
        for i in range(n_zeros):
            line = f.readline()
            search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
            (r, i) = search.groups()
            zeros.append(complex(float(r), float(i)))
        f.readline() # comment line: poles
        n_poles = int(f.readline())
        poles = []
        for i in range(n_poles):
            line = f.readline()
            search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
            try:
                (r, i) = search.groups()
            except:
                print(line)
            poles.append(complex(float(r), float(i)))
        tr.stats.paz = AttribDict({
            'sensitivity': A0,
            'poles': poles,
            'gain': 1./count2ms,
            'zeros': zeros
            })
        f.close()
    elif 'ZEROS' in pz[0] or 'zeros' in pz[0]:  # SAC format
        f = open(pz_file, 'r')
        n_zeros = int(f.readline().split()[1])
        zeros = []
        for i in range(n_zeros):
            line = f.readline()
            search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
            (r, i) = search.groups()
            zeros.append(complex(float(r), float(i)))
        n_poles = int(f.readline().split()[1])
        poles = []
        for i in range(n_poles):
            line = f.readline()
            search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
            try:
                (r, i) = search.groups()
            except:
                print(line)
            poles.append(complex(float(r), float(i)))
        CONSTANT = float(f.readline().split()[1])
        tr.stats.paz = AttribDict({
            'sensitivity': CONSTANT,  # set sensitivity as CONSTANT (A0/C), asummed norm_fact (C) = 1 (1/gain)
            'poles': poles,
            'gain': 1.,
            'zeros': zeros
        })
        f.close()
    else:
        raise ValueError


def update_arrtime(self):
    engine = LocalEngine(store_superdirs=[self.GF_store_dir])
    for sts in self.stations:
        store = engine.get_store(sts['model'])
        dist, az, baz = gps2dist_azimuth(float(self.centroid['lat']), float(self.centroid['lon']),
                                         float(sts['lat']), float(sts['lon']))
        sts['arr_time'] = store.t('begin', (self.centroid['z'], dist)) + self.centroid['shift']
