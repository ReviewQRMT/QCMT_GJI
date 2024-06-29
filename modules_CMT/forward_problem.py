#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from obspy import Stream
from pyrocko.gf import LocalEngine, MTSource
from pyrocko.obspy_compat.base import to_obspy_trace
from modules_CMT.geometry import haversine


def precalc_greens(event, grid, targets, stations, comps_order='ZNE', store_dir=''):
    """
    :param event: earthquake coordinate location to get precalculated green's function
    :param grid: grid shift x,y,z from event as precalculated green's function source
    :param stations: list of station parameter
    :param targets: list of station targets to get precalculated green's function
    :param comps_order: component order 'ZNE'
    :param store_dir: directory of pyrocko precalculated greens function store
    :return: 6 elementary seismogram for each receiver components
    """

    engine = LocalEngine(store_superdirs=[store_dir])

    depth = grid['z']
    n_shft = grid['x']
    e_shft = grid['y']

    mt_source1 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
                          depth=depth, east_shift=e_shft, north_shift=n_shft,
                          mnn=0, mee=0, mdd=0, mne=1, mnd=0, med=0)
    mt_source2 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
                          depth=depth, east_shift=e_shft, north_shift=n_shft,
                          mnn=0, mee=0, mdd=0, mne=0, mnd=1, med=0)
    mt_source3 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
                          depth=depth, east_shift=e_shft, north_shift=n_shft,
                          mnn=0, mee=0, mdd=0, mne=0, mnd=0, med=-1)
    mt_source4 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
                          depth=depth, east_shift=e_shft, north_shift=n_shft,
                          mnn=-1, mee=0, mdd=1, mne=0, mnd=0, med=0)
    mt_source5 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
                          depth=depth, east_shift=e_shft, north_shift=n_shft,
                          mnn=0, mee=-1, mdd=1, mne=0, mnd=0, med=0)
    mt_source6 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
                          depth=depth, east_shift=e_shft, north_shift=n_shft,
                          mnn=1, mee=1, mdd=1, mne=0, mnd=0, med=0)

    mt_sources = [mt_source1, mt_source2, mt_source3, mt_source4, mt_source5, mt_source6]

    mt_synthetic_traces = [None] * len(mt_sources)
    for i, elem_source in zip(range(len(mt_sources)), mt_sources):
        mt_response = engine.process(elem_source, targets)
        mt_synthetic_traces[i] = mt_response.pyrocko_traces()

    # sts_num = 0  # q_test
    # src_num = 0
    # store = engine.get_store(store_id)
    # dist = targets[sts_num].distance_to(mt_sources[src_num])
    # # targets[sts_num].lat
    # depth = mt_sources[src_num].depth
    # arrival_time = store.t('begin', (depth, dist))
    # print('{:s} dist = {:.4f} km, arr = {:.2f}'.format(targets[sts_num].codes[1], dist / 1000, arrival_time))

    elemse_all = []
    for sts in stations:
        stn = sts['code']
        net = sts['network']
        elemse = []
        for elem_source in mt_synthetic_traces:
            select_trace = [trc for trc in elem_source if trc.station == stn and trc.network == net]
            if len(select_trace) == 3:
                streams = Stream()
                for comp in comps_order:
                    streams += to_obspy_trace([trc for trc in select_trace if trc.channel == comp][0])
                elemse.append(streams)
        elemse_all.append(elemse)

    return elemse_all


def select_greens(greensfile, lat, lon, meanmodel=None):
    """
    Select representative green's function (gf) based on earthquake location
    :param lat: latitude of earthquake epicenter
    :param lon: longitude of earthquake epicenter
    :param greensfile: file of list gf store informations
    :param meanmodel: mean gf store id
    :return: list of nearest gf store id
    """

    df = pd.read_csv(greensfile, sep='\t')
    df['dist'] = df.apply(lambda x: haversine(lon, lat, x['lon'], x['lat']), axis=1)
    df.sort_values(by=['dist'], inplace=True)
    df.drop_duplicates(subset='id', inplace=True)
    df = df.head(3)
    df.reset_index(inplace=True)
    if meanmodel:
        gfstores = [meanmodel]
    else:
        gfstores = []
    for index, row in df.iterrows():
        gfstores.append(row['id'])
    return gfstores
