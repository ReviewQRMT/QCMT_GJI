#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import sys
import re
# import json
# import requests
import pandas as pd
import xmltodict
import mysql.connector as mysql
# import xml.etree.ElementTree as ET
# from modules_CMT.extras import isnumber
from obspy import UTCDateTime
from datetime import datetime as dt
# from datetime import timedelta as td
from requests.exceptions import ConnectionError


# todo: set all parameters datatype from here, add **args status analysis


def read_index3_event(index3_url):
    """
    Get hyposenter parameter of last earthquake from PGN index3
    :param index3_url: index3 address
    :return: seiscomp event_id, origin, mag, lat, lon, dep
    """
    df = pd.read_csv(index3_url, sep='|', skiprows=[0, 1, 3], skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.head(1)
    seiscomp_id = 'index3'
    origin = df['Origin Time (GMT)'].values[0].strip()
    lat = df['Lat'].values[0]
    if 'N' in lat:
        lat = float(lat.split()[0])
    else:
        lat = -float(lat.split()[0])
    lon = df['Lon'].values[0]
    if 'E' in lon:
        lon = float(lon.split()[0])
    else:
        lon = -float(lon.split()[0])
    mag = float(df['Mag'].values)
    dep = float(df['Depth'].values[0].split()[0])

    if df['Status'].values[0].strip() == 'manual':
        status = 'M'
    else:
        status = 'A'

    return seiscomp_id, origin, mag, lat, lon, dep, status


def read_inatews_API(request_session, url_api):
    """
    Get hyposenter parameter of last earthquake from InaTEWS lastQL.json
    (https://bmkg-content-inatews.storage.googleapis.com/)
    :param request_session: request session
    :param url_api: InaTEWS API url
    :return: seiscomp event_id, origin, mag, lat, lon, dep
    """
    t = int(dt.now().timestamp() * 1e3)
    url_api = f'{url_api}?t={t}'  # add "? ..." to prevent cache
    response = request_session.get(url_api)
    # response = requests.get(url_api)
    data_json = response.json()
    # data_json = json.loads(urlopen(url).read())
    Par = data_json['features'][0]['properties']
    Loc = data_json['features'][0]['geometry']
    seiscomp_id = Par['id']
    origin = Par['time']
    status = Par['status']
    remark = Par['place']
    lat = round(float(Loc['coordinates'][1]), 3)
    lon = round(float(Loc['coordinates'][0]), 3)
    mag = round(float(Par['mag']), 1)
    dep = round(float(Par['depth']))

    return seiscomp_id, UTCDateTime(origin), mag, lat, lon, dep, status, remark


def read_inatews_APIresp(response):
    """
    Get hyposenter parameter of last earthquake from InaTEWS lastQL.json response
    (https://bmkg-content-inatews.storage.googleapis.com/)
    :param response: InaTEWS API url json response
    :return: seiscomp event_id, origin, mag, lat, lon, dep
    """
    Par = response['features'][0]['properties']
    Loc = response['features'][0]['geometry']
    seiscomp_id = Par['id']
    origin = Par['time']
    status = Par['status']
    remark = Par['place']
    lat = float(Loc['coordinates'][1])
    lon = float(Loc['coordinates'][0])
    mag = float(Par['mag'])
    dep = float(Par['depth'])

    return seiscomp_id, UTCDateTime(origin), mag, lat, lon, dep, status, remark


def read_seiscomp_local_event(datapath, filename='mailexportfile.txt'):
    file = os.path.join(datapath, filename)

    flag_evt = False
    flag_ori = False
    flag_mag = False
    # flag_pha = False
    # flag_mlst = False

    bmkg_dic = {}

    i = 0
    ev = 0
    with open(file, "r", -1) as f:

        for l in f:

            i += 1
            if 'Event:' in l and len(l.split()) < 4:
                flag_evt = True
                ev += 1
                continue

            if 'Origin:' in l and len(l.split()) < 4:
                flag_ori = True
                flag_evt = False
                continue

            if 'Network magnitudes:' in l and len(l.split()) < 4:
                flag_mag = True
                continue

            # if 'Phase arrivals:' in l and len(l.split()) < 4:
            #     sta = []
            #     net = []
            #     dis = []
            #     azi = []
            #     pha = []
            #     dtm = []
            #     res = []
            #     wth = []
            #     pha_dic = {}
            #     flag_pha = True
            #     continue
            #
            # if 'Station magnitudes:' in l and len(l.split()) < 4:
            #     sta = []
            #     net = []
            #     dis = []
            #     azi = []
            #     typ = []
            #     val = []
            #     res = []
            #     amp = []
            #     mag_dic = {}
            #     flag_mlst = True
            #     continue

            if flag_evt:

                if 'Public ID' in l:
                    pid = l.split()[2]
                    continue
                if 'region name:' in l:
                    remark = re.sub(r"^\s+", "", l.split(':')[1].replace('\n', ''))
                    continue

            if flag_ori:

                if flag_ori and not l.strip():
                    flag_ori = False
                    continue

                if 'Date' in l:
                    tahun, bulan, tanggal = map(int, l.split()[1].split('-')[0:3])
                    continue

                if 'Time' in l:
                    jam, menit = map(int, l.split()[1].split(':')[0:2])
                    detik = float(l.split()[1].split(':')[2])
                    sec = int(detik)
                    msec = round((detik - sec) * 1e6)

                    if '+/-' in l:
                        err_tim = float(l.split()[3])

                    try:
                        ot = dt(tahun, bulan, tanggal, jam, menit, sec, msec)
                    except ValueError:
                        print(f'Error encountered on event {ev} data-line {i}\n')
                    continue

                if 'Latitude' in l:
                    lintang = float(l.split()[1])
                    if '+/-' in l:
                        err_lat = float(l.split()[4])
                    continue

                if 'Longitude' in l:
                    bujur = float(l.split()[1])
                    if '+/-' in l:
                        err_lon = float(l.split()[4])
                    continue

                if 'Depth' in l:
                    depth = float(l.split()[1])
                    if '+/-' in l:
                        err_dep = l.split()[4]
                    else:
                        err_dep = '-0.0'
                    continue

                if 'manual' in l:
                    mod = l.split()[1]
                    status = 'M'
                    continue
                elif 'automatic' in l:
                    mod = l.split()[1]
                    status = 'A'
                    continue

                if 'RMS' in l:
                    rms = float(l.split()[2])
                    continue

                if 'gap' in l:
                    gap = int(l.split()[2])
                    continue

            if flag_mag:

                if 'preferred' in l:
                    mag = float(l.split()[1])
                    mag_type = l.split()[0]
                    if '+/-' in l:
                        err_mag = l.split()[3]
                    else:
                        err_mag = '-0.0'

                    bmkg_dic[ot] = {'pid': pid,
                                    'rem': remark,
                                    'lat': lintang,
                                    'lon': bujur,
                                    'dep': depth,
                                    'mag': mag,
                                    'typ': mag_type,
                                    'gap': gap,
                                    'rms': rms,
                                    'mod': mod,
                                    'err': {'e_tim': err_tim,
                                            'e_lat': err_lat,
                                            'e_lon': err_lon,
                                            'e_dep': err_dep,
                                            'e_mag': err_mag}}

                    flag_mag = False
                    break
                    # continue
            #
            # if flag_pha:
            #
            #     if flag_pha and not l.strip():
            #         pha_dic[ot] = {'arr': {'sta': sta,
            #                                'net': net,
            #                                'dis': dis,
            #                                'azi': azi,
            #                                'pha': pha,
            #                                'del': dtm,
            #                                'res': res,
            #                                'wth': wth}}
            #
            #         bmkg_dic[ot].update(pha_dic[ot])
            #
            #         flag_pha = False
            #         continue
            #
            #     if '##' not in l and isnumber(l.split()[6]) and 'X' not in l.split()[7]:
            #
            #         jam2, menit = map(int, l.split()[5].split(':')[0:2])
            #         detik = float((l.split())[5].split(':')[2])
            #         sec = int(detik)
            #         msec = round((detik - sec) * 1e6)
            #
            #         try:
            #             at = dt(tahun, bulan, tanggal, jam2, menit, sec, msec)
            #             if jam2 < jam:
            #                 at = at + td(days=1)
            #             deltatime = float('%.2f' % (at.timestamp() - ot.timestamp()))
            #         except ValueError:
            #             print(f'Error encountered on event {ev} data-line {i}\n')
            #
            #         dist = float(l.split()[2])
            #
            #         sta.append(l.split()[0])
            #         net.append(l.split()[1])
            #         dis.append(dist)
            #         azi.append(int(l.split()[3]))
            #         pha.append(l.split()[4])
            #         dtm.append(deltatime)
            #         res.append(float(l.split()[6]))
            #         wth.append(l.split()[8])
            #
            # if flag_mlst:
            #
            #     if flag_mlst and not l.strip():
            #         mag_dic[ot] = {'mls': {'sta': sta,
            #                                'net': net,
            #                                'dis': dis,
            #                                'azi': azi,
            #                                'typ': typ,
            #                                'val': val,
            #                                'res': res,
            #                                'amp': amp}}
            #
            #         bmkg_dic[ot].update(mag_dic[ot])
            #
            #         flag_mlst = False
            #         continue
            #
            #     if isnumber(l.split()[2]) and isnumber(l.split()[3]) and isnumber(l.split()[5]) and \
            #             isnumber(l.split()[6]) and isnumber(l.split()[7]):
            #
            #         dist = float(l.split()[2])
            #
            #         sta.append(l.split()[0])
            #         net.append(l.split()[1])
            #         dis.append(dist)
            #         azi.append(int(l.split()[3]))
            #         typ.append(l.split()[4])
            #         val.append(float(l.split()[5]))
            #         res.append(float(l.split()[6]))
            #         amp.append(l.split()[7])

    # return bmkg_dic
    return pid, UTCDateTime(ot), mag, lintang, bujur, depth, status, remark


def get_seiscomp_event(host, user="sysop", password="sysop", db="seiscomp3"):
    """
    Get hyposenter parameter of last earthquake from Seiscomp3
    :param host: SeiscomP3 address
    :param user: database user
    :param password: database password
    :param db: database name
    :return: seiscomp event_id, origin, mag, lat, lon, dep
    """
    cnx = mysql.connect(host=host, user=user, password=password, database=db)

    if not cnx:
        sys.exit(ConnectionError)

    mycursor = cnx.cursor()

    query = f"SELECT PEvent.publicID, Origin.time_value " \
            f"FROM Event, PublicObject as PEvent, Origin, PublicObject as POrigin " \
            f"WHERE Event._oid=PEvent._oid and Origin._oid=POrigin._oid and Event.preferredOriginID=POrigin.publicID " \
            f"ORDER BY Origin.time_value DESC LIMIT 1"

    mycursor.execute(query)
    myresult = mycursor.fetchall()
    if not myresult:
        query = "SELECT PEvent.publicID, Origin.time_value " \
                "FROM Event, PublicObject as PEvent, Origin, PublicObject as POrigin " \
                "WHERE Event._oid=PEvent._oid and Origin._oid=POrigin._oid and " \
                "Event.preferredOriginID=POrigin.publicID " \
                "ORDER BY Origin.time_value DESC LIMIT 1"
        mycursor.execute(query)
        myresult = mycursor.fetchall()
    last_event_ID = myresult[0][0]

    query = f"SELECT PEvent.publicID, Origin.time_value, ROUND(Magnitude.magnitude_value,1), " \
            f"ROUND(Origin.latitude_value,4), ROUND(Origin.longitude_value,4), ROUND(Origin.depth_value), " \
            f"Origin.evaluationMode, EventDescription.text " \
            f"FROM Event, PublicObject as PEvent, Origin, PublicObject as POrigin, Magnitude, " \
            f"PublicObject as PMagnitude, EventDescription, PublicObject as PEventDescription " \
            f"WHERE PEvent.publicID='{last_event_ID}' AND Event._oid=PEvent._oid AND Origin._oid=POrigin._oid AND " \
            f"Magnitude._oid=PMagnitude._oid AND Event.preferredOriginID=POrigin.publicID AND " \
            f"Event.preferredMagnitudeID=PMagnitude.publicID AND EventDescription._parent_oid=PEvent._oid " \
            f"ORDER BY Origin.time_value desc limit 1"

    mycursor.execute(query)
    r = mycursor.fetchall()[0]

    if r[6] == 'manual':
        status = 'M'
    else:
        status = 'A'

    return f'L{r[0]}', UTCDateTime(r[1]), r[2], r[3], r[4], r[5], status, r[7]


def read_scxml(parent_dir, xml_file='event.xml'):
    evt_xml = os.path.join(parent_dir, xml_file)
    with open(evt_xml, 'r', encoding='utf-8') as file:
        evt_xml = file.read()
    xml_dict = xmltodict.parse(evt_xml)

    ot = mag = lat = lon = dep = status = None

    event = xml_dict['seiscomp']['EventParameters']['event']
    origins = xml_dict['seiscomp']['EventParameters']['origin']
    pid = event['@publicID']
    rem = event['description']['text']
    pref_Org = event['preferredOriginID']
    pref_Mag = event['preferredMagnitudeID']
    if isinstance(origins, dict):
        ot = UTCDateTime(origins['time']['value'])
        lat = float(origins['latitude']['value'])
        lon = float(origins['longitude']['value'])
        dep = float(origins['depth']['value'])
        if origins['evaluationMode'] == 'manual' or origins['evaluationMode'] == 'Manual':
            status = 'M'
        else:
            status = 'A'
        if isinstance(origins['magnitude'], dict):
            mag = origins['magnitude']['value']
        elif isinstance(origins['magnitude'], list):
            for M in origins['magnitude']:
                if M['@publicID'] == pref_Mag:
                    mag = float(M['magnitude']['value'])
                    break
    elif isinstance(origins, list):
        for O in origins:
            if O['@publicID'] == pref_Org:
                ot = UTCDateTime(O['time']['value'])
                lat = float(O['latitude']['value'])
                lon = float(O['longitude']['value'])
                dep = float(O['depth']['value'])
                if O['evaluationMode'] == 'manual' or O['evaluationMode'] == 'Manual':
                    status = 'M'
                else:
                    status = 'A'
                if isinstance(O['magnitude'], dict):
                    mag = O['magnitude']['value']
                elif isinstance(O['magnitude'], list):
                    for M in O['magnitude']:
                        if M['@publicID'] == pref_Mag:
                            mag = float(M['magnitude']['value'])
                            break
                break

    return pid, ot, mag, lat, lon, dep, status, rem
    #
    # xmltree = ET.parse(evt_xml)
    # xmlroot = xmltree.getroot()
    # pref_Org = ''
    # pref_Mag = ''
    # for elem in xmlroot.findall('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}EventParameters'):
    #     event = elem.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}event')
    #     pid = event.get('publicID')
    #     pref_Org = event.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}preferredOriginID').text
    #     pref_Mag = event.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}preferredMagnitudeID').text
    # for elem in xmlroot.findall('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}EventParameters'):
    #     for evt in elem.findall('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}origin'):
    #         if evt.attrib['publicID'] == pref_Org:
    #             print('yes')
    #             ot = evt.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}time')[0].text
    #                 # .text
    #             for m in evt.findall('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}magnitude'):
    #                 if m.attrib['publicID'] == pref_Mag:
    #                     mag = m.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}magnitude')[0].text
    #
    # return pid, UTCDateTime(ot), mag, lintang, bujur, depth, status, remark