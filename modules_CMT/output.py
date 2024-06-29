#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import hashlib
import subprocess
import numpy as np
import facebook as fb
import xml.etree.ElementTree as ET
from datetime import datetime as dt
from datetime import timezone
from ftplib import FTP
from os.path import exists, join
from obspy.core import UTCDateTime
from obspy.geodetics import kilometers2degrees
from modules_CMT.MT_comps import a2mt
from modules_MGMT.sc_config import masterpro_vars, clientpro_vars


def hash_pid(var):
    hashmd5 = hashlib.md5(f"{var}{str(dt.now(timezone.utc).strftime('%d-%b-%Y %H:%M:%S'))}".encode('utf-8')).hexdigest()
    return f"{hashmd5[:8]}-{hashmd5[8:12]}-{hashmd5[12:16]}-{hashmd5[16:20]}-{hashmd5[20:]}"


def make_event_scxml(evt_pid, host, user, passw, db,  parent_dir):

    os.system(f'scxmldump -d mysql://{passw}:{user}@{host}/{db} -E {evt_pid} -PAMfF > event.xml')

    # sh_file = join(parent_dir, 'rmt2seiscomp.sh')
    # # sh_file = '/home/sysop/rmt2seiscomp.sh'
    # with open(sh_file, 'w') as f:
    #     # SC3
    #     # f.write(f'#!/bin/sh\nscxmldump -d mysql://sysop:sysop@{host}/seiscomp3 -E {evt_pid} -PAMfF > event.xml')
    #     # SC4
    #     # f.write(f'#!/bin/sh\nscxmldump -d mysql://sysop:sysop@{host}/seiscomp -E {evt_pid} -PAMfF > event.xml')
    #     f.write(f'#!/bin/sh\nscxmldump -d mysql://{passw}:{user}@{host}/{db} -E {evt_pid} -PAMfF > event.xml')
    # os.chmod(sh_file, 0b111101101)
    # subprocess.Popen([sh_file], stdin=subprocess.PIPE)


def send_to_web(server, port, user, password, solution_dir):

    ftp = FTP(server, port)
    ftp.login(user=user, passwd=password)

    ftp.cwd('/extra/')

    parent_dir = os.getcwd()
    os.chdir(solution_dir)
    desc = ''
    try:
        if exists('solution_map.png'):
            ftp.storbinary("STOR solution_map.png", open("solution_map.png", 'rb'))
            desc = '\n   Success send summary to Stageof Ambon Website...'
        # if exists('MT_solution.png'):
        #     ftp.storbinary("STOR MT_solution.png", open("MT_solution.png", 'rb'))
        # if exists('solution_quality.png'):
        #     ftp.storbinary("STOR solution_quality.png", open("solution_quality.png", 'rb'))
        # if exists('final_sta_azimuth.png'):
        #     ftp.storbinary("STOR final_sta_azimuth.png", open("final_sta_azimuth.png", 'rb'))
        # if exists('fit_true_waveform.png'):
        #     ftp.storbinary("STOR fit_true_waveform.png", open("fit_true_waveform.png", 'rb'))
        # if exists('fit_cova_waveform.png'):
        #     ftp.storbinary("STOR fit_cova_waveform.png", open("fit_cova_waveform.png", 'rb'))
        # if exists('log.txt'):
        #     ftp.storbinary("STOR log.txt", open("log.txt", 'rb'))
    except (Exception, EOFError) as e:
        desc = f"\n   Error send summary to Stageof Ambon Website: {e}"
    ftp.quit()
    os.chdir(parent_dir)
    return desc


def send_to_facebook(token, event_id, solution_dir, update=False):

    post_FB = fb.GraphAPI(token)

    parent_dir = os.getcwd()
    os.chdir(solution_dir)
    desc = ''
    try:
        if exists('solution_map.png'):
            if update:
                post_FB.put_photo(open("solution_map.png", 'rb'),
                                  message=f"AutoRMT Solution Event '{event_id}' (update)")
            else:
                post_FB.put_photo(open("solution_map.png", 'rb'),
                                  message=f"AutoRMT Solution Event '{event_id}'")
            desc = '\n   Success send summary to Facebook Page Auto_RMT...'
        # if exists('MT_solution.png'):
        #     post_FB.put_photo(open("MT_solution.png", 'rb'), message="Moment Tensor Solution (deviatoric)")
        # if exists('solution_quality.png'):
        #     post_FB.put_photo(open("solution_quality.png", 'rb'), message="Solution quality")
        # if exists('final_sta_azimuth.png'):
        #     post_FB.put_photo(open("final_sta_azimuth.png", 'rb'), message="Station used in final inversion")
        # if exists('fit_true_waveform.png'):
        #     post_FB.put_photo(open("fit_true_waveform.png", 'rb'), message="Fitting of True Waveform")
        # if exists('fit_cova_waveform.png'):
        #     post_FB.put_photo(open("fit_cova_waveform.png", 'rb'),
        #                       message="Fitting of Standardized Data (corrected using matrix covariance data)")
    except Exception as e:
        desc = f"\n   Error send summary to Facebook Page Auto_RMT: {e}"
    os.chdir(parent_dir)
    return desc


def send_to_seiscomp(_vars, misfit_from_q, eval_mode='automatic'):
    # todo: from obspy import read_events
    #  cat = read_events('event_sc4.xml')
    #  cat[0]['origins'] + 1
    #  modif content
    #  rewrite to event.xml
    #  dispatch

    evt_pid = _vars.event['id']
    host, user, passw, db = masterpro_vars()
    # make_event_scxml(evt_pid, host, parent_dir)  ##

    C = _vars.centroid
    fp = C['faultplanes']
    sts_dist = [sts_dict['dist']/1e3 for sts_dict in _vars.stations]
    mt = np.array(a2mt(C['a'], 'USE')) * 10 ** 7

    FM_PID = hash_pid('focmec')
    CEN_PID = hash_pid('centroid')

    # p = C['p']
    # t = C['t']
    # n = C['n']

    # evt_xml = os.path.join('Integrasi Seiscomp', 'tes.xml')
    # evt_xml = os.path.join(parent_dir, f'{evt_pid}.xml')  ##
    evt_xml = os.path.join(_vars.parentdir, 'event.xml')  ##
    xmltree = ET.parse(evt_xml)
    xmlroot = xmltree.getroot()
    Trig_Ori_id = ''
    # SC3
    # for elem in xmlroot.findall('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}EventParameters'):
    #     event = elem.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}event')
    #     Trig_Ori_id = event.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.10}preferredOriginID').text
    # SC4
    for elem in xmlroot.findall('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.11}EventParameters'):
        event = elem.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.11}event')
        Trig_Ori_id = event.find('{http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.11}preferredOriginID').text

    mt_xml_fmt = open(os.path.join(_vars.parentdir, 'inc', 'fmt_focmec.xml'), 'r')
    mt_xml = mt_xml_fmt.read()
    mt_xml_fmt.close()
    mt_xml = mt_xml.replace('$Centroid_PID', CEN_PID)
    mt_xml = mt_xml.replace('$C_time', str(_vars.event['t'] + C['shift']))
    mt_xml = mt_xml.replace('$C_lat', str(C['lat']))
    mt_xml = mt_xml.replace('$C_lon', str(C['lon']))
    mt_xml = mt_xml.replace('$C_dep', str(C['z'] / 1e3))
    mt_xml = mt_xml.replace('$fix_time', 'False')
    mt_xml = mt_xml.replace('$fix_epic', str(_vars.centroid_inv))
    mt_xml = mt_xml.replace('$n_sta', str(_vars.num_sta))
    mt_xml = mt_xml.replace('$Gap', str(_vars.azimuth_gap[0]))
    mt_xml = mt_xml.replace('$Max_dist', str(kilometers2degrees(sts_dist[-1])))
    mt_xml = mt_xml.replace('$Min_dist', str(kilometers2degrees(sts_dist[0])))
    mt_xml = mt_xml.replace('$Med_dist', str(kilometers2degrees(np.median(sts_dist))))
    mt_xml = mt_xml.replace('$timestamp', str(UTCDateTime(dt.now(timezone.utc))))
    mt_xml = mt_xml.replace('$Mw_PID', hash_pid('momenmag'))
    mt_xml = mt_xml.replace('$Mw', str(C['Mw']))

    mt_xml = mt_xml.replace('$FM_PID', FM_PID)
    mt_xml = mt_xml.replace('$TrigOriginPID', Trig_Ori_id)
    mt_xml = mt_xml.replace('$S1', str(fp[0][0]))
    mt_xml = mt_xml.replace('$D1', str(fp[0][1]))
    mt_xml = mt_xml.replace('$R1', str(fp[0][2]))
    mt_xml = mt_xml.replace('$S2', str(fp[1][0]))
    mt_xml = mt_xml.replace('$D2', str(fp[1][1]))
    mt_xml = mt_xml.replace('$R2', str(fp[1][2]))
    # mt_xml = mt_xml.replace('$TA', )
    # mt_xml = mt_xml.replace('$TP', )
    # mt_xml = mt_xml.replace('$TV', )
    # mt_xml = mt_xml.replace('$PA', )
    # mt_xml = mt_xml.replace('$PP', )
    # mt_xml = mt_xml.replace('$PV', )
    # mt_xml = mt_xml.replace('$NA', )
    # mt_xml = mt_xml.replace('$NP', )
    # mt_xml = mt_xml.replace('$NV', )
    mt_xml = mt_xml.replace('$misfit', str(misfit_from_q))
    mt_xml = mt_xml.replace('$EvaluationMode', eval_mode)
    mt_xml = mt_xml.replace('$MT_PID', hash_pid('momentensor'))
    mt_xml = mt_xml.replace('$moment', str(C['mom']))
    mt_xml = mt_xml.replace('$Mrr', str(mt[0]))
    mt_xml = mt_xml.replace('$Mtt', str(mt[1]))
    mt_xml = mt_xml.replace('$Mpp', str(mt[2]))
    mt_xml = mt_xml.replace('$Mrt', str(mt[3]))
    mt_xml = mt_xml.replace('$Mrp', str(mt[4]))
    mt_xml = mt_xml.replace('$Mtp', str(mt[5]))
    mt_xml = mt_xml.replace('$variance', f'{C["VR"]:.6f}')
    mt_xml = mt_xml.replace('$VR',  f'{C["VR"]*100:.6f}')
    mt_xml = mt_xml.replace('$DC', str(C['dc_perc']))
    mt_xml = mt_xml.replace('$CLVD', str(C['clvd_perc']))
    mt_xml = mt_xml.replace('$ISO', str(C['iso_perc']))
    mt_xml = mt_xml.replace('$n_comp', str(_vars.components))
    mt_xml = mt_xml.replace('$shortest_T', str(1/_vars.fmax))

    foc_mec_line = f'      <preferredFocalMechanismID>smi:local/{FM_PID}</preferredFocalMechanismID>'
    ref_line = f'      <originReference>smi:local/{CEN_PID}</originReference>\n' \
               f'      <focalMechanismReference>smi:local/{FM_PID}</focalMechanismReference>'

    evt_xml_ = open(f'{evt_xml}', 'r')
    evt_xml_data = evt_xml_.read()
    evt_xml_.close()

    evt_xml_data = evt_xml_data.replace(f'    <event publicID="{evt_pid}">',
                                        f'{mt_xml}\n    <event publicID="{evt_pid}">')
    if 'preferredMagnitudeID' not in evt_xml_data:
        evt_xml_data = evt_xml_data.replace('</preferredOriginID>', f'</preferredMagnitudeID>\n{foc_mec_line}')
    else:
        evt_xml_data = evt_xml_data.replace('</preferredMagnitudeID>', f'</preferredMagnitudeID>\n{foc_mec_line}')
    evt_xml_data = evt_xml_data.replace('    </event>', f'{ref_line}\n    </event>')

    evt_xml_ = open(f'{evt_xml}', 'w')
    evt_xml_.write(evt_xml_data)
    evt_xml_.close()

    #inputs=f'-i{join(_vars.parentdir, "event.xml")}'
    #opr = '-Omerge'
    #dbs = f'-dmysql://{passw}:{user}@{host}/{db}'
    #subprocess.run(['/home/sysop/seiscomp/bin/scdispatch', inputs, opr, dbs])

    os.system(f'scdispatch -i {join(_vars.parentdir, "event.xml")} -O merge -d mysql://{passw}:{user}@{host}/{db}')

    #stream = os.popen(f'scdispatch -i {join(_vars.parentdir, "event.xml")} -O merge -d mysql://{passw}:{user}@{host}/{db}')

    # sh_file = join(_vars.parentdir, 'rmt2seiscomp.sh')
    # # print(sh_file)
    # # client_outdir = '/home/sysop/Desktop/QRMT_output/'
    # with open(sh_file, 'w') as f:
    #     # f.write(f'#!/bin/sh\nscdispatch -i {join(parent_dir, evt_pid)}.xml -O merge -d mysql://sysop:sysop@{host}/seiscomp3')
    #     # SC3
    #     # f.write(f'#!/bin/sh\nscdispatch -i {join(_vars.parentdir, "event")}.xml -O merge -d mysql://sysop:sysop@{host}/seiscomp3')
    #     # SC4
    #     f.write(f"#!/bin/sh\n")
    #     f.write(f"scdispatch -i {join(_vars.parentdir, 'event.xml')} -O merge -d mysql://{passw}:{user}@{host}/{db}\n")
    #     # f.write(f"ssh {user}@172.60.132.61 'mkdir -p {client_outdir} && rm -rf {join(client_outdir, '*')}'\n")
    # os.chmod(sh_file, 0b111101101)
    # subprocess.call([sh_file], stdin=subprocess.PIPE, shell=True)


def send_to_clientpro(parentdir, output_dir, client_outdir):

    host, user, passw, db = clientpro_vars()
    sh_file = join(parentdir, 'rmt2clientpro.sh')
    with open(sh_file, 'w') as f:
        f.write(f"#!/bin/sh\n")
        f.write(f"ssh {user}@{host} 'mkdir -p {client_outdir} && rm -rf {join(client_outdir, '*')}'\n")
        f.write(f"scp -r {join(output_dir, '*')} {user}@{host}:{client_outdir}\n")
    os.chmod(sh_file, 0b111101101)
    subprocess.call([sh_file], stdin=subprocess.PIPE, shell=True)


def write_rmt_config(event, output_dir, configs):

    if len(event.split('.')) > 1:
        evt = event.split('.')[1]
    else:
        evt = event

    config_object = [
        {
            'CONFIG_EVENT': evt,
            'CONFIG_MANUAL_INVERSION': False,
            'CONFIG_MANUAL_STATION': False,
            'INVERSION_SETTINGS': configs["INVERSION_SETTINGS"],
            'STATIONS_SELECTION': configs["STATIONS_SELECTION"]
        }
    ]

    with open(os.path.join(output_dir, f'{evt}.ini'), 'w') as yamlfile:
        yaml.dump(config_object, yamlfile)
