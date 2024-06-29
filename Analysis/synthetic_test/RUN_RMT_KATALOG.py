#! /usr/bin/env python
# coding=utf-8

from RMT import *
from MGMT import *
# from synt_CMT.forward_problem import select_greens
from obspy.core import UTCDateTime

"""
===========================================
Regional Moment Tensor Inversion by @eqhalauwet
==========================================
"""

QCMTdir   = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
inp_dir   = os.path.join(QCMTdir, 'input')
grn_dir   = os.path.join(QCMTdir, 'gf_stores')
datadir   = os.path.join(os.getcwd(), 'data')
stn_file  = os.path.join(datadir, 'stations-synt.dat')
cat_file  = os.path.join(datadir, 'katalog.txt')

modules_MGMT = MGMT(inp_dir, QCMTdir, grn_dir, stn_file)
log_sys = modules_MGMT.logger
catalog_solution_initial, catalog_solution_final = modules_MGMT.get_files()
ev_catalogs = modules_MGMT.read_catalog(cat_file)

for cat in ev_catalogs:
    # evt_all += 1
    cat = cat.strip()
    ot = UTCDateTime(cat.split(';')[0])
    lat, lon, dep, mag = map(float, cat.split(';')[1:5])
    ID = cat.split(';')[5]
    remark = ''
    # cat = f"{ot};{lat:.2f};{lon:.2f};{dep:.1f};{mag:.1f};{ID}"
    evt = f"{ot.strftime('%Y%m%dT%H%M%S')}.{ID}"
    out_dir = modules_MGMT.set_output_dir(evt)

    # initial velocity model for model sensitivity check
    # Greens_func_store = ['thesis_MEAN', 'thesis_OC', 'thesis_IA', 'thesis_CO']  # test2
    Greens_func_store = ['SE']
    # Greens_func_store = ['LE']

    try:

        Q_MT = CMT(event_id=evt, parentdir=QCMTdir, configdir=modules_MGMT.cnf_dir, outdir=out_dir,
                   GF_stores=Greens_func_store, GF_store_dir=grn_dir)

        if Q_MT.save_sens_file:
            sens_file = modules_MGMT.set_sens_file()
        else:
            sens_file = ''

        if not Q_MT.load_data(lat, lon, dep, mag, ot, remark, stn_file, datadir=datadir,
                              inv_file=os.path.join(datadir, 'thesis_response.xml'), noise_amp=int(ID)):
            desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
            Q_MT.log(desc, printcopy=True)
            sys.exit()

        if Q_MT.selected_station:

            Q_MT.set_manual_station()
            Q_MT.correct_data_xml()

            if not Q_MT.count_stations():
                desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
                Q_MT.log(desc, printcopy=True)
                sys.exit()

            # Q_MT.max_gap, start_gap, end_gap = Q_MT.obs_azimuth()

            if not Q_MT.set_parameters(log=True):
                desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
                Q_MT.log(desc, printcopy=True)
                sys.exit()

        else:

            # if not Q_MT.data_QC(plot=False):
            #     desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
            #     Q_MT.log(desc, printcopy=True)
            #     sys.exit()
            Q_MT.count_stations()
            Q_MT.set_initial_frequencies(fmin=Q_MT.minfreq, fmax=Q_MT.maxfreq)

            Q_MT.select_best_station(sector_num=8, numsta_sect=4,
                                     outfile=os.path.join(out_dir, 'initial_sta_azimuth.png'))

            if not Q_MT.initial_inversion_settings():
                desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
                Q_MT.log(desc, printcopy=True)
                sys.exit()

            feedback, quality = Q_MT.run_initial_inversion(sens_file, plot=True)
            if feedback is False:
                desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
                Q_MT.log(desc, printcopy=True)
                sys.exit()
            else:
                solution = Q_MT.write_solution(ID, cat, feedback, quality, catalog_solution_initial)

            if not Q_MT.optimum_inversion_settings(sensitivity_file=sens_file):
                desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
                Q_MT.log(desc, printcopy=True)
                sys.exit()

        feedback, quality = Q_MT.run_final_inversion(inp_dir, sens_file, plot=True, save_config=True)
        if feedback is False:
            desc = modules_MGMT.unfinished(evt, cat, Q_MT.config_dict())
            Q_MT.log(desc, printcopy=True)
            sys.exit()
        else:
            solution = Q_MT.write_solution(ID, cat, feedback, quality, catalog_solution_final)

        desc = modules_MGMT.finished(evt, cat, quality)
        Q_MT.log(desc, printcopy=True)

    except Exception as e:
        modules_MGMT.unfinished(evt, cat)
        if exists(os.path.join(QCMTdir, 'tmp', 'pid.pid')):
            os.remove(os.path.join(QCMTdir, 'tmp', 'pid.pid'))
        print(f"{dt.now(timezone.utc).strftime('%d-%b-%Y %H:%M:%S')} [ERROR] event: {evt}\n{e}\n")
        log_sys.exception(f'Event: {evt}')
