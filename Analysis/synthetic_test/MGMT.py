import os
import sys
import time
import shutil
import logging
# import json
from glob import glob
from os.path import exists
from copy import deepcopy
from datetime import timezone
from datetime import datetime as dt
from pytimedinput import timedKey
from synt_CMT.output import write_rmt_config, send_to_clientpro
# from synt_CMT.output import send_to_web, send_to_facebook
# from modules_MGMT.confidential_materials import facebook_token, pgr9_ftp_server


def watchingAPI():
    print("\n\n\nBMKG Regional Moment Tensor:\n\nWatching WRS NewGen API changes\n")


def analyze_lastEQ(current_resp, timeout=5):
    q, timeout = timedKey('\nDo you wanna analyze last earthquake on WRS NewGen API? (y/[n])', allowCharacters="ynYN",
                          timeout=timeout)
    if timeout:
        time.sleep(0.1)
        print("\nTime out... Not analyze last earthquake on WRS NewGen")
        time.sleep(0.5)
        watchingAPI()
    else:
        if q == "y" or q == 'Y':
            print("\n\n\nBMKG Regional Moment Tensor:\n\nAnalyze last earthquake on WRS NewGen\n")
            current_resp = ""
        else:
            watchingAPI()
    return current_resp


def analyze_smallEQ(magnitudo, status, min_mag=3., timeout=5):
    if float(magnitudo) < min_mag or status != 'M':
        q, timeout = timedKey('\nLast event magnitudo is < 3 or analysis status is automatic.'
                              '\nWanna continue the analysis? (y/[n])', allowCharacters="ynYN", timeout=timeout)
        if timeout:
            time.sleep(0.1)
            print("\nTime out... Not analyze the earthquake")
            time.sleep(0.5)
            sys.exit()
        else:
            if q == "y" or q == 'Y':
                pass
            else:
                sys.exit()


class MGMT:
    """
    Class for manage BMKG Regional Moment Tensor directories and files.
    """
    def __init__(self, input_dir, qrmt_dir, greens_dir, station_file):

        # self.server, self.port, self.user, self.password = pgr9_ftp_server()
        # self.fb_token = facebook_token()

        self.projectdir = projectdir = os.getcwd()
        self.inp_dir = input_dir
        self.grn_dir = greens_dir
        self.qrmt_dir = qrmt_dir
        self.stn_file = station_file
        # self.vel_file = velmod_file
        self.ren_event = ''

        self.tmp_dir = os.path.join(qrmt_dir, 'tmp')
        self.cnf_dir = os.path.join(projectdir, 'config')
        self.fns_dir = os.path.join(projectdir, 'output', 'finished')
        self.unf_dir = os.path.join(projectdir, 'output', 'unfinished')

        self.logger = logging

        if not exists('output'):
            os.mkdir('output')
        if not exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        if not exists(self.cnf_dir):
            os.mkdir(self.cnf_dir)
        if not exists(os.path.join(self.cnf_dir, 'default.ini')):
            shutil.copyfile(os.path.join(self.qrmt_dir, 'config', 'default.ini'),
                            os.path.join(self.cnf_dir, 'default.ini'))
        if not exists(self.fns_dir):
            os.mkdir(self.fns_dir)
        if not exists(self.unf_dir):
            os.mkdir(self.unf_dir)
        if not exists(self.inp_dir):
            os.mkdir(self.inp_dir)
            sys.exit('Input files not exist')

        self.sl_cache = os.path.join(self.tmp_dir, 'seedlink_cache.tmp')
        self.cat_fnsh = os.path.join(projectdir, 'output', 'katalog-finished.txt')
        self.cat_ufns = os.path.join(projectdir, 'output', 'katalog-unfinished.txt')
        self.cat_sln1 = os.path.join(projectdir, 'output', 'katalog-init_solution.txt')
        self.cat_slni = os.path.join(projectdir, 'output', 'katalog-iter_solution.txt')
        self.cat_slnf = os.path.join(projectdir, 'output', 'katalog-final_solution.txt')

        self.logger.basicConfig(filename=os.path.join(projectdir, 'output', 'sys_log.txt'), level=logging.ERROR,
                                format='\n%(asctime)s [%(levelname)s] %(message)s', datefmt='%d-%b-%Y %H:%M:%S')
        self.logger.Formatter.converter = time.gmtime

        with open(os.path.join(self.tmp_dir, 'pid.pid'), 'w') as f:
            f.write(f'{os.getpid()}\n')

    def get_dirs(self):
        return self.fns_dir, self.unf_dir

    def get_files(self, iteration=False):
        if iteration:
            return self.cat_sln1, self.cat_slnf, self.cat_slni
        else:
            return self.cat_sln1, self.cat_slnf

    def read_catalog(self, cat_file):
        self.cat_file = cat_file
        cat_data = open(cat_file, 'r')
        self.catalogs = catalogs = cat_data.readlines()
        cat_data.close()

        if not os.path.exists(cat_file):
            sys.exit('   Catalog file not exist.\n   exiting...')
        if len(catalogs) == 0:
            sys.exit('   Catalog is empty.\n   exiting...')
        return deepcopy(catalogs)

    def rewrite_catalog(self, current_catalog):
        self.catalogs.remove(f'{current_catalog}\n')
        with open(self.cat_file, 'w') as f:  # rewrite catalog exclude finished event
            for line in self.catalogs:
                f.write(line)

    def set_sens_file(self):
        self.sens_file = os.path.join(self.output_dir, 'freq_sens_results.txt')
        if exists(self.sens_file):
            os.remove(self.sens_file)
        return self.sens_file

    def set_output_dir(self, event):
        if not glob(os.path.join(self.projectdir, 'output', f'*{event.split(".")[1]}')):
            self.ren_event = event
            self.output_dir = os.path.join(self.projectdir, 'output', event)
        else:
            for i in range(99):
                ren_event = f'{event}_{i+1:02}'
                if not glob(os.path.join(self.projectdir, 'output', f'*{ren_event.split(".")[1]}')):
                    self.ren_event = ren_event
                    self.output_dir = os.path.join(self.projectdir, 'output', ren_event)
                    break
        return self.output_dir

    def finished(self, event, catalog_event, quality, send_online=False):
        descs = ''
        # write_rmt_config(event, self.output_dir, config_dict)
        self.rewrite_catalog(current_catalog=catalog_event)
        if exists(self.cat_fnsh):  # write finished event to finished catalog
            with open(self.cat_fnsh, 'a') as f:
                f.write(f'{catalog_event}\n')
        else:
            with open(self.cat_fnsh, 'w') as f:
                f.write(f'{catalog_event}\n')

        # fixed_outdir = '/home/sysop/Desktop/QRMT_output/'
        # send_to_clientpro(self.parentdir, self.output_dir, fixed_outdir)

        # if glob(os.path.join(self.fns_dir, f'*{event.split(".")[1]}*')):
        #     flag_update = True
        # else:
        #     flag_update = False

        # if send_online and quality == 'A' or send_online and quality == 'B':
        #     descs += send_to_web(self.server, self.port, self.user, self.password, output_dir)
        #     descs += send_to_facebook(self.fb_token, event.split('.')[1], output_dir, flag_update)
        #     descs += '\n'
        # self.del_snr_mouse()
        renamed_dir = os.path.join(self.fns_dir, self.ren_event)
        if not glob(os.path.join(self.fns_dir, f'*{self.ren_event.split(".")[1]}')):
            try:
                shutil.move(self.output_dir, self.fns_dir)
                time.sleep(0.1)
                descs += '\n   Moving finished event "{:s}" files to:' \
                         '\n     "{:s}"'.format(self.ren_event, renamed_dir)
            except (IOError, OSError, shutil.Error):
                descs += '\n   Not fully moving finished event "{:s}" files'.format(self.ren_event)
        else:
            for i in range(99):
                ren_event = f'{event}_{i + 1:02}'
                if not glob(os.path.join(self.fns_dir, f'*{ren_event.split(".")[1]}')):
                    renamed_dir = os.path.join(self.fns_dir, ren_event)
                    try:
                        os.rename(self.output_dir, renamed_dir)
                        time.sleep(0.1)
                        descs += '\n   Moving finished event "{:s}" files to:' \
                                 '\n     "{:s}"'.format(self.ren_event, renamed_dir)
                    except (IOError, OSError, shutil.Error):
                        descs += '\n   Not fully moving finished event "{:s}" files'.format(self.ren_event)
                    break

        descs += '\n   ______________________________________________________\n ' \
                 '\n   Event "{:s}" finished {:s}\n'. \
            format(event, dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))

        return descs

    def unfinished(self, event, catalog_event, config_dict=None):
        descs = ''
        self.rewrite_catalog(current_catalog=catalog_event)
        if config_dict:
            write_rmt_config(event, self.output_dir, config_dict)
        if exists(self.cat_ufns):  # write unfinished event to unfinished catalog
            with open(self.cat_ufns, 'a') as fl:
                fl.write(f'{catalog_event}\n')
        else:
            with open(self.cat_ufns, 'w') as fl:
                fl.write(f'{catalog_event}\n')

        if not glob(os.path.join(self.unf_dir, f'*{self.ren_event.split(".")[1]}')):
            renamed_dir = os.path.join(self.unf_dir, self.ren_event)
            try:
                shutil.move(self.output_dir, self.unf_dir)
                time.sleep(0.1)
                descs += '\n   Moving unfinished event "{:s}" files to:' \
                         '\n     "{:s}"'.format(self.ren_event, renamed_dir)
            except (IOError, OSError, shutil.Error):
                descs += '\n   Not fully moving unfinished event "{:s}" files'.format(self.ren_event)
        else:
            for i in range(99):
                ren_event = f'{event}_{i + 1:02}'
                if not glob(os.path.join(self.unf_dir, f'*{ren_event.split(".")[1]}')):
                    renamed_dir = os.path.join(self.unf_dir, ren_event)
                    try:
                        os.rename(self.output_dir, renamed_dir)
                        time.sleep(0.1)
                        descs += '\n   Moving unfinished event "{:s}" files to:' \
                                 '\n     "{:s}"'.format(self.ren_event, renamed_dir)
                    except (IOError, OSError, shutil.Error):
                        descs += '\n   Not fully moving unfinished event "{:s}" files'.format(self.ren_event)
                    break

        descs += '\n   ______________________________________________________\n ' \
                 '\n   Event "{:s}" unfinished {:s}\n'. \
            format(event, dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))

        return descs

    def del_snr_mouse(self):
        mouse_dir = os.path.join(self.output_dir, 'mouse')
        snr_dir = os.path.join(self.output_dir, 'snr')
        if exists(mouse_dir):
            shutil.rmtree(mouse_dir)
        if exists(snr_dir):
            shutil.rmtree(snr_dir)
        time.sleep(2)

