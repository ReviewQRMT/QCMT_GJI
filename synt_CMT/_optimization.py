#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from copy import deepcopy
from pyrocko.gf import Target
from matplotlib.font_manager import FontProperties
from synt_CMT.extras import q_filter
from synt_CMT.geometry import azimuth_gap
from synt_CMT.inverse_problem import calc_taper_window, precalc_greens


def find_optimum_params(self, sensitivity_file=''):
    """
    Calculates the variance reduction from each component and the variance reduction from a subset of stations
    of difference model and freq range.
    find optimum model and freq for each station, use all comp with good VR
    """
    npts = self.npts_slice
    # data = self.data_shifts[self.centroid['shift_idx']]
    data_raw = self.raw_data_shifts[self.centroid['shift_idx']]

    for model in self.GF_stores:
        targets = []
        for sts in self.stations:
            targets += [Target(quantity=self.seismogram, lat=float(sts['lat']), lon=float(sts['lon']),
                               store_id=model, tmin=self.t_min, tmax=(self.npts_elemse - 1) / self.samprate,
                               codes=(sts['network'], sts['code'], '', channel_code))
                        for channel_code in self.comps_order]

        elemse_raw = precalc_greens(self.event, self.centroid, targets, self.stations, self.comps_order,
                                    self.GF_store_dir)

        for j in range(len(self.freq_shifts)):
            fshift = self.freq_shifts[j]
            elemse = deepcopy(elemse_raw)
            data = deepcopy(data_raw)
            for r in range(self.nr):
                stn_fshift = [sts for sts in fshift if sts['code'] == self.stations[r]['code'] and
                              sts['network'] == self.stations[r]['network']]
                self.stations[r]['fmin'] = fmin = stn_fshift[0]['fmin']
                self.stations[r]['fmax'] = fmax = stn_fshift[0]['fmax']
                self.stations[r]['model'] = model
                tarr2 = UTCDateTime(0) + self.t_min + self.stations[r]['arr_time']
                tlen = calc_taper_window(self.stations[r]['dist'])
                for e in range(6):
                    q_filter(elemse[r][e], fmin, fmax, tarr2, tlen, self.taper_perc)
                    elemse[r][e].trim(UTCDateTime(0) + self.elemse_start_origin)

                for st in data[r]:
                    tarr = self.event['t'] + self.stations[r]['arr_time'] + self.centroid['shift']
                    q_filter(st, fmin, fmax, tarr, tlen, self.taper_perc)
            # COMPS_USED = 0
            # for sta in range(self.nr):
                MISFIT = 0
                NORM_D = 0
                # VR_avg = 0
                SYNT = {}
                for comp in range(3):
                    SYNT[comp] = np.zeros(npts)
                    for e in range(6):
                        SYNT[comp] += elemse[r][e][comp].data[0:npts] * self.centroid['a'][e, 0]
                comps_used = 0
                for comp in range(3):
                    synt = SYNT[comp]
                    d = data[r][comp][0:npts]

                    comps_used += 1
                    misfit = np.sum(np.square(d - synt))
                    norm_d = np.sum(np.square(d))
                    # t = np.arange(0, (npts-0.5) / self.samprate, 1. / self.samprate) # DEBUG
                    # fig = plt.figure() # DEBUG
                    # plt.plot(t, synt, label='synt') # DEBUG
                    # plt.plot(t, d, label='data') # DEBUG
                    # plt.plot(t, d-synt, label='difference') # DEBUG
                    # plt.legend() # DEBUG
                    # plt.show() # DEBUG
                    VR = 1 - misfit / norm_d
                    self.stations[r][{0: 'VR_Z', 1: 'VR_N', 2: 'VR_E'}[comp]] = VR
                    MISFIT += misfit
                    NORM_D += norm_d
                    VR_sum = 1 - MISFIT / NORM_D
                    # VR_avg += VR
                    # COMPS_USED += 1
                    # print r, comp, VR, VR_sum # DEBUG
                self.stations[r]['VR'] = VR_sum
                # self.stations[r]['VR'] = VR_avg/3

            if not self.sens_param:
                for sts in self.stations:
                    self.sens_param.append({'code': sts['code'], 'net': sts['network'], 'model': sts['model'],
                                            'fmin': sts['fmin'], 'fmax': sts['fmax'], 'fid': j + 1, 'VR': sts['VR'],
                                            'VR_Z': sts['VR_Z'], 'VR_N': sts['VR_N'], 'VR_E': sts['VR_E']})
            else:
                for sts, sts_p in zip(self.stations, self.sens_param):
                    if sts['VR'] > sts_p['VR']:
                        if sts['code'] == sts_p['code']:
                            sts_p['model'] = sts['model']
                            sts_p['fmin'] = sts['fmin']
                            sts_p['fmax'] = sts['fmax']
                            sts_p['fid'] = j + 1
                            sts_p['VR'] = sts['VR']
                            sts_p['VR_Z'] = sts['VR_Z']
                            sts_p['VR_N'] = sts['VR_N']
                            sts_p['VR_E'] = sts['VR_E']
                        else:
                            sys.exit('Error sensitivity parameter order')
            if sensitivity_file:
                self.print_components(sensitivity_file)


def set_sens_parameters(self, freq_shifting):
    """
    Sets frequency range for each station from shifting freq called from function set_working_frequencies.
    """
    for stn in self.stations:
        stn_fshift = [sts for sts in freq_shifting if sts['code'] == stn['code'] and
                      sts['network'] == stn['network']]
        stn['fmin'] = stn_fshift[0]['fmin']
        stn['fmax'] = stn_fshift[0]['fmax']
        stn['model'] = self.GF_store
        self.fmax = max(self.fmax, stn['fmax'])


def calc_threshold(self, max_threshold=0.3):
    # find trade-off minimum VR, stations comps, overal VR and number of sts.
    list_threshold = np.arange(max_threshold, 0.01, -0.05)
    for threshold in list_threshold:
        count_sts = 0
        count_comp = 0
        for sts, sts_p in zip(self.stations, self.sens_param):
            if sts['code'] == sts_p['code']:
                if sts_p['VR'] >= 0.5:
                    # sts['useN'] = sts['useE'] = sts['useZ'] = True
                    count_comp += 3
                    count_sts += 1
                    continue
                for comp in self.comps_order:
                    if sts_p['VR_{:s}'.format(comp)] >= threshold:
                        count_comp += 1
                        # sts['use{:s}'.format(comp)] = True
                    # else:
                    #     sts['use{:s}'.format(comp)] = False
                if sts['useZ'] or sts['useN'] or sts['useE']:
                    count_sts += 1
            else:
                sys.exit('Error sensitivity parameter order')
        # if count_sts >= self.min_sta and count_comp >= self.min_comp:
        if count_comp >= self.min_comp:
            self.log("\n   VR threshold set to " + str(threshold), printcopy=True)
            return threshold
    return max_threshold


# def set_optimum_params(self, flex_threshold=False):
#     """
#     Sets frequency range for each station from best shifting freq get from function find_optimum_params.
#     """
#     self.fmax = 0.
#     if flex_threshold:
#         threshold = self.calc_threshold(self.vr_th)
#     else:
#         threshold = self.vr_th
#     for sts, sts_p in zip(self.stations, self.sens_param):
#         if sts['code'] == sts_p['code']:
#             sts['model'] = sts_p['model']
#             sts['fmin'] = sts_p['fmin']
#             sts['fmax'] = sts_p['fmax']
#             sts['fid'] = sts_p['fid']
#             if sts_p['VR'] >= 0.5:
#                 sts['useN'] = sts['useE'] = sts['useZ'] = True
#                 self.fmax = max(self.fmax, sts['fmax'])
#                 continue
#             for comp in self.comps_order:
#                 if sts_p['VR'] >= 0.4:
#                     if sts_p['VR_{:s}'.format(comp)] >= 0:
#                         sts['use{:s}'.format(comp)] = True
#                     elif sts_p['VR_{:s}'.format(comp)] < -0.005:
#                         sts['use{:s}'.format(comp)] = False
#                     # else:
#                     #     sts['use{:s}'.format(comp)] = False
#                 # elif sts_p['VR'] >= 0.3:
#                 #     if sts_p['VR_{:s}'.format(comp)] >= 0.05:
#                 #         sts['use{:s}'.format(comp)] = True
#                     # elif sts_p['VR_{:s}'.format(comp)] < 0:
#                     #     sts['use{:s}'.format(comp)] = False
#                     # else:
#                     #     sts['use{:s}'.format(comp)] = False
#                 else:
#                     if sts_p['VR_{:s}'.format(comp)] >= threshold:
#                         sts['use{:s}'.format(comp)] = True
#                     elif sts_p['VR_{:s}'.format(comp)] < -0.005:
#                         sts['use{:s}'.format(comp)] = False
#                     # else:
#                     #     sts['use{:s}'.format(comp)] = False
#             if sts['useZ'] or sts['useN'] or sts['useE']:
#                 self.fmax = max(self.fmax, sts['fmax'])
#         else:
#             sys.exit('Error sensitivity parameter order')


def set_optimum_params(self, outfile=''):
    """
    Sets frequency range for each station from best shifting freq get from function find_optimum_params.
    """
    self.fmax = 0.
    for sts, sts_p in zip(self.stations, self.sens_param):
        # set optimum model and freq filter
        if sts['code'] == sts_p['code']:
            sts['model'] = sts_p['model']
            sts['fmin'] = sts_p['fmin']
            sts['fmax'] = sts_p['fmax']
            sts['fid'] = sts_p['fid']
            sts['VR'] = sts_p['VR']
            sts['VR_Z'] = sts_p['VR_Z']
            sts['VR_N'] = sts_p['VR_N']
            sts['VR_E'] = sts_p['VR_E']
            sts['res_Z'] = sts_p['res_Z']
            sts['res_N'] = sts_p['res_N']
            sts['res_E'] = sts_p['res_E']
        sts['useN'] = sts['useE'] = sts['useZ'] = False

    for i in range(self.sector_number):
        sect_sta = [sts for sts in self.stations if sts['sect'] == i + 1]
        if not sect_sta:
            continue
        # Select best VR each sector, set use all comp True
        sect_sta = sorted(sect_sta, key=lambda k: k['VR'], reverse=True)
        # for j in range(len(sect_sta)):
        for j, select_sta in zip(range(len(sect_sta)), sect_sta):
            if j < self.numsta_sect:
                select_sta['useN'] = select_sta['useE'] = select_sta['useZ'] = True
                self.fmax = max(self.fmax, select_sta['fmax'])

        # unselect_sta = deepcopy(sect_sta)
        #
        # for j, select_sta in zip(range(len(sect_sta)), sect_sta):
        #     if j < self.numsta_sect:
        #         unselect_sta.remove(select_sta)
        #
        # if unselect_sta:
        #     for stn in unselect_sta:
        #         stn['useN'] = stn['useE'] = stn['useZ'] = False

    used_sta = [sts for sts in self.stations if sts['useZ'] is True]
    used_sta = sorted(used_sta, key=lambda k: k['VR'], reverse=True)

    if self.vr_th and len(used_sta) > self.min_sta:
        flex_sta = used_sta[self.min_sta:]
        for sts in flex_sta:
            for comp in self.comps_order:
                if sts['VR_{:s}'.format(comp)] < self.vr_th:  # -0.005:
                    sts['use{:s}'.format(comp)] = False

    sts = self.stations
    # for azimuth plot
    list_azdist = []
    list_azimuth = []
    select_azdist = []
    # dataset for pygmt plot
    use_sta = []
    unuse_sta = []
    part_use_sta = []
    for stn in sts:  # get list of az and dist of used station
        if stn['useZ'] and stn['useN'] and stn['useE']:
            list_azimuth.append(stn['az'])
            select_azdist.append((stn['az'], stn['dist'] / 1000, stn['weight'], stn['code']))
            use_sta.append(stn)
        elif stn['useE'] or stn['useN'] or stn['useZ']:
            list_azimuth.append(stn['az'])
            select_azdist.append((stn['az'], stn['dist'] / 1000, stn['weight'], stn['code']))
            part_use_sta.append(stn)
        else:
            unuse_sta.append(stn)
        list_azdist.append((stn['az'], stn['dist'] / 1000, stn['code']))

    # gap, start, end = self.azimuth_gap
    self.azimuth_gap = gap, start, end = azimuth_gap(list_azimuth)
    self.used_a = pd.DataFrame(use_sta, columns=['network', 'code', 'channelcode', 'lon', 'lat'])
    self.used_n = pd.DataFrame(unuse_sta, columns=['network', 'code', 'channelcode', 'lon', 'lat'])
    self.used_p = pd.DataFrame(part_use_sta, columns=['network', 'code', 'channelcode', 'lon', 'lat'])

    if outfile:
        plt.rcParams.update({'font.size': 10})
        fontP = FontProperties()
        fontP.set_size('xx-small')
        # print(fontP.get_size())
        # fontP.set_size(4.8)
        ax = plt.subplot(111, projection='polar')
        if start > end:  # plot gap shading
            ax.fill_between(np.linspace(np.deg2rad(start), np.deg2rad(360), 100), 0, sts[-1]['dist'] / 1000,
                            alpha=0.2, color='b', label='Max Azimuth Gap {:d}$^\circ$'.format(int(gap)))
            ax.fill_between(np.linspace(np.deg2rad(0), np.deg2rad(end), 100), 0, sts[-1]['dist'] / 1000,
                            alpha=0.2, color='b')
        else:
            ax.fill_between(np.linspace(np.deg2rad(start), np.deg2rad(end), 100), 0, sts[-1]['dist'] / 1000,
                            alpha=0.2, color='b', label='Max Azimuth Gap {:d}$^\circ$'.format(int(gap)))

        for az in self.lst_qdr_edge:
            ax.plot([np.deg2rad(az), np.deg2rad(az)], [0, sts[-1]['dist'] / 1000], lw=1, c='k')

        for (az, dst, wgt, stn), j in zip(select_azdist, range(len(select_azdist))):
            if j == 0:
                ax.scatter(np.deg2rad(az), dst, c='red', marker='v', s=60, edgecolors='k',
                           label='Used Station')
            else:
                ax.scatter(np.deg2rad(az), dst, c='red', marker='v', s=60, edgecolors='k')
            plt.text(np.deg2rad(az), dst, stn, horizontalalignment='center', verticalalignment='top',
                     fontsize=9, weight='bold')

        j = 0
        for az, dst, stn in list_azdist:
            if not any(stn in i for i in select_azdist):
                if j == 0:
                    ax.scatter(np.deg2rad(az), dst, c='gray', marker='v', s=25, edgecolors='k',
                               label='Not Used Station')
                else:
                    ax.scatter(np.deg2rad(az), dst, c='gray', marker='v', s=25, edgecolors='k')
                plt.text(np.deg2rad(az), dst, stn, horizontalalignment='center', verticalalignment='top',
                         fontsize=9)
                j += 1

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # clockwise
        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.tick_params(axis='y', colors='blue')
        plt.savefig(outfile, bbox_inches='tight', dpi=320)
        plt.clf()
        plt.close()

    # self.create_station_index()
    # else:
    #     sts['use{:s}'.format(comp)] = False
    #     best_VR = 0
    #     for sts, sts_p in zip(self.stations, self.sens_param):
    #         if sts['sect'] == i+1:
    #             if sts_p['VR'] > best_VR:
    #                 best_VR = sts_p['VR']
    #             if sts['code'] == sts_p['code']:
    #                 if sts_p['VR'] >= 0.5:
    #                     sts['useN'] = sts['useE'] = sts['useZ'] = True
    #                     self.fmax = max(self.fmax, sts['fmax'])
    #                     continue
    #                 else:
    #
    # for sts, sts_p in zip(self.stations, self.sens_param):
    #     if sts['code'] == sts_p['code']:
    #         sts['model'] = sts_p['model']
    #         sts['fmin'] = sts_p['fmin']
    #         sts['fmax'] = sts_p['fmax']
    #         sts['fid'] = sts_p['fid']
    #         if sts_p['VR'] >= 0.5:
    #             sts['useN'] = sts['useE'] = sts['useZ'] = True
    #             self.fmax = max(self.fmax, sts['fmax'])
    #             continue
    #         for comp in self.comps_order:
    #             if sts_p['VR'] >= 0.4:
    #                 if sts_p['VR_{:s}'.format(comp)] >= 0:
    #                     sts['use{:s}'.format(comp)] = True
    #                 elif sts_p['VR_{:s}'.format(comp)] < -0.005:
    #                     sts['use{:s}'.format(comp)] = False
    #                 # else:
    #                 #     sts['use{:s}'.format(comp)] = False
    #             # elif sts_p['VR'] >= 0.3:
    #             #     if sts_p['VR_{:s}'.format(comp)] >= 0.05:
    #             #         sts['use{:s}'.format(comp)] = True
    #                 # elif sts_p['VR_{:s}'.format(comp)] < 0:
    #                 #     sts['use{:s}'.format(comp)] = False
    #                 # else:
    #                 #     sts['use{:s}'.format(comp)] = False
    #             else:
    #                 if sts_p['VR_{:s}'.format(comp)] >= threshold:
    #                     sts['use{:s}'.format(comp)] = True
    #                 elif sts_p['VR_{:s}'.format(comp)] < -0.005:
    #                     sts['use{:s}'.format(comp)] = False
    #                 # else:
    #                 #     sts['use{:s}'.format(comp)] = False
    #         if sts['useZ'] or sts['useN'] or sts['useE']:
    #             self.fmax = max(self.fmax, sts['fmax'])
    #     else:
    #         sys.exit('Error sensitivity parameter order')
