#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from synt_CMT.extras import lcmm, next_power_of_2


def set_time_grid(self):
    """
    Sets equidistant time grid defined by ``self.shift_min``, ``self.shift_max``, and ``self.shift_step`` (in secs).
    The corresponding values ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step`` are (rounded)
    in samples related to the highest sampling rate common to all stations.
    """
    # self.rupture_length =
    # rupture_length = math.sqrt(111 * 10**self.event['mag'])		# M6 ~ 111 km2, M5 ~ 11 km2
    # self.shift_max  = shift_max  = self.time_unc + rupture_length / self.rupture_velocity
    self.shift_max = shift_max = self.time_unc
    self.shift_min = shift_min = -self.shift_max
    self.shift_step = shift_step = 1. / self.fmax * 0.01
    self.SHIFT_min = int(round(shift_min * self.max_samprate))
    self.SHIFT_max = int(round(shift_max * self.max_samprate))
    self.SHIFT_step = max(int(round(shift_step * self.max_samprate)), 1)  # max() to avoid step beeing zero
    self.SHIFT_min = int(
        round(self.SHIFT_min / self.SHIFT_step)) * self.SHIFT_step  # shift the grid to contain zero time shift
    self.log('\nGrid-search over time:\n  min = {sn:5.2f} s ({Sn:3d} samples)\n'
             '  max = {sx:5.2f} s ({Sx:3d} samples)\n  step = {step:4.2f} s ({STEP:3d} samples)'.
             format(sn=shift_min, Sn=self.SHIFT_min, sx=shift_max, Sx=self.SHIFT_max,
                    step=shift_step, STEP=self.SHIFT_step))


def set_grid(self, depth_min=None, depth_max=None):
    """
    Generates grid ``self.grid`` of points, where the inverse problem will be solved.
    `Rupture length` is estimated as :math:`111 \cdot 10^{M_W}`.
    Horizontal diameter of the grid is determined as ``self.location_unc`` + `rupture_length`.
    Vertical half-size of the grid is ``self.depth_unc`` + `rupture_length`.
    If ``self.circle_shape`` is ``True``, the shape of the grid is cylinder, otherwise it is rectangular box.
    The horizontal grid spacing is defined by ``self.step_x``, the vertical by ``self.step_z``.

    :param self:
    :param depth_min: minimum depth
    :param depth_max: maximum depth
    :param cmt: find for centroid horizontal location or only depth

    """
    min_depth = self.store_mindepth
    max_depth = self.store_maxdepth
    step_z = self.step_z
    radius = self.grid_radius
    radius_z = self.grid_z_radius

    if self.centroid_inv:
        step_x = self.step_x
        # self.depth_min = depth_min = max(min_depth, self.event['depth'] - int(radius/step_z)*step_z)
        if depth_min is None:
            depth_min = max(min_depth, self.event['depth'] - int(radius_z / step_z) * step_z)
        if depth_max is None:
            depth_max = min(max_depth, self.event['depth'] + int(radius_z / step_z) * step_z)
        if depth_max > max_depth:
            self.log('\n Max centroid depth set to {} Km following Green\'s function max source depth'.
                     format(max_depth / 1000), printcopy=True)
            depth_max = max_depth
        self.depth_min = depth_min
        self.depth_max = depth_max  # min(max_depth, self.event['depth'] + int(1.5*radius/step_z)*step_z)
        depths = np.arange(depth_min, depth_max, step_z)
        if max(depths) + step_z <= depth_max:
            depths = np.arange(depth_min, depth_max + step_z, step_z)

        n_steps = int(radius / step_x)
        grid = []
        self.steps_x = []
        for i in range(-n_steps, n_steps + 1):
            x = i * step_x
            self.steps_x.append(x)
            for j in range(-n_steps, n_steps + 1):
                y = j * step_x
                if math.sqrt(x ** 2 + y ** 2) > radius and self.circle_shape:
                    continue
                for z in depths:
                    edge = z == depths[0] or z == depths[-1] or (
                            math.sqrt((abs(x) + step_x) ** 2 + y ** 2) > radius or
                            math.sqrt((abs(y) + step_x) ** 2 + x ** 2) > radius) and \
                           self.circle_shape or max(abs(i), abs(j)) == n_steps
                    grid.append({'x': x, 'y': y, 'z': z, 'err': 0, 'edge': edge})
        self.grid = grid
        self.depths = depths
        self.step_x = step_x
        self.step_z = step_z
        self.log(
            '\nGrid parameters:\n  number of points: {0:4d}\n  horizontal step: {1:5.0f} m'
            '\n  vertical step: {2:5.0f} m\n  grid radius: {3:6.3f} km\n  minimal depth: {4:6.3f} km'
            '\n  maximal depth: {5:6.3f} km'.format(
                len(self.grid), step_x, step_z, radius / 1e3, depth_min / 1e3, depth_max / 1e3))
    else:
        if depth_min is None:
            depth_min = max(min_depth, self.event['depth'] - int(radius_z / step_z) * step_z)
        if depth_max is None:
            depth_max = min(max_depth, self.event['depth'] + int(radius_z / step_z) * step_z)

        self.depth_min = depth_min
        self.depth_max = depth_max

        depths = np.arange(depth_min, depth_max, step_z)
        if max(depths) + step_z <= depth_max:
            depths = np.arange(depth_min, depth_max + step_z, step_z)
        grid = []
        self.steps_x = [0]
        x = y = 0
        for z in depths:
            edge = z == depths[0] or z == depths[-1]
            grid.append({'x': x, 'y': y, 'z': z, 'err': 0, 'edge': edge})
        self.grid = grid
        self.depths = depths
        self.log('\nGrid parameters:\n  number of points: {0:4d}\n'
                 '  minimal depth: {1:6.3f} km\n  maximal depth: {2:6.3f} km'.
                 format(len(self.grid), depth_min / 1e3, depth_max / 1e3))


def set_working_sampling(self, multiple8=False):  # q_edit
    """
    Determine maximal working sampling as at least 8-multiple of maximal inverted frequency (``self.fmax``).
    If needed, increases the value to eneables integer decimation factor.

    :param self:
    :param multiple8: if ``True``, force the decimation factor to be such multiple, that decimation can be done
        with factor 8 (more times, if needed) and finaly with factor <= 8. The reason for this is decimation
        pre-filter unstability for higher decimation factor (now not needed).
    :type multiple8: bool, optional
    """
    min_sampling = self.store_samprate
    SAMPRATE = 1. / lcmm(*self.data_deltas)
    # decimate = int(SAMPRATE / min_sampling)
    # if multiple8:
    #     if decimate > 128:
    #         decimate = int(decimate/64) * 64
    #     elif decimate > 16:
    #         decimate = int(decimate/8) * 8
    #     else:
    #         decimate = int(decimate)
    # else:
    #     decimate = int(decimate)
    decimate = int(SAMPRATE / min_sampling)  # q_edit
    self.max_samprate = SAMPRATE
    self.samprate = SAMPRATE / decimate
    self.logtext['samplings'] = samplings_str = ", ".join(
        ["{0:5.1f} Hz".format(1. / delta) for delta in self.data_deltas])
    self.log('\nSampling frequencies:\n  Data sampling: {0:s}\n  Common sampling: {3:5.1f}\n  Decimate factor: '
             '{1:3d} x\n  Sampling used: {2:5.1f} Hz'.format(samplings_str, decimate, self.samprate, SAMPRATE))


def set_time_window(self):
    """
    Determines number of samples for inversion (``self.npts_slice``) and for Green's function calculation
    (``self.npts_elemse`` and ``self.npts_exp``) from ``self.min_time`` and ``self.max_time``.

    :math:`\mathrm{npts\_slice} \le \mathrm{npts\_elemse} = 2^{\mathrm{npts\_exp}} < 2\cdot\mathrm{npts\_slice}`
    """
    # self.min_time(self.stations[0]['dist'])  # q_edit
    self.max_time(self.stations[self.nr - 1]['dist'])  # q_edit
    self.t_min = 0.  # q_edit
    self.elemse_start_origin = -self.t_min
    self.t_len = self.t_max - self.t_min
    self.npts_slice = int(math.ceil(self.t_len * self.samprate))
    self.npts_elemse = next_power_of_2(int(math.ceil(self.t_len * self.samprate)))
    if self.npts_elemse < 64:  # FIXED OPTION
        self.npts_exp = 6
        self.npts_elemse = 64
    else:
        self.npts_exp = int(math.log(self.npts_elemse, 2))


def set_Greens_parameters(self):
    """
    Sets parameters for Green's function calculation:
     - time window length ``self.tl``
     - number of frequencies ``self.freq``
     - spatial periodicity ``self.xl``

    Writes used parameters to the log file.
    """
    self.tl = self.npts_elemse / self.samprate
    # freq = int(math.ceil(fmax*tl))
    # self.freq = min(int(math.ceil(self.fmax*self.tl))*2, self.npts_elemse/2)
    self.freq = int(self.npts_elemse / 2) + 1
    # self.xl = max(np.ceil(self.stations[self.nr-1]['dist']/1000), 100)*1e3*20
    # self.log("\nGreen's function parameters:\n  npts: {0:4d}\n  tl: {1:4.2f}\n  freq: {2:4d}\n
    # npts for inversion: {3:4d}".format(self.npts_elemse, self.tl, self.freq, self.npts_slice))


def set_initial_frequencies(self, fmin=None, fmax=None):
    """
    Sets frequency range for each station according its distance.

    :param self:
    :type fmax: float, optional
    :param fmax: maximal inverted frequency for all stations
    :type fmin: float
    :param fmin: minimal inverted frequency for all stations
    """
    if not fmin:
        if float(self.event['mag']) < 1:
            fmin = 0.05
        elif float(self.event['mag']) < 2:
            fmin = 0.03
        elif float(self.event['mag']) < 3.5:
            fmin = 0.02
        elif float(self.event['mag']) < 4.5:
            fmin = 0.015
        else:
            fmax = 0.01
    if not fmax:
        if float(self.event['mag']) < 1:
            fmax = 0.2
        elif float(self.event['mag']) < 2:
            fmax = 0.13
        elif float(self.event['mag']) < 3.5:
            fmax = 0.08
        elif float(self.event['mag']) < 4.5:
            fmax = 0.07
        elif float(self.event['mag']) < 5:
            fmax = 0.06
        else:
            fmax = 0.05
    for stn in self.stations:
        stn['fmax'] = fmax
        stn['fmin'] = fmin
    self.fmax = fmax
    # self.count_components()


def set_working_frequencies(self, sampling_fmax=0.125, list_fmin=None, min_range=0.03):
    # todo : change sampling_max from store_min_sampling/4
    """
    Sets list of shifting frequency range for each station according its distance.

    :param self:
    :type sampling_fmax: float, optional
    :param sampling_fmax: maximal inverted frequency for all stations
    :type list_fmin: list
    :param list_fmin: list of several minimal inverted frequency
    :type min_range: float
    :param min_range: minimum range between min and max freq

    The maximal frequency for each station is determined according to the following formula:
    :math:`\min ( f_{max} = \mathrm{wavecycles} \cdot \mathrm{self.s\_velocity} / r, \; fmax )`,
    where `r` is the distance the source and the station.
    """
    # if float(self.event['mag']) < 3.8:
    #     lst_wavecycles = [4, 5, 6, 7]
    #     # lst_wavecycles = [6, 7, 8, 9]
    # else:
    #     lst_wavecycles = [2, 3, 4, 5]
    #     # lst_wavecycles = [4, 5, 6]
    #     # lst_wavecycles = [5, 6, 7, 8]
    # # lst_wavecycles = [4]
    #
    # if float(self.event['depth']) <= 15000:
    #     for i in range(len(lst_wavecycles)):
    #         lst_wavecycles[i] += 1
    #
    # if list_fmin is None:
    #     list_fmin=[0.01, 0.015, 0.02, 0.025, 0.03]
    #     # # list_fmin=[0.015, 0.02]
    #     # # list_fmin=[0.01, 0.015, 0.02]
    #     # # list_fmin=[0.005, 0.01, 0.015, 0.02]
    #     # if float(self.event['mag']) < 3.8:
    #     #     list_fmin=[0.02, 0.025, 0.03]
    #     # else:
    #     #     list_fmin=[0.01, 0.015, 0.02]
    #
    # freq_shifts = []
    # for fmin in list_fmin:
    #     for wl in lst_wavecycles:
    #         freq_shift = []
    #         for stn in self.stations:
    #             dist = np.sqrt(stn['dist']**2 + self.event['depth']**2)
    #             fmax = round(min(wl * self.s_velocity / dist, sampling_fmax), 3)
    #             # if fmax <= fmin:
    #             if fmax - fmin < 0.01:
    #                 fmax = fmin + 0.01
    #             # if fmax > fmin:
    #             freq_shift.append({'code': stn['code'], 'network': stn['network'], 'fmin': fmin, 'fmax': fmax})
    #         freq_shifts.append(freq_shift)
    if float(self.event['mag']) < 1:
        # list_fmax = [0.07, 0.08, 0.09, 0.1, 0.12]
        list_fmin = [0.05, 0.06, 0.08, 0.1, 0.12]
        list_fmax = [0.8, 0.9, 1.1, 1.4, 1.8, 2.3]
        # list_fmax = [0.08, 0.09, 0.1, 0.12]
    elif float(self.event['mag']) < 2:
        # list_fmax = [0.07, 0.08, 0.09, 0.1, 0.12]
        list_fmin = [0.04, 0.05, 0.06, 0.08, 0.1]
        list_fmax = [0.6, 0.8, 1.1, 1.3, 1.6, 2.]
        # list_fmax = [0.08, 0.09, 0.1, 0.12]
    elif float(self.event['mag']) < 3.5:
        # list_fmax = [0.07, 0.08, 0.09, 0.1, 0.12]
        list_fmax = [0.06, 0.07, 0.08, 0.09, 0.1, 0.12]
        # list_fmax = [0.08, 0.09, 0.1, 0.12]
    elif float(self.event['mag']) < 5:
        # list_fmax = [0.05, 0.06, 0.07, 0.08, 0.1]
        list_fmax = [0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
        # list_fmax = [0.06, 0.07, 0.08, 0.1]
    else:
        # list_fmax = [0.025, 0.035, 0.05, 0.06, 0.07, 0.08]
        # list_fmax = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
        list_fmax = [0.02, 0.04, 0.06, 0.08]
        list_fmin = [0.01, 0.015, 0.02, 0.025]
        # list_fmax = [0.04, 0.05, 0.06, 0.08]

    # if float(self.event['depth']) <= 15000:
    #     for i in range(len(list_fmax)):
    #         list_fmax[i] += 0.01

    if list_fmin is None:
        list_fmin = [0.01, 0.015, 0.02, 0.025, 0.03]

    list_fmin = [self.minfreq]  # test2
    list_fmax = [self.maxfreq]  # test2

    freq_shifts = []
    for fmin in list_fmin:
        for fmax in list_fmax:
            freq_shift = []
            for stn in self.stations:
                # fmax = fmx
                if fmax - fmin < min_range:
                    fmax = fmin + min_range
                freq_shift.append({'code': stn['code'], 'network': stn['network'], 'fmin': fmin, 'fmax': fmax})
            if not freq_shift in freq_shifts:
                freq_shifts.append(freq_shift)

    # if float(self.event['mag']) < 3.5:
    #     list_f = [(0.03, 0.08), (0.04, 0.09), (0.05, 0.15)]
    # elif float(self.event['mag']) < 5:
    #     list_f = [(0.02, 0.06), (0.03, 0.08), (0.04, 0.09)]
    # else:
    #     list_f = [(0.01, 0.04), (0.02, 0.06), (0.03, 0.08)]
    # freq_shifts = []
    # for f in list_f:
    #     freq_shift = []
    #     for stn in self.stations:
    #         fmin = f[0]
    #         fmax = f[1]
    #         freq_shift.append({'code': stn['code'], 'network': stn['network'], 'fmin': fmin, 'fmax': fmax})
    #     freq_shifts.append(freq_shift)

    self.freq_shifts = freq_shifts


def set_parameters(self, sampling_fmax=0.125, log=False, init=False, depth_min=None, depth_max=None):
    """
    Sets some technical parameters of the inversion.

    Technically, just runs following functions:
     - :func:`set_frequencies`
     - :func:`set_working_sampling`
     - :func:`set_time_window`
     - :func:`set_Greens_parameters`
     - :func:`set_grid`
     - :func:`set_time_grid`
     - :func:`count_components`

    The parameters are parameters of the same name of these functions.
    """
    self.set_working_sampling()
    self.set_time_window()
    self.set_Greens_parameters()
    self.set_grid(depth_min, depth_max)
    self.set_time_grid()
    self.count_components(log)
    if self.components < self.min_comp:
        self.log('\nNumber of components for this event is {:d} not processed.\n'.format(self.components))
        return False
    if init:
        self.set_working_frequencies(sampling_fmax=sampling_fmax, min_range=0.035)
    return True


def min_time(self, distance, mag=0, v=8000):
    """
    Defines the beginning of inversion time window in seconds from location origin time.
    Save it into ``self.t_min`` (now save 0 -- FIXED OPTION)

    :param self:
    :param distance: station distance in meters
    :type distance: float
    :param mag: magnitude (unused)
    :param v: the first inverted wave-group characteristic velocity in m/s
    :type v: float

    Sets ``self.t_min`` as minimal time of interest (in seconds).
    """
    t = distance / v  # FIXED OPTION
    # t = 5  # q_edit
    # if t<5:
    # t = 0
    self.t_min = t


def max_time(self, distance, mag=0, v=1000):
    """
    Defines the end of inversion time window in seconds from location origin time.
        Calculates it as :math:`\mathrm{distance} / v`.
    Save it into ``self.t_max``.

    :param self:
    :param distance: station distance in meters
    :type distance: float
    :param mag: magnitude (unused)
    :param v: the last inverted wave-group characteristic velocity in m/s
    :type v: float
    """
    # t = distance/v		# FIXED OPTION
    t = distance / self.max_t_v
    self.t_max = t


def config_dict(self):
    GF_stores = ''
    # keys = ['network', 'code', 'location', 'channelcode', 'fmin', 'fmax', 'model', 'weightZ', 'weightN', 'weightE',
    #         'VR_Z', 'VR_N', 'VR_E', 'sect']
    keys = ['network', 'code', 'location', 'channelcode', 'fmin', 'fmax', 'model', 'VR_Z', 'VR_N', 'VR_E', 'sect']
    stations = []
    for station in self.stations:
        comp_str = ''
        for comp in 'ZNE':
            if station[f'use{comp}']:
                comp_str += comp
        # sts_dict = {x: station[x] for x in keys}
        sts_dict = {}
        for key in keys:
            if key in station:
                sts_dict[key] = str(station[key])
        sts_dict['comp'] = comp_str
        stations.append(sts_dict)
    config = {'INVERSION_SETTINGS': {  # 'GF_stores': 'AUTO',  # AUTO/*GF_name*
        #  'seismogram': self.seismogram,  # velocity/displacement
        'agency': self.agency,
        'dc_interest_eq': self.dc_interest_eq,
        'centroid_inversion': self.centroid_inv,
        'cpu_threads': self.threads,
        'deviatoric': self.deviatoric,  # deviatoric/full_mt
        'freq_max': self.maxfreq,
        'freq_min': self.minfreq,  # Hz  freq range for automatic filter_corner
        'grid_radius': self.grid_radius,  # m
        'grid_step_x': self.step_x,  # m
        'grid_step_z': self.step_z,  # m
        'grid_time_uncertainty': self.time_unc,  # in seconds
        'grid_z_radius': self.grid_z_radius,  # m
        'max_time_velocity': self.max_t_v,  # m/s endtime seismogram using linear relationship (dist/max_t_v)
        'merge2seiscomp': self.merge_to_seiscomp,  # m/s endtime seismogram using linear relationship (dist/max_t_v)
        'mergeCD': self.force_merge_CD,  # m/s endtime seismogram using linear relationship (dist/max_t_v)
        'min_comp': self.min_comp,
        'min_sta': self.min_sta,
        'save_sens_file': self.save_sens_file,  # save sensitivity check for freq_band and vel_mod
        'snr_min': self.snr_min,
        'stn_min_dist': self.stn_min_dist,
        'stn_max_dist': self.stn_max_dist,
        'taper_perc': self.taper_perc*100,
        'use_cov_residual': self.use_cov_residual,
        'use_cov_noise': self.use_cov_noise
    },
        'STATIONS_SELECTION': stations}

    return config
