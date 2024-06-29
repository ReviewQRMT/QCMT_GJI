#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from copy import deepcopy
from obspy import read_inventory
from matplotlib.font_manager import FontProperties
# from datetime import timezone
# from datetime import datetime as dt
from synt_CMT.multiprocessings import time_shifting_mp, mouse_mp
from synt_CMT.geometry import azimuth_gap, divide_sector_edge
from synt_CMT.extras import q_filter, next_power_of_2
from synt_CMT.inverse_problem import calc_taper_window
from synt_CMT.MouseTrap import mouse, ToDisplacement, demean


def detect_mouse(self, mouse_len=2.5 * 60, mouse_onset=1 * 60, fit_t1=-20, fit_t2c=0, fit_t2v=1200,
                 figures=None, figures_mkdir=True):
    """
    Wrapper for :class:`MouseTrap`

    :param self:
    :param mouse_len: synthetic mouse length in second
    :param mouse_onset: the onset of the synthetic mouse is `mouse_onset` seconds after synthetic mouse starttime
    :param fit_t1: mouse fitting starts this amount of seconds after an event origin time (negative value to start
        fitting before the origin time)
    :param fit_t2c: mouse fitting endtime -- constant term
    :param fit_t2v: mouse fitting endtime -- linear term (see equation below)
    :param figures: plot directory location
    :param figures_mkdir: bool: plot mouse/not

    Endtime of fitting is :math:`t_2 = \mathrm{fit\_t2c} + \mathrm{dist} / \mathrm{fit\_t2v}`
    where :math:`\mathrm{dist}` is station epicentral distance.
    """
    self.log('\nMouse detection:')
    out = ''
    max_amp = 0
    i = -1
    for st0 in self.data_raw:
        i += 1
        st = st0.copy()
        sta = st[0].stats.station
        stn_idx = next((index for (index, stn) in enumerate(self.stations) if stn['code'] == sta), None)  # q_rmv stn_idx max min freq
        t_arr = self.stations[stn_idx]["arr_time"]
        t_start = max(st[0].stats.starttime, st[1].stats.starttime, st[2].stats.starttime)
        t_start_origin = self.event['t'] - t_start
        t_arr_origin = self.event['t'] + t_arr - 20 - t_start  # 20 to ensure data contain arrival signal
        paz = st[0].stats.paz
        demean(st, t_arr_origin)
        ToDisplacement(st)
        # error = PrepareRecord(st, t_start_origin) # demean, integrate, check signal-to-noise ratio
        # if error:
        # print ('    %s' % error)
        # create synthetic m1
        # t_end = min(st[0].stats.endtime, st[1].stats.endtime, st[2].stats.endtime)
        # t_len = t_end - t_start
        dt = st[0].stats.delta
        m1 = mouse(fit_time_before=50, fit_time_after=60)
        m1.create(paz, int(mouse_len / dt), dt, mouse_onset)
        # fit waveform by synthetic m1
        sum_fit = 0
        sum_amp = 0
        num_fit = 0
        for comp in range(3):
            stats = st[comp].stats
            # dist = self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])]['dist']
            try:
                # m1.fit_mouse(st[comp], t_min=t_start_origin+fit_t1, t_max=t_start_origin+fit_t2c+dist/fit_t2v)
                m1.fit_mouse(st[comp], t_min=t_arr_origin + fit_t1, t_max=t_arr_origin + fit_t2c)
            except:
                out += '  ' + sta + ' ' + stats.channel + ': MOUSE detecting problem (record too short?), ' \
                                                          'ignoring component in inversion\n'
                self.stations_index['_'.join([stats.network, sta, stats.location,
                                              stats.channel[0:2]])]['use' + stats.channel[2]] = False
            else:
                onset, amp, dummy, dummy, fit = m1.params(degrees=True)
                amp = abs(amp)
                detected = False
                num_fit += 1
                sum_amp += amp
                sum_fit += fit
                if amp > max_amp:
                    max_amp = amp
                if (amp > 50e-8) or (amp > 10e-8 and fit > 0.8) or (amp > 7e-8 and fit > 0.9) or \
                        (amp > 5e-9 and fit > 0.94) or (fit > 0.985):
                    # DEBUGGING: fit > 0.95 in the before-last parentheses?
                    out += '  ' + sta + ' ' + stats.channel + ': MOUSE detected, ignoring component in inversion ' \
                                                              '(onset time: {o:6.1f} s, amplitude: {a:10.2e} ' \
                                                              'm.s^-2, fit: {f:7.2f})\n'. \
                        format(o=onset - t_start_origin, a=amp, f=fit)
                    self.stations_index['_'.join([stats.network, sta, stats.location,
                                                  stats.channel[0:2]])]['use' + stats.channel[2]] = False
                    detected = True
                if figures:
                    if not os.path.exists(figures) and figures_mkdir:
                        os.mkdir(figures)
                    m1.plot(st[comp], outfile=os.path.join(figures, 'mouse_' + ('no', 'YES')[detected] + '_' +
                                                           sta + '.' + st[comp].stats.channel + '.png'),
                            xmin=t_arr_origin - 75, xmax=t_arr_origin + fit_t2c + 50,
                            ylabel='displacement [counts]',
                            title="{{net:s}}:{{sta:s}} {{ch:s}}, fit: {fit:4.2f}".format(fit=fit))
        if num_fit != 0:
            self.stations[i]['mean_fit'] = sum_fit / num_fit
            self.stations[i]['mean_amp'] = sum_amp / num_fit
    self.max_amp = max_amp
    self.logtext['mouse'] = out
    if out:
        self.log(out.rstrip(), newline=False, printcopy=True)
    else:
        self.log(out, newline=False, printcopy=True)


def detect_mouse_mp(self, mouse_len=2.5 * 60, mouse_onset=1 * 60, fit_t1=-20, fit_t2c=0, fit_t2v=1200,
                    figures=None, figures_mkdir=True):
    """
    Wrapper for :class:`MouseTrap`

    :param self:
    :param mouse_len: synthetic mouse length in second
    :param mouse_onset: the onset of the synthetic mouse is `mouse_onset` seconds after synthetic mouse starttime
    :param fit_t1: mouse fitting starts this amount of seconds after an event origin time (negative value to start
        fitting before the origin time)
    :param fit_t2c: mouse fitting endtime -- constant term
    :param fit_t2v: mouse fitting endtime -- linear term (see equation below)
    :param figures: plot directory location
    :param figures_mkdir: bool: plot mouse/not

    Endtime of fitting is :math:`t_2 = \mathrm{fit\_t2c} + \mathrm{dist} / \mathrm{fit\_t2v}`
    where :math:`\mathrm{dist}` is station epicentral distance.
    """
    self.log('\nMouse detection:')
    stations = []
    max_amp  = 0
    out      = ''

    pool = mp.Pool(processes=self.threads)
    results = [pool.apply_async(mouse_mp, args=(i, stream, station, self.event['t'], mouse_len, mouse_onset,
                                                fit_t1, fit_t2c, figures, figures_mkdir))
               for i, stream, station in zip(range(len(self.stations)), self.data_raw, self.stations)]
    output = [p.get() for p in results]
    output = sorted(output, key=lambda d: d['sts_n'])

    for d_out in output:

        stations.append(d_out['sts'])
        out += d_out['desc']
        if d_out['amp'] > max_amp:
            max_amp = d_out['amp']

    self.stations = stations
    self.max_amp = max_amp
    self.logtext['mouse'] = out
    if out:
        self.log(out.rstrip(), newline=False, printcopy=True)
    else:
        self.log(out, newline=False, printcopy=True)


def xml_correct_snr_filter(self, xml_file='', min_snr=10, window_width=150, figures=None, figures_mkdir=True):
    """
    Reselect data based on SNR value on corrected and filtered trace and add mean_snr value to self.station
        for station selection weighting

    :param self:
    :param xml_file: path of xml_response file
    :param min_snr: threeshold to use data in processing
    :param window_width: widht of both signal and noise window for calculation and plot
    :param figures: path of plotted image
    :param figures_mkdir: bool: plot snr/not
    """
    self.log('\nCorrect data and SNR checking:')
    out = ''
    not_pass = []
    inv = read_inventory(xml_file)

    # for i in range(len(self.data_raw)):
    max_snr = 0
    i = 0
    while i < len(self.data_raw):
        sts = self.stations[i]

        sta = self.data_raw[i][0].stats.station
        net = self.data_raw[i][0].stats.network
        filt_resp = inv.select(network=net, station=sta, time=self.event['t'])
        if len(filt_resp) == 0:
            self.data_raw.remove(self.data_raw[i])
            self.stations.remove(sts)
            self.log('Cannot find xml response file(s) for station {0:s}:{1:s}. Removing station from further '
                     'processing.'.format(net, sta), printcopy=True)
            continue

        # prepare and stream correct
        data_fs = self.data_raw[i][0].stats.sampling_rate
        f4 = data_fs / 2 - 1
        if data_fs < 20:
            f3 = f4 - 1
        elif data_fs < 40:
            f3 = f4 - 2
        else:
            f4 -= 4
            f3 = f4 - 5
        pre_filt = (0.005, 0.008, f3, f4)
        self.data_raw[i].remove_response(inventory=filt_resp, output=self.seisunit, pre_filt=pre_filt)

        st2 = self.data_raw[i].copy()
        # sta = st2[0].stats.station
        # net = st2[0].stats.network
        # sts = self.stations[i]
        maxfreq = sts['fmax']
        minfreq = sts['fmin']

        st2.filter("bandpass", freqmin=minfreq, freqmax=maxfreq, corners=2, zerophase=True)
        st2.detrend(type='demean')
        t_start = max(st2[0].stats.starttime, st2[1].stats.starttime, st2[2].stats.starttime)
        t_end = min(st2[0].stats.endtime, st2[1].stats.endtime, st2[2].stats.endtime)
        t_arr = self.event['t'] + sts['arr_time']
        # t_start_origin = self.event['t'] - t_start
        t_arr_origin = t_arr - t_start
        num_err = 0  # number of phase below snr criterion, if all phase is out, remove station
        sum_snr = 0
        num_snr = 0
        for tr in st2:
            stats = tr.stats
            # tr.simulate(paz_remove=tr.stats.paz)
            # tr.filter("bandpass", freqmin=minfreq, freqmax=maxfreq)
            # tr.detrend(type='demean')
            min_t = max(t_arr - window_width - 20, t_start)  # minus 20 sec to ensure signal not in noise window
            max_t = min(t_arr + window_width, t_end)
            snr = self.calc_snr(tr, [min_t, t_arr - 20], [t_arr, max_t])
            num_snr += 1
            sum_snr += snr
            if snr > max_snr:  # get max snr of all data to normalize weight on sts selection
                max_snr = snr
            if snr < min_snr:
                out += '  ' + sta + ' ' + stats.channel + ': SNR={snr:d} less than Min SNR {minsnr:d}, ignoring ' \
                                                          'component in inversion\n'.format(snr=int(snr),
                                                                                            minsnr=min_snr)
                self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])][
                    'use' + stats.channel[2]] = False
                detected = True
                num_err += 1
            elif np.isnan(snr):
                out += '  ' + sta + ' ' + stats.channel + ': cannot calculate SNR, ignoring ' \
                                                          'component in inversion\n'
                self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])][
                    'use' + stats.channel[2]] = False
                detected = True
                num_err += 1
            else:
                # out += '  ' + sta + ' ' + stats.channel + ': SNR={snr:d}, using component in ' \
                #                                           'inversion\n'.format(snr=int(snr))
                detected = False
            if figures:
                if not os.path.exists(figures) and figures_mkdir:
                    os.mkdir(figures)
                self.plot_snr(tr, outfile=os.path.join(figures, 'snr_' + ('PASS', 'no')[detected] + '_' + sta +
                                                       '.' + stats.channel + '.png'), arr_t=t_arr_origin,
                              xmin=t_arr_origin - window_width, xmax=t_arr_origin + window_width,
                              ylabel=self.unit, title="{{net:s}}:{{sta:s}} {{ch:s}}, SNR: {snr:d}".
                              format(snr=int(snr)))
        sts['mean_snr'] = sum_snr / num_snr

        if num_err == 3:
            out += '  All {1:s} components does not meet SNR criteria. ' \
                   'Removing station from further processing.\n'.format(net, sta)
            not_pass.append(i)

        i += 1

    if len(not_pass) > 0:
        for i in range(len(not_pass) - 1, -1, -1):
            del self.data_raw[not_pass[i]]
            del self.stations[not_pass[i]]

    self.create_station_index()
    self.data_are_corrected = True
    self.max_snr = max_snr
    self.log(out, newline=False)


def snr_filter(self, min_snr=5, window_width=150, figures=None, figures_mkdir=True):
    """
    Reselect data based on SNR value on corrected and filtered trace and add mean_snr value to
        self.station for station selection weighting

    :param self:
    :param min_snr: threeshold to use data in processing
    :param window_width: widht of both signal and noise window for calculation and plot
    :param figures: path of plotted image
    :param figures_mkdir: bool: plot snr/not
    """
    self.log('\nSNR checking:')
    out = ''
    max_snr = 0
    for i in range(len(self.data_raw) - 1, -1, -1):  # reverse order to delete sts and data if not pass
        st2 = self.data_raw[i].copy()
        sta = st2[0].stats.station
        net = st2[0].stats.network
        sts = self.stations[i]
        maxfreq = sts['fmax']
        minfreq = sts['fmin']
        t_start = max(st2[0].stats.starttime, st2[1].stats.starttime, st2[2].stats.starttime)
        t_end = min(st2[0].stats.endtime, st2[1].stats.endtime, st2[2].stats.endtime)
        t_arr = self.event['t'] + sts['arr_time']
        t_arr_origin = t_arr - t_start  # t_arr when ot = 0
        num_err = 0  # number of phase below snr criterion, if all phase is out, remove station and data_raw
        sum_snr = 0
        num_snr = 0
        # statsts = st2[0].stats
        for tr in st2:
            stats = tr.stats
            if not self.data_are_corrected:
                if self.seisunit == 'DISP':
                    tr.stats.paz['zeros'].append(0j)
                tr.simulate(paz_remove=tr.stats.paz)
            tr.filter("bandpass", freqmin=minfreq, freqmax=maxfreq)
            tr.detrend(type='demean')
            min_t = max(t_arr - window_width - 20, t_start)  # minus 20 sec to ensure signal not in noise window
            max_t = min(t_arr + window_width, t_end)
            if min_t >= (t_arr - 20) or t_arr >= max_t:
                out += '  {sta:s} {ch:s}: cannot calculate SNR (data too short), ignoring component in ' \
                       'inversion\n'.format(sta=sta, ch=stats.channel)
                self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])][
                    'use' + stats.channel[2]] = False
                num_err += 1
                continue
            snr = self.calc_snr(tr, [min_t, t_arr - 20], [t_arr, max_t])
            # if snr > max_snr:  # get max snr of all data to normalize weight on stations selection
            #     max_snr = snr
            # if snr < min_snr:
            # out += '  {sta:s} {ch:s}: SNR={snr:d} less than Min SNR {minsnr:d}, ignoring ' \
            #        'component in inversion\n'.format(sta=sta, ch=stats.channel, snr=int(snr), minsnr=min_snr)
            # self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])][
            #     'use' + stats.channel[2]] = False
            # detected = True
            # num_err += 1
            if np.isnan(snr):
                out += '  {sta:s} {ch:s}: cannot calculate SNR, ignoring component in inversion\n'. \
                    format(sta=sta, ch=stats.channel)
                # self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])][
                #     'use' + stats.channel[2]] = False
                num_err += 1
                continue
            # else:
            # out += '  {sta:s} {ch:s}: SNR={snr:d}, using component in inversion\n'.\
            #     format(sta=sta, ch=stats.channel, snr=int(snr))
            # detected = False
            num_snr += 1
            sum_snr += snr
            if figures:
                if not os.path.exists(figures) and figures_mkdir:
                    os.mkdir(figures)
                self.plot_snr(tr, outfile=os.path.join(figures, 'snr_' + sta + '.' + stats.channel + '.png'),
                              arr_t=t_arr_origin, xmin=t_arr_origin - window_width, xmax=t_arr_origin + window_width,
                              ylabel=self.unit, title="{{net:s}}:{{sta:s}} {{ch:s}}, SNR: {snr:.1f}".
                              format(snr=snr))

        if num_err == 3:
            out += '  All {1:s} components data has error. ' \
                   'Removing station from further processing.\n'.format(net, sta)
            del self.data_raw[i]
            del self.stations[i]
            continue

        if sum_snr != 0:
            sts['mean_snr'] = sum_snr / num_snr
            if sum_snr / num_snr > max_snr:  # get max snr of all data to normalize weight on stations selection
                max_snr = sum_snr / num_snr

        # if num_snr == 3 and num_err == 0 and sum_snr / num_snr >= min_snr:
        #     self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = True
        # else:
        if num_snr != 3 or num_err != 0 or sum_snr / num_snr < min_snr:
            self.stations[i]['useN'] = self.stations[i]['useE'] = self.stations[i]['useZ'] = False
            out += '  {sta:s}: Mean SNR={snr:.2f} less than Min SNR {minsnr:.2f}, removing station from ' \
                   'further processing \n'.format(sta=sta, snr=sum_snr / num_snr, minsnr=min_snr)
            del self.data_raw[i]
            del self.stations[i]

    self.max_snr = max_snr
    self.create_station_index()
    self.log(out, newline=False)


def calc_snr(self, tr, noise_window, signal_window):
    """
    Compute ratio of maximum signal amplitude to rms noise amplitude.

    :param self:
    :param tr: Trace to compute signal-to-noise ratio for
    :param noise_window: (start, end) of window to use for noise
    :param signal_window: (start, end) of window to use for signal

    :return: Signal-to-noise ratio, noise amplitude
    """

    noise_amp = np.sqrt(np.mean(np.square(tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data)))
    if np.isnan(noise_amp):
        self.log('Could not calculate noise amplitudo, setting to 1')
        noise_amp = 1.0
    try:
        signal_amp = np.sqrt(np.mean(np.square(
            tr.slice(starttime=signal_window[0], endtime=signal_window[1]).data)))  # mean signal
        # signal_amp = tr.slice(starttime=signal_window[0], endtime=signal_window[1]).data.max() # max signal
    except ValueError as e:
        self.log(str(e))
        # raise ValueError('Cannot compute Signal to Noise Ratio')
        return np.nan
    return signal_amp / noise_amp


def correct_data(self):
    """
    Remove instrument response ``self.data_raw`` using poles and zeroes
    """
    for st in self.data_raw:
        st.detrend(type='demean')
        st.filter('highpass', freq=0.008)
        for tr in st:
            if self.seisunit == 'DISP':
                tr.stats.paz['zeros'].append(0j)
            #     tr.stats.paz['zeros'] = [0j, 0j, 0j]
            # elif self.seisunit == 'VEL':
            #     tr.stats.paz['zeros'] = [0j, 0j]
            # else:
            #     tr.stats.paz['zeros'] = [0j]
            tr.simulate(paz_remove=tr.stats.paz)
    self.data_are_corrected = True


def correct_data_xml(self):
    """
    Remove instrument response ``self.data_raw`` using inventory xml_response
    # :param xml_file: path of xml_response file
    """
    # not_pass = []
    # inv = read_inventory(xml_file)

    for i in range(len(self.data_raw) - 1, -1, -1):  # reverse order to delete sts and data if not pass
        # while i < len(self.data_raw):
        sta = self.data_raw[i][0].stats.station
        net = self.data_raw[i][0].stats.network
        cha = self.data_raw[i][0].stats.channel[:2]
        loc = self.data_raw[i][0].stats.location
        # sts = self.stations[i]
        filt_resp = self.inv.select(location=loc, network=net, station=sta, channel=f"{cha}*", time=self.event['t'])
        if len(filt_resp) == 0:
            del self.data_raw[i]
            del self.stations[i]
            # self.data_raw.remove(self.data_raw[i])
            # self.stations.remove(sts)
            self.log('Cannot find xml response file(s) for station {0:s}:{1:s}. Removing station from further '
                     'processing.'.format(net, sta), printcopy=True)
            continue

        # prepare and stream correct
        data_fs = self.data_raw[i][0].stats.sampling_rate
        f4 = data_fs / 2 - 1
        if data_fs < 20:
            f3 = f4 - 1
        elif data_fs < 40:
            f3 = f4 - 2
        else:
            f4 -= 4
            f3 = f4 - 5
        pre_filt = (0.005, 0.008, f3, f4)
        self.data_raw[i].remove_response(inventory=filt_resp, output=self.seisunit, pre_filt=pre_filt,
                                         hide_sensitivity_mismatch_warning=True)

    self.create_station_index()
    self.data_are_corrected = True


def trim_filter_data(self, noise_slice=True, noise_starttime=None, noise_length=None):
    """
    Filter ``self.data_raw`` using function :func:`prefilter_data` and :func:`q_filter`.
    Decimate ``self.data_raw`` to common sampling rate ``self.max_samprate``.
    Optionally, copy a time window for the noise analysis.
    Copy a slice to ``self.data``.

    :param self:
    :type noise_slice: bool, optional
    :param noise_slice: If set to ``True``, copy a time window of the length ``lenght`` for later noise analysis.
        Copied noise is in ``self.noise``.
    :type noise_starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param noise_starttime: Set the starttime of the noise time window. If ``None``, the time window starts in time
        ``starttime``-``length`` (in other words, it lies just before trimmed data time window).
    :type noise_length: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param noise_length: Length of the noise time window (in seconds).
    """
    # Copy a slice of ``self.data_raw`` to ``self.data_unfiltered``
    # (the slice range is the same as for ``self.data``).

    # self.log('\nBandpass filter frequencies:\n  min: {0:4.2f}\n  max: {1:4.2f}'.format(self.fmin, self.fmax))
    # calculate time values for creating slices
    starttime = self.event['t'] + self.shift_min + self.t_min
    length = self.t_max - self.t_min + self.shift_max + 10
    endtime = starttime + length
    if noise_slice:
        if not noise_length:
            noise_length = length * 4
        if not noise_starttime:
            noise_starttime = starttime - noise_length
            noise_endtime = starttime
        else:
            noise_endtime = noise_starttime + noise_length
        DECIMATE = int(round(self.max_samprate / self.samprate))

    if self.data_orig:
        for st in self.data_orig:
            decimate_ = int(round(st[0].stats.sampling_rate / self.max_samprate))
            self.prefilter_data(st)
            st.decimate(decimate_, no_filter=True)
            st.trim(starttime, endtime)

    self.data = deepcopy(self.data_raw)
    # self.noise = []  # q_add
    noise = []  # q_add

    for st in self.data:
        stats = st[0].stats
        fmin = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
            'fmin']
        fmax = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
            'fmax']
        decimate_ = int(round(st[0].stats.sampling_rate / self.max_samprate))
        if noise_slice:
            noise.append(st.slice(noise_starttime, noise_endtime))
            # print self.noise[-1][0].stats.endtime-self.noise[-1][0].stats.starttime, '<', length*1.1 # DEBUG
            if (len(noise[-1]) != 3 or (
                    noise[-1][0].stats.endtime - noise[-1][0].stats.starttime < length * 1.1)) and \
                    self.stations_index[
                        '_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                        'use' + stats.channel[2]]:  # q_edit
                self.log(
                    'Noise slice too short to generate covariance matrix (station ' +
                    st[0].stats.station + '). Stopping generating noise slices.')
                noise_slice = False
                noise = []
            elif len(noise[-1]):
                q_filter(noise[-1], fmin / 2, fmax * 2)
                noise[-1].decimate(int(decimate_ * DECIMATE / 2),
                                   no_filter=True)  # noise has 2-times higher sampling than data
        self.prefilter_data(st)
        st.decimate(decimate_, no_filter=True)
        # q_filter(st, fmin, fmax) # moved to decimate_shift()
        st.trim(starttime, endtime)
    self.noise = noise


def prefilter_data(self, st):
    """
    Drop frequencies above Green's function computation high limit using :func:`numpy.fft.fft`.

    :param self:
    :param st: stream to be filtered
    :type st: :class:`~obspy.core.stream`
    """
    f = self.freq / self.tl
    for tr in st:
        npts = tr.stats.npts
        NPTS = next_power_of_2(npts)
        TR = np.fft.fft(tr.data, NPTS)
        df = tr.stats.sampling_rate / NPTS
        # print (NPTS, df, int(np.ceil(f/df)), f, tr.stats.delta) # DEBUG
        flim = int(np.ceil(f / df))
        # for i in range(flim, NPTS-flim+1):
        # TR[i] = 0+0j
        TR[flim:NPTS - flim + 1] = 0 + 0j
        tr_filt = np.fft.ifft(TR)
        tr.data = np.real(tr_filt[0:npts])


def decimate_shift_mp(self):
    """
    Wrapper to:
    Generate ``self.data_shifts`` where are multiple copies of ``self.data`` (needed for plotting).
    Decimate ``self.data_shifts`` to sampling rate for inversion ``self.samprate``.
    Generate ``self.d_shifts`` where are multiple vectors :math:`d`, each of them shifted according to
        ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step``

    Runs function :func:`time_shifting_mp` in parallel.
    """
    d_shifts = []
    data_shifts = []
    data_orig_shifts = []
    raw_data_shifts = []
    shifts = []
    starttime = self.event['t']  # + self.t_min
    length = self.t_max - self.t_min
    endtime = starttime + length
    decimate = int(round(self.max_samprate / self.samprate))

    if self.threads > 1:
        pool = mp.Pool(processes=self.threads)
        results = [pool.apply_async(time_shifting_mp,
                                    args=(SHIFT, self.data, self.data_orig, starttime, endtime, decimate, self.elemse_start_origin,
                                          self.npts_slice, self.event, self.stations, self.nr, self.components,
                                          self.stations_index, self.max_samprate, self.taper_perc))
                   for SHIFT in range(self.SHIFT_min, self.SHIFT_max + 1, self.SHIFT_step)]
        output = [p.get() for p in results]
        output = sorted(output, key=lambda shf: shf['shift'])
        for res in output:
            shifts.append(res['shift'])
            raw_data_shifts.append(res['raw_data_shift'])
            data_shifts.append(res['data_shift'])
            d_shifts.append(res['d_shift'])
            data_orig_shifts.append(res['data_orig_shift'])
        # for res in results:
        #     shifts.append(res.get()['shift'])
        #     raw_data_shifts.append(res.get()['raw_data_shift'])
        #     data_shifts.append(res.get()['data_shift'])
        #     d_shifts.append(res.get()['d_shift'])

    else:
        for SHIFT in range(self.SHIFT_min, self.SHIFT_max + 1, self.SHIFT_step):
            result = time_shifting_mp(SHIFT, self.data, self.data_orig, starttime, endtime, decimate, self.elemse_start_origin,
                                      self.npts_slice, self.event, self.stations, self.nr, self.components,
                                      self.stations_index, self.max_samprate, self.taper_perc)
            shifts.append(result['shift'])
            raw_data_shifts.append(result['raw_data_shift'])
            data_shifts.append(result['data_shift'])
            data_orig_shifts.append(result['data_orig_shift'])
            d_shifts.append(result['d_shift'])

    self.shifts = shifts
    self.raw_data_shifts = raw_data_shifts
    self.data_shifts = data_shifts
    self.data_orig_shifts = data_orig_shifts
    self.d_shifts = d_shifts


def decimate_shift(self):
    """
    Generate ``self.data_shifts`` where are multiple copies of ``self.data`` (needed for plotting).
    Decimate ``self.data_shifts`` to sampling rate for inversion ``self.samprate``.
    Generate ``self.d_shifts`` where are multiple vectors :math:`d`, each of them shifted according to
        ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step``
    """
    d_shifts = []
    data_shifts = []
    raw_data_shifts = []
    shifts = []
    starttime = self.event['t']  # + self.t_min
    length = self.t_max - self.t_min
    endtime = starttime + length
    decimate = int(round(self.max_samprate / self.samprate))
    for SHIFT in range(self.SHIFT_min, self.SHIFT_max + 1, self.SHIFT_step):
        # data = deepcopy(self.data)
        shift = SHIFT / self.max_samprate
        shifts.append(shift)
        data = []
        raw_data = []
        for st in self.data:
            st2 = st.slice(starttime + shift - self.elemse_start_origin,
                           endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
            st2.trim(starttime + shift - self.elemse_start_origin, endtime + shift + 1, pad=True,
                     fill_value=0.)  # short records are not inverted, but they should be padded because of plotting
            st2.decimate(decimate, no_filter=True)
            st_raw = st.slice(starttime + shift - self.elemse_start_origin,
                              endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
            st_raw.trim(starttime + shift - self.elemse_start_origin, endtime + shift + 1, pad=True,
                        fill_value=0.)  # short records are not inverted, but they should be padded because of plotting
            st_raw.decimate(decimate, no_filter=True)
            stats = st2[0].stats
            fmin = \
                self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                    'fmin']
            fmax = \
                self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                    'fmax']
            tarr = self.event['t'] + self.stations_index[
                '_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['arr_time']
            tlen = calc_taper_window(
                self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                    'dist'])
            q_filter(st2, fmin, fmax, tarr + shift, tlen, self.taper_perc)
            # q_filter(st2, fmin, fmax)
            st2.trim(starttime + shift,
                     endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
            st_raw.trim(starttime + shift,
                        endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
            data.append(st2)
            raw_data.append(st_raw)
        data_shifts.append(data)
        raw_data_shifts.append(raw_data)
        c = 0
        d_shift = np.empty((self.components * self.npts_slice, 1))
        # d_shift2 = np.empty((self.components * self.npts_slice, 1))
        for r in range(self.nr):
            for comp in range(3):
                if self.stations[r][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # component 'use in inversion'
                    weight = self.stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]
                    for i in range(self.npts_slice):
                        try:
                            d_shift[c * self.npts_slice + i] = data[r][comp].data[i] * weight
                        except:
                            self.log('Index out of range while generating shifted data vectors. '
                                     'Waveform file probably too short.', printcopy=True)
                            print('values for debugging: ', r, comp, c, self.npts_slice, i, c * self.npts_slice + i,
                                  len(d_shift), len(data[r][comp].data), SHIFT)
                            raise Exception('Index out of range while generating shifted data vectors. '
                                            'Waveform file probably too short.')
                    # try:
                    #     d_shift[c * self.npts_slice:c * self.npts_slice + self.npts_slice] = \
                    #         data[r][comp].data[:self.npts_slice].reshape(self.npts_slice, 1) * weight
                    # except:
                    #     self.log('Index out of range while generating shifted data vectors. '
                    #              'Waveform file probably too short.', printcopy=True)
                    #     # print('values for debugging: ', r, comp, c, self.npts_slice, i, c * self.npts_slice + i,
                    #     #       len(d_shift), len(data[r][comp].data), SHIFT)
                    #     raise Exception('Index out of range while generating shifted data vectors. '
                    #                     'Waveform file probably too short.')
                    c += 1
        d_shifts.append(d_shift)
    self.d_shifts = d_shifts
    self.data_shifts = data_shifts
    self.raw_data_shifts = raw_data_shifts
    self.shifts = shifts


def dist_mag_weight(dist, mag):
    """
    weight station for station selection by distance and magnitudo
    :return:
    """
    if dist < 50000:
        if mag < 4.4:
            weight = 2
        else:
            weight = 1
    elif dist < 150000:
        if mag < 4.4:
            weight = 4
        elif mag < 5.6:
            weight = 3
        else:
            weight = 2
    elif dist < 300000:
        if mag < 4.4:
            weight = 3
        elif mag < 5.6:
            weight = 4
        else:
            weight = 3
    else:
        if mag < 4.4:
            weight = 1
        elif mag < 5.6:
            weight = 2
        else:
            weight = 4

    return weight


def station_weight(self):
    """
    weight station for station selection by distance, magnitudo, snr, and mousetrap amp and fit
    :return:
    """
    for sts in self.stations:
        # if 'weight' in sts:
        #     weight = sts['weight']
        # else:
        weight = dist_mag_weight(sts['dist'], self.event['mag'])

        if 'mean_snr' in sts:
            weight += sts['mean_snr'] / self.max_snr
        if 'mean_amp' in sts:
            weight -= (sts['mean_amp'] / self.max_amp) * 0.5
        if 'mean_fit' in sts:
            weight -= sts['mean_fit'] * 0.5

        sts['weight'] = weight


def station_selection(self, sector_num=8, numsta_sect=1, outfile=''):
    """
    divide azimuth by n sector, calculate station weighting and select optimum sts each sector

    :param self:
    :param sector_num: number of divided sector
    :param numsta_sect: number of max station each sector
    :param outfile: filename of station selection plot image
    :return:
    """
    self.sector_number = sector_num
    self.numsta_sect = numsta_sect
    self.log('\nStation selection:')
    out = ''
    self.station_weight()
    sts = self.stations
    list_azdist = []
    list_azimuth = []
    for stn in sts:  # get list of az and dist of used station
        if stn['useE'] or stn['useN'] or stn['useZ']:
            list_azimuth.append(stn['az'])
        list_azdist.append((stn['az'], stn['dist'] / 1000, stn['code']))

    self.azimuth_gap = gap, start, end = azimuth_gap(
        list_azimuth)  # calculate az gap, start & end gap azimuth clockwise
    self.lst_qdr_edge = lst_qdr_edge = divide_sector_edge(start, end, sector_num)  # get list of sector boundary

    select_sts = []
    select_azdist = []
    for i in range(len(lst_qdr_edge) - 1):
        if lst_qdr_edge[i] == max(lst_qdr_edge):
            sel_sta = [stn for stn in sts if lst_qdr_edge[i] < stn['az'] or stn['az'] <= lst_qdr_edge[i + 1]]
        else:
            sel_sta = [stn for stn in sts if lst_qdr_edge[i] < stn['az'] <= lst_qdr_edge[i + 1]]
        for elsta in sel_sta:
            elsta['sect'] = i + 1
        if len(sel_sta) == 0:
            continue
        elif len(sel_sta) == 1:
            sel_sta = sel_sta[0]
            sel_sta['sector'] = i + 1
            select_azdist.append((sel_sta['az'], sel_sta['dist'] / 1000, sel_sta['weight'], sel_sta['code']))
            select_sts.append(sel_sta)
        else:
            select_sta = sorted(sel_sta, key=lambda k: k['weight']+k['weightZ'], reverse=True)
            # Mark lowest station weight on sector cluster to be deleted
            for j, elsta in zip(range(len(select_sta)), select_sta):
                if j < numsta_sect:
                    elsta['sector'] = i + 1
                    select_azdist.append((elsta['az'], elsta['dist'] / 1000, elsta['weight'], elsta['code']))
                    select_sts.append(elsta)
                if j >= numsta_sect:
                    elsta['sect_cl'] = i + 1

    if outfile:
        fontP = FontProperties()
        fontP.set_size('xx-small')
        ax = plt.subplot(111, projection='polar')
        if start > end:  # plot gap shading
            ax.fill_between(np.linspace(np.deg2rad(start), np.deg2rad(360), 100), 0, sts[-1]['dist'] / 1000,
                            alpha=0.2, color='b', label='Max Azimuth Gap {:d}$^\circ$'.format(int(gap)))
            ax.fill_between(np.linspace(np.deg2rad(0), np.deg2rad(end), 100), 0, sts[-1]['dist'] / 1000,
                            alpha=0.2, color='b')
        else:
            ax.fill_between(np.linspace(np.deg2rad(start), np.deg2rad(end), 100), 0, sts[-1]['dist'] / 1000,
                            alpha=0.2, color='b', label='Max Azimuth Gap {:d}$^\circ$'.format(int(gap)))

        for az in lst_qdr_edge:
            ax.plot([np.deg2rad(az), np.deg2rad(az)], [0, sts[-1]['dist'] / 1000], lw=1, c='k')

        for (az, dst, wgt, stn), j in zip(select_azdist, range(len(select_azdist))):
            if j == 0:
                ax.scatter(np.deg2rad(az), dst, c='red', marker='v', s=wgt * 15, edgecolors='k',
                           label='Used Station', zorder=10)
            else:
                ax.scatter(np.deg2rad(az), dst, c='red', marker='v', s=wgt * 15, edgecolors='k', zorder=10)
            plt.text(np.deg2rad(az), dst, stn, horizontalalignment='center', verticalalignment='top',
                     fontsize=9, weight='bold', zorder=10)

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

    for stn, dt_raw in zip(self.stations.copy(), self.data_raw.copy()):
        out += '  {net:s} {sta:s} {ch:s} : dist={dist:.1f} km, az={az:.1f}, weight={wgt:.1f}, '. \
            format(net=stn['network'], sta=stn['code'], ch=stn['channelcode'], dist=stn['dist'] / 1000,
                   az=stn['az'], wgt=stn['weight'])
        if 'mean_snr' in stn:
            out += 'mean_snr={snr:.1f}, '.format(snr=stn['mean_snr'])
        if 'mean_amp' in stn:
            out += 'mean_mouse_amp={amp:.1f}, '.format(amp=stn['mean_amp'])
        if 'mean_fit' in stn:
            out += 'mean_mouse_fit={fit:.1f}, '.format(fit=stn['mean_fit'])
        if 'sector' in stn:
            out += 'sector {sct:d} - selected.\n'.format(sct=stn['sector'])
        else:
            stn['useZ'] = stn['useN'] = stn['useE'] = False  # keep, not used in inversion
            if 'sect' in stn:
                out += 'sector {sct:d}'.format(sct=stn['sect'])
            else:
                out += 'not passed SNR and Mouse_Detect'
                self.stations.remove(stn)
                self.data_raw.remove(dt_raw)
            if 'sect_cl' in stn:
                out += ' - removed (decluster azimuth).\n'
                self.stations.remove(stn)
                self.data_raw.remove(dt_raw)
            else:
                out += '.\n'
            # rmv_data = [data for data in self.data_raw if data[0].stats.station == stn['code']]
            # self.data_raw.remove(rmv_data[0])  # remove stream for eliminated sts

    # select_sts = sorted(select_sts, key=lambda stsn: stsn['dist'])
    self.log(out)
    # self.stations = select_sts
    self.create_station_index()


def obs_azimuth(self):
    list_azimuth = []
    for stn in self.stations:  # get list of az and dist of used station
        if stn['useE'] or stn['useN'] or stn['useZ']:
            list_azimuth.append(stn['az'])

    return azimuth_gap(list_azimuth)  # calculate az gap, start & end gap azimuth clockwise


def skip_short_records(self, noise=False):
    """
    Checks whether all records are long enough for the inversion and skips unsuitable ones.

    :param self:
    :parameter noise: checks also whether the record is long enough for generating the noise slice for the
        covariance matrix (if the value is ``True``, choose minimal noise length automatically;
        if it's numerical, take the value as minimal noise length)
    :type noise: bool or float, optional
    """
    self.log('\nChecking record length:')
    for st in self.data_raw:
        for comp in range(3):
            stats = st[comp].stats
            if stats.starttime > self.event['t'] + self.t_min + self.SHIFT_min or \
                    stats.endtime < self.event['t'] + self.t_max + self.SHIFT_max:
                self.log(
                    '  ' + stats.station + ' ' + stats.channel + ': record too short, ignoring component in inversion')
                self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                    'use' + stats.channel[2]] = False
            if noise:
                if type(noise) in (float, int):
                    noise_len = noise
                else:
                    noise_len = (self.t_max - self.t_min + self.SHIFT_max + 10) * 1.1 - self.SHIFT_min - \
                                self.t_min
                    # print stats.station, stats.channel, noise_len, '>', self.event['t']-stats.starttime # DEBUG
                if stats.starttime > self.event['t'] - noise_len:
                    self.log(
                        '  ' + stats.station + ' ' + stats.channel + ': record too short for noise covariance,'
                                                                     ' ignoring component in inversion')
                    self.stations_index[
                        '_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                        'use' + stats.channel[2]] = False


def print_components(self, out_file, final=False):
    if final:
        out = '\nComponents used in final inversion and their weights\nstation     ' \
              '\t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\tvelmod\tVR\t Z\t N\t E\t *fid\n            ' \
              '\t   \t   \t   \t   \t(km)\t (deg)\t(Hz)\t(Hz)\n'
    else:
        out = '\nComponents used in initial inversion and their weights\nstation     ' \
              '\t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\tvelmod\tVR\t Z\t N\t E\t *fid\n            ' \
              '\t   \t   \t   \t   \t(km)\t (deg)\t(Hz)\t(Hz)\n'
    stn = self.stations
    for r in range(self.nr):
        out += '{net:>3s}:{sta:5s} {loc:2s}\t{ch:2s} \t'.format(sta=stn[r]['code'], net=stn[r]['network'],
                                                                loc=stn[r]['location'], ch=stn[r]['channelcode'])
        for c in range(3):
            if stn[r][self.idx_use[c]]:
                out += '{0:3.1f}\t'.format(stn[r][self.idx_weight[c]])
            else:
                out += '---\t'
        if stn[r]['dist'] > 2000:
            out += '{0:4.0f}    '.format(stn[r]['dist'] / 1e3)
        elif stn[r]['dist'] > 200:
            out += '{0:6.1f}  '.format(stn[r]['dist'] / 1e3)
        else:
            out += '{0:8.3f}'.format(stn[r]['dist'] / 1e3)
        out += '\t{2:3.0f}\t{0:5.3f}\t{1:5.3f}'.format(stn[r]['fmin'], stn[r]['fmax'], stn[r]['az'])
        if 'model' in stn[r]:
            out += '\t{:s}'.format(stn[r]['model'])
        else:
            out += '\t---'
        if 'VR' in stn[r]:
            out += '\t{:.5f}'.format(stn[r]['VR'])
        else:
            out += '\t---'
        if 'VR_Z' in stn[r]:
            out += '\t{:.5f}'.format(stn[r]['VR_Z'])
        else:
            out += '\t---'
        if 'VR_N' in stn[r]:
            out += '\t{:.5f}'.format(stn[r]['VR_N'])
        else:
            out += '\t---'
        if 'VR_E' in stn[r]:
            out += '\t{:.5f}'.format(stn[r]['VR_E'])
        else:
            out += '\t---'
        if 'fid' in stn[r]:
            out += '\t{:d}'.format(stn[r]['fid'])
        else:
            out += '\t---'
        out += '\n'
    with open(out_file, 'a') as f:
        f.write(out)


def count_components(self, log=True):
    """
    Counts number of components, which should be used in inversion (e.g. ``self.stations[n]['useZ']
    = True`` for `Z` component). This is needed for allocation proper size of matrices used in inversion.

    :param self:
    :param log: if true, write into log table of stations and components with information about
        component usage and weight
    :type log: bool, optional
    """
    c = 0
    stn = self.stations
    for r in range(self.nr):
        if stn[r]['useZ']:
            c += 1
        if stn[r]['useN']:
            c += 1
        if stn[r]['useE']:
            c += 1
    self.components = c
    # print(c)
    if log:
        self.log_components()


def log_components(self):
    stn = self.stations
    out = '\nComponents used in inversion and their weights\nstation     \t   \t Z \t N \t E \tdist\tazimuth' \
          '\tfmin\tfmax\tvelmod\n            \t   \t   \t   \t   \t(km)\t (deg)\t(Hz)\t(Hz)\n'
    for r in range(self.nr):
        out += '{net:>3s}:{sta:5s} {loc:2s}\t{ch:2s} \t'.format(sta=stn[r]['code'], net=stn[r]['network'],
                                                                loc=stn[r]['location'],
                                                                ch=stn[r]['channelcode'])
        for c in range(3):
            if stn[r][self.idx_use[c]]:
                out += '{0:3.1f}\t'.format(stn[r][self.idx_weight[c]])
            else:
                out += '---\t'
        if stn[r]['dist'] > 2000:
            out += '{0:4.0f}    '.format(stn[r]['dist'] / 1e3)
        elif stn[r]['dist'] > 200:
            out += '{0:6.1f}  '.format(stn[r]['dist'] / 1e3)
        else:
            out += '{0:8.3f}'.format(stn[r]['dist'] / 1e3)
        out += '\t{2:3.0f}\t{0:5.3f}\t{1:5.3f}'.format(stn[r]['fmin'], stn[r]['fmax'], stn[r]['az'])
        if 'model' in stn[r]:
            out += '\t{:s}'.format(stn[r]['model'])
        else:
            out += '\t---'
        out += '\n'
    self.logtext['components'] = out
    self.log(out, newline=False)


def count_stations(self):
    num_sta_ch = 0
    sta_ch = self.stations_index
    for sta_nm in sta_ch:
        if sta_ch[sta_nm]['useE'] or sta_ch[sta_nm]['useN'] or sta_ch[sta_nm]['useZ']:
            num_sta_ch += 1

    self.num_sta = num_sta_ch
    if num_sta_ch < self.min_sta:
        self.log('\nNumber of used stations for this event is {:d}. Not processed.\n'.
                 format(num_sta_ch), printcopy=True)
        return False
    return True
