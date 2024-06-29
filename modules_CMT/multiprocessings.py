#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from copy import deepcopy
from modules_CMT.extras import q_filter, decimate, moving_average_scale, covariance_function, correlation_function
from modules_CMT.MouseTrap import mouse, ToDisplacement, demean
from modules_CMT.inverse_problem import calc_taper_window, precalc_greens
from obspy.core import AttribDict
from obspy import UTCDateTime, Stream
from obspy.clients.seedlink.seedlinkexception import SeedLinkException
from pyrocko.gf import Target, LocalEngine, MTSource
from pyrocko.obspy_compat.base import to_obspy_trace


def seedlink_mp(i, station, sl_client, t0, t1, comps_order, inventory, origin):
    # while i < len(self.stations):
    sta = station['code']
    net = station['network']
    loc = station['location']
    cha = station['channelcode']

    # if len(data_raw) > 0:
    #     if sta == data_raw[-1][0].stats.station:
    #         self.stations.pop(i)
    #         continue
    try:
        st = sl_client.get_waveforms(network=net, station=sta, location=loc, channel=f'{cha}*',
                                     starttime=t0, endtime=t1)
    except (SeedLinkException, TypeError):
        # self.stations.pop(i)
        # continue
        desc = '{0:s}:{1:s}:{2:s}: Waveform not Available. Removing station from further ' \
               'processing.'.format(net, sta, cha)
        return{'sts': station, 'sts_n': i, 'st': '', 'st_d': '', 'desc': desc}

    if len(st) != 3:
        # self.stations.pop(i)
        desc = '{0:s}:{1:s}:{2:s}: Gap in data / wrong number of components. Removing station from further ' \
                 'processing.'.format(net, sta, cha)
        return {'sts': station, 'sts_n': i, 'st': '', 'st_d': '', 'desc': desc}
        # continue

    if st[0].stats.location != loc:  # seedlink not filter data location (obspy seedlink/server bugs?)
        station['location'] = st[0].stats.location
    if st[0].stats.channel[:2] != cha:
        station['channelcode'] = st[0].stats.channel[:2]

    ch = {}
    for comp in range(3):
        ch[st[comp].stats.channel[2]] = st[comp]
    if sorted(ch.keys()) != ['E', 'N', 'Z']:
        desc = '{0:s}:{1:s}:{2:s}: Unoriented components. ' \
                 'Removing station from further processing.'.format(net, sta, cha)
        return {'sts': station, 'sts_n': i, 'st': '', 'st_d': '', 'desc': desc}
        # self.stations.pop(i)
        # self.log('{0:s}:{1:s}:{2:s}: Unoriented components. '
        #          'Removing station from further processing.'.format(net, sta, cha))
        # continue
    str_comp = comps_order
    st = Stream(traces=[ch[str_comp[0]], ch[str_comp[1]], ch[str_comp[2]]])
    # data_raw.append(st)
    station['useN'] = station['useE'] = station['useZ'] = True

    for tr in st:
        filt_resp = inventory.select(location=tr.stats.location, network=tr.stats.network,
                                     station=tr.stats.station, channel=tr.stats.channel, time=origin)
        if len(filt_resp) == 0:
            # data_raw.pop()
            # self.stations.pop(i)
            desc = 'Cannot find responses metadata(s) for station {0:s}:{1:s}:{2:s}. Removing station from ' \
                   'further processing.'.format(net, sta, cha)
            return {'sts': station, 'sts_n': i, 'st': st, 'st_d': st[0].stats.delta, 'desc': desc}
            # break
        response = filt_resp[0][0][0].response
        try:
            poles = response.response_stages[0].poles
            zeros = response.response_stages[0].zeros
            norm_fact = response.response_stages[0].normalization_factor
            inst_sens = response.instrument_sensitivity.value
            # stage_gain = response.response_stages[0].stage_gain
        except IndexError:
            # data_raw.pop()
            # self.stations.pop(i)
            desc = 'Responses metadata(s) for station {0:s}:{1:s}:{2:s} is not complete. Removing station from ' \
                   'further processing.'.format(net, sta, cha)
            return {'sts': station, 'sts_n': i, 'st': st, 'st_d': st[0].stats.delta, 'desc': desc}
            # break

        tr.stats.paz = AttribDict({
            'sensitivity': inst_sens,
            'poles': poles,
            'gain': norm_fact,
            'zeros': zeros
        })
    # else:
    desc = 'data ok'
    # i += 1  # station not removed
    return {'sts': station, 'sts_n': i, 'st': st, 'st_d': st[0].stats.delta, 'desc': desc}
    # if not st[0].stats.delta in data_deltas:
    #     data_deltas.append(st[0].stats.delta)


def mouse_mp(i, stream, station, origin, mouse_len, mouse_onset, fit_t1, fit_t2c, figures, figures_mkdir):
    # for st0 in self.data_raw:
    # i += 1
    out = ''
    amp = 0

    st = stream.copy()
    sta = st[0].stats.station
    # stn_idx = next((index for (index, stn) in enumerate(stations) if stn['code'] == sta),
    #                None)  # q_rmv stn_idx max min freq
    t_arr = station["arr_time"]
    t_start = max(st[0].stats.starttime, st[1].stats.starttime, st[2].stats.starttime)
    t_start_origin = origin - t_start
    t_arr_origin = origin + t_arr - 20 - t_start  # 20 to ensure data not contain arrival signal
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
            station['use' + stats.channel[2]] = False
        else:
            onset, amp, dummy, dummy, fit = m1.params(degrees=True)
            amp = abs(amp)
            detected = False
            num_fit += 1
            sum_amp += amp
            sum_fit += fit
            if (amp > 50e-8) or (amp > 10e-8 and fit > 0.8) or (amp > 7e-8 and fit > 0.9) or \
                    (amp > 5e-9 and fit > 0.94) or (fit > 0.985):
                # DEBUGGING: fit > 0.95 in the before-last parentheses?
                out += '  ' + sta + ' ' + stats.channel + ': MOUSE detected, ignoring component in inversion ' \
                                                          '(onset time: {o:6.1f} s, amplitude: {a:10.2e} ' \
                                                          'm.s^-2, fit: {f:7.2f})\n'. \
                    format(o=onset - t_start_origin, a=amp, f=fit)
                station['use' + stats.channel[2]] = False
                station['weight' + stats.channel[2]] = station['weight' + stats.channel[2]] - 0.0

                detected = True
            # else:
            #     out += '  ' + sta + ' ' + stats.channel + ': No MOUSE detected, ignoring component in inversion ' \
            #                                               '(onset time: {o:6.1f} s, amplitude: {a:10.2e} ' \
            #                                               'm.s^-2, fit: {f:7.2f})\n'. \
            #         format(o=onset - t_start_origin, a=amp, f=fit)  # selection_scheme
            if figures:
                if not os.path.exists(figures) and figures_mkdir:
                    os.mkdir(figures)
                m1.plot(st[comp], outfile=os.path.join(figures, 'mouse_' + ('no', 'YES')[detected] + '_' +
                                                       sta + '.' + st[comp].stats.channel + '.png'),
                        xmin=t_arr_origin - 75, xmax=t_arr_origin + fit_t2c + 50,
                        ylabel='displacement [counts]',
                        title="{{net:s}}:{{sta:s}} {{ch:s}}, fit: {fit:4.2f}".format(fit=fit))
    if num_fit != 0:
        station['mean_fit'] = sum_fit / num_fit
        station['mean_amp'] = sum_amp / num_fit

    return {'sts': station, 'sts_n': i, 'desc': out, 'amp': amp}


def time_shifting_mp(SHIFT, DATA, starttime, endtime, decimate, elemse_start_origin, npts_slice, event,
                     stations, nr, components, stations_index, max_samprate, taper_perc=None):
    """
    decimate shift
    :param SHIFT:
    :param DATA:
    :param starttime:
    :param endtime:
    :param decimate:
    :param elemse_start_origin:
    :param npts_slice:
    :param event:
    :param stations:
    :param nr:
    :param components:
    :param stations_index:
    :param max_samprate:
    :return:
    """
    # for SHIFT in range(self.SHIFT_min, self.SHIFT_max + 1, self.SHIFT_step):
    # data = deepcopy(self.data)
    shift = SHIFT / max_samprate
    # shifts.append(shift)
    data = []
    raw_data = []
    for st in DATA:
        st2 = st.slice(starttime + shift - elemse_start_origin, endtime + shift + 1)
        # we add 1 s to be sure that no index will point outside the range
        st2.trim(starttime + shift - elemse_start_origin, endtime + shift + 1, pad=True,
                 fill_value=0.)  # short records are not inverted, but they should by padded because of plotting
        st2.decimate(decimate, no_filter=True)
        st_raw = st.slice(starttime + shift - elemse_start_origin,
                          endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
        st_raw.trim(starttime + shift - elemse_start_origin, endtime + shift + 1, pad=True,
                    fill_value=0.)  # short records are not inverted, but they should by padded because of plotting
        st_raw.decimate(decimate, no_filter=True)
        stats = st2[0].stats
        fmin = stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmin']
        fmax = stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])]['fmax']
        tarr = event['t'] + stations_index['_'.join([stats.network, stats.station, stats.location,
                                                     stats.channel[0:2]])]['arr_time']
        tlen = calc_taper_window(stations_index['_'.join([stats.network, stats.station, stats.location,
                                                          stats.channel[0:2]])]['dist'])
        if taper_perc:
            q_filter(st2, fmin, fmax, tarr + shift, tlen, taper_perc)
        else:
            q_filter(st2, fmin, fmax, tarr + shift, tlen)
            # q_filter(st2, fmin, fmax)
        st2.trim(starttime + shift, endtime + shift + 1)  # add 1 s to be sure no index point outside the range
        st_raw.trim(starttime + shift, endtime + shift + 1)  # add 1 s to be sure  no index point outside the range
        data.append(st2)
        raw_data.append(st_raw)
    # data_shifts.append(data)
    # raw_data_shifts.append(raw_data)
    c = 0
    d_shift = np.empty((components * npts_slice, 1))
    # d_shift2 = np.empty((self.components * self.npts_slice, 1))
    for r in range(nr):
        for comp in range(3):
            if stations[r][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # component 'use in inversion'
                weight = stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]
                try:
                    d_shift[c * npts_slice:c * npts_slice + npts_slice] = \
                        data[r][comp].data[:npts_slice].reshape(npts_slice, 1) * weight
                except:
                    # desc = 'Index out of range while generating shifted data vectors. ' \
                    #        'Waveform file probably too short.'
                    # print('values for debugging: ', r, comp, c, self.npts_slice, i, c * self.npts_slice + i,
                    #       len(d_shift), len(data[r][comp].data), SHIFT)
                    raise Exception('Index out of range while generating shifted data vectors. '
                                    'Waveform file probably too short.')
                c += 1

    return {'shift': shift, 'raw_data_shift': raw_data, 'data_shift': data, 'd_shift': d_shift}


def find_optimum_params_mp(stn_num, list_sts, npts_slice, data_raw_shifts, list_models, seismogram, t_min, npts_elemse,
                           samprate, comps_order, centroid, event, GF_store_dir, freq_shifts, elemse_start_origin,
                           sensitivity_file, stations, idx_use, idx_weight, taper_perc=None):
    """
    Calculates the variance reduction from each component and the variance reduction from a subset of stations
    of difference model and freq range.
    find optimum model and freq for each station, use all comp with good VR
    Module :class:`multiprocessing` does not allow running function of the same class in parallel,
        so the function :func:`invert` cannot be method of class :class:`modules_CMT` and this wrapper is needed.
    """
    npts = npts_slice
    data_raw = data_raw_shifts[centroid['shift_idx']][stn_num]
    # data_raw = data_raw_shift

    sts = list_sts[stn_num]

    sts_sens_param = {}
    for model in list_models:
        targets = [Target(quantity=seismogram, lat=float(sts['lat']), lon=float(sts['lon']),
                          store_id=model, tmin=t_min, tmax=(npts_elemse - 1) / samprate,
                          codes=(sts['network'], sts['code'], '', channel_code))
                   for channel_code in comps_order]

        elemse_raw = precalc_greens(event, centroid, targets, [sts], comps_order, GF_store_dir)

        for j in range(len(freq_shifts)):
            fshift = freq_shifts[j]
            elemse = deepcopy(elemse_raw)
            data = deepcopy(data_raw)
            elemse2 = deepcopy(elemse_raw)
            data2 = deepcopy(data_raw)  # data freq 2 times higher than freq inversion for residual covariance matrix

            stn_fshift = [stn for stn in fshift if stn['code'] == sts['code'] and
                          stn['network'] == sts['network']]
            sts['fmin'] = fmin = stn_fshift[0]['fmin']
            sts['fmax'] = fmax = stn_fshift[0]['fmax']
            sts['model'] = model
            tarr2 = UTCDateTime(0) + t_min + sts['arr_time']
            tlen = calc_taper_window(sts['dist'])
            for e in range(6):
                if taper_perc:
                    q_filter(elemse[0][e], fmin, fmax, tarr2, tlen, taper_perc)
                else:
                    q_filter(elemse[0][e], fmin, fmax, tarr2, tlen)
                elemse[0][e].trim(UTCDateTime(0) + elemse_start_origin)
                q_filter(elemse2[0][e], fmin / 2, fmax * 2)
                elemse2[0][e].trim(UTCDateTime(0) + elemse_start_origin)

            tarr = event['t'] + sts['arr_time'] + centroid['shift']
            if taper_perc:
                q_filter(data, fmin, fmax, tarr, tlen, taper_perc)
            else:
                q_filter(data, fmin, fmax, tarr, tlen)
            q_filter(data2, fmin / 2, fmax * 2)
            # COMPS_USED = 0
            MISFIT = 0
            NORM_D = 0
            VR_sum = 0
            SYNT = {}
            SYNT2 = {}
            for comp in range(3):
                SYNT[comp] = np.zeros(npts)
                SYNT2[comp] = np.zeros(npts)
                for e in range(6):
                    SYNT[comp] += elemse[0][e][comp].data[0:npts] * centroid['a'][e, 0]
                    SYNT2[comp] += elemse2[0][e][comp].data[0:npts] * centroid['a'][e, 0]
            comps_used = 0
            for comp in range(3):
                synt = SYNT[comp]
                d = data[comp][0:npts]
                synt2 = SYNT2[comp]
                d2 = data2[comp][0:npts]

                comps_used += 1
                res = d2 - synt2
                # res = d - synt
                misfit = np.sum(np.square(d - synt))
                norm_d = np.sum(np.square(d))
                # t = np.arange(0, (npts-0.5) / samprate, 1. / samprate) # DEBUG
                # fig = plt.figure() # DEBUG
                # plt.plot(t, synt, label='synt') # DEBUG
                # plt.plot(t, d, label='data') # DEBUG
                # plt.plot(t, d-synt, label='difference') # DEBUG
                # plt.legend() # DEBUG
                # plt.show() # DEBUG
                VR = 1 - misfit / norm_d
                sts[{0: 'res_Z', 1: 'res_N', 2: 'res_E'}[comp]] = res
                sts[{0: 'VR_Z', 1: 'VR_N', 2: 'VR_E'}[comp]] = VR
                MISFIT += misfit
                NORM_D += norm_d
                VR_sum = 1 - MISFIT / NORM_D
                # VR_avg += VR
                # COMPS_USED += 1
            sts['VR'] = VR_sum
            # sts['VR'] = VR_avg/3

            if not sts_sens_param:
                sts_sens_param = {'code': sts['code'], 'net': sts['network'], 'model': sts['model'],
                                  'fmin': sts['fmin'], 'fmax': sts['fmax'], 'fid': j + 1, 'VR': sts['VR'],
                                  'res_Z': sts['res_Z'], 'res_N': sts['res_N'], 'res_E': sts['res_E'],
                                  'VR_Z': sts['VR_Z'], 'VR_N': sts['VR_N'], 'VR_E': sts['VR_E'], 'dist': sts['dist']}
            else:
                if sts['VR'] > sts_sens_param['VR']:
                    if sts['code'] == sts_sens_param['code']:
                        sts_sens_param['model'] = sts['model']
                        sts_sens_param['fmin'] = sts['fmin']
                        sts_sens_param['fmax'] = sts['fmax']
                        sts_sens_param['fid'] = j + 1
                        sts_sens_param['VR'] = sts['VR']
                        sts_sens_param['VR_Z'] = sts['VR_Z']
                        sts_sens_param['VR_N'] = sts['VR_N']
                        sts_sens_param['VR_E'] = sts['VR_E']
                        sts_sens_param['res_Z'] = sts['res_Z']
                        sts_sens_param['res_N'] = sts['res_N']
                        sts_sens_param['res_E'] = sts['res_E']
                    else:
                        sys.exit('Error sensitivity parameter order')
            if sensitivity_file:
                print_sens_params(stations, idx_use, idx_weight, sensitivity_file, j + 1)
    return sts_sens_param


def print_sens_params(stations, idx_use, idx_weight, out_file, fid='', final=False):
    if final:
        out = '\n\nComponents used in final inversion and their weights\nstation     ' \
              '\t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\tvelmod\tVR\t Z\t N\t E\t *fid\n            ' \
              '\t   \t   \t   \t   \t(km)\t (deg)\t(Hz)\t(Hz)\n'
    else:
        out = f'\nfrequency shift id: {fid}\nComponents used in initial inversion and their weights\nstation     ' \
              f'\t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\tvelmod\tVR\t Z\t N\t E\t *fid\n            ' \
              f'\t   \t   \t   \t   \t(km)\t (deg)\t(Hz)\t(Hz)\n'
    stn = stations
    for r in range(len(stations)):
        out += '{net:>3s}:{sta:5s} {loc:2s}\t{ch:2s} \t'.format(sta=stn[r]['code'], net=stn[r]['network'],
                                                                loc=stn[r]['location'], ch=stn[r]['channelcode'])
        for c in range(3):
            if stn[r][idx_use[c]]:
                out += '{0:3.1f}\t'.format(stn[r][idx_weight[c]])
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


def covariance_matrix_prenoise_mp(stn_num, list_sts, npts_slice, sts_noise, save_covariance_function, crosscovariance,
                                  toeplitz, correlation=False):
    """
    Creates covariance matrix :math:`C_d` from ``self.noise``.

    :param stn_num: station number
    :param list_sts: list of the stations
    :param npts_slice: number of sampling data used
    :param sts_noise: list of the noise datas
    :param save_covariance_function: If ``True``, save also the covariance function matrix,
        which can be plotted later.
    :type save_covariance_function: bool, optional
    :param crosscovariance: Set ``True`` to calculate crosscovariance between components. If ``False``,
        it assumes that noise at components is not correlated, so non-diagonal blocks are identically zero.
    :type crosscovariance: bool, optional
    :param toeplitz: If ``False``, scalling covariance function following dettmer, 2007.
    :type toeplitz: bool, optional
    :param correlation: If ``True``, using correlation function else using covariance function following dettmer, 2007.
    :type correlation: bool
    """

    sta = list_sts[stn_num]
    sts_CD = {'sts': sta['code'], 'dist': sta['dist']}
    n = npts_slice
    Cf = []

    idx = []
    if sta['useZ']:
        idx.append(0)
    if sta['useN']:
        idx.append(1)
    if sta['useE']:
        idx.append(2)
    size = len(idx) * n
    C = np.zeros((size, size))
    if save_covariance_function:
        Cf.append(np.ndarray(shape=(3, 3), dtype=np.ndarray))
    if not crosscovariance:
        for i in idx:
            I = idx.index(i) * n
            for j in idx:
                if i == j:
                    if toeplitz:
                        dev = np.ones(n)
                        if correlation:
                            cov = np.correlate(sts_noise[i].data, sts_noise[i].data, 'full') / len(sts_noise[i].data)
                        else:
                            cov = covariance_function(sts_noise[i].data) / len(sts_noise[i].data)  # TEST1
                    else:
                        ni, dev = moving_average_scale(sts_noise[i].data, int(len(sts_noise[i].data) / 16))
                        if correlation:
                            cov = np.correlate(ni, ni, 'full') / len(sts_noise[i].data)
                        else:
                            cov = covariance_function(ni) / len(sts_noise[i].data)
                    cov = decimate(cov, 2)  # sts_noise has 2-times higher sampling than data
                    middle = int(len(cov) / 2)
                    if save_covariance_function:
                        if toeplitz:
                            Cf[-1][i, i] = cov.copy() / np.max(cov.copy())
                        else:
                            Cf[-1][i, i] = cov.copy()
                    sts_CD['Cf_len'] = len(cov)
                    for k in range(n):
                        for l in range(k, n):
                            C[l + I, k + I] = C[k + I, l + I] = cov[middle + k - l] * dev[k] * dev[l]
                # if i != j:
                #     J = idx.index(j) * n
                #     C[I:I + n, J:J + n] = 0.
    else:
        for i in idx:
            I = idx.index(i) * n
            for j in idx:
                J = idx.index(j) * n
                if i > j:
                    continue
                if toeplitz:
                    devi = np.ones(n)
                    devj = np.ones(n)
                    if correlation:
                        cov = np.correlate(sts_noise[i].data, sts_noise[j].data, 'full') / len(sts_noise[i].data)
                    else:
                        cov = covariance_function(sts_noise[i].data, sts_noise[j].data) / len(sts_noise[i].data)
                else:
                    ni, devi = moving_average_scale(sts_noise[i].data, int(len(sts_noise[i].data) / 16))
                    nj, devj = moving_average_scale(sts_noise[j].data, int(len(sts_noise[i].data) / 16))
                    if correlation:
                        cov = np.correlate(ni, nj, 'full') / len(sts_noise[i].data)
                    else:
                        cov = covariance_function(ni, nj) / len(sts_noise[i].data)
                cov = decimate(cov, 2)  # sts_noise has 2-times higher sampling than data
                middle = int(len(cov) / 2)
                if save_covariance_function:
                    if toeplitz:
                        Cf[-1][i, j] = cov.copy() / np.max(cov.copy())
                    else:
                        Cf[-1][i, j] = cov.copy()
                    sts_CD['Cf_len'] = len(cov)
                for k in range(n):
                    if i == j:
                        for l in range(k, n):
                            C[l + I, k + I] = C[k + I, l + I] = cov[middle + k - l] * devi[k] * devi[l]
                    else:
                        for l in range(n):
                            C[l + J, k + I] = C[k + I, l + J] = cov[middle + k - l] * ((devi[k] + devj[k]) / 2) * \
                                                                ((devi[l] + devj[l]) / 2)

    for i in idx:  # add to diagonal 1% of its average
        I = idx.index(i) * n
        C[I:I + n, I:I + n] += np.diag(np.zeros(n) + np.average(C[I:I + n, I:I + n].diagonal()) * 0.01)

    sts_CD['Cf'] = Cf
    sts_CD['C'] = C

    return sts_CD


def covariance_matrix_residual_mp(stn_num, list_sts, npts_slice, save_covariance_function, crosscovariance,
                                  toeplitz, correlation=False):
    """
    Creates covariance matrix :math:`C_t` from ``residual between synthetic seismogram each velocity model``.

    :param stn_num: station number
    :param list_sts: list of the stations
    :param npts_slice: number of sampling data used
    :param sts_noise: list of the noise datas
    :param save_covariance_function: If ``True``, save also the covariance function matrix,
        which can be plotted later.
    :type save_covariance_function: bool, optional
    :param crosscovariance: Set ``True`` to calculate crosscovariance between components. If ``False``,
        it assumes that residual error at components is not correlated, so non-diagonal blocks are identically zero.
    :type crosscovariance: bool, optional
    :param toeplitz: If ``False``, scalling covariance function following dettmer, 2007.
    :type toeplitz: bool, optional
    :param correlation: If ``True``, using correlation function else using covariance function following dettmer, 2007.
    :type correlation: bool

    Author: Dettmer 2007

    """

    sta = list_sts[stn_num]
    sts_CD = {'sts': sta['code'], 'dist': sta['dist']}
    n = npts_slice
    Cf = []

    idx = []
    if sta['useZ']:
        idx.append(0)
    if sta['useN']:
        idx.append(1)
    if sta['useE']:
        idx.append(2)
    size = len(idx) * n
    C = np.zeros((size, size))
    if save_covariance_function:
        Cf.append(np.ndarray(shape=(3, 3), dtype=np.ndarray))
    if not crosscovariance:
        for i in idx:
            res_i = {0: 'res_Z', 1: 'res_N', 2: 'res_E'}[i]
            I = idx.index(i) * n
            for j in idx:
                if i == j:
                    if toeplitz:
                        dev = np.ones(n)
                        if correlation:
                            cov = np.correlate(sta[res_i], sta[res_i], 'full') / len(sta[res_i])
                        else:
                            cov = covariance_function(sta[res_i], sta[res_i]) / len(sta[res_i])
                    else:
                        ni, dev = moving_average_scale(sta[res_i],
                                                       int(len(sta[res_i]) / 16))
                        if correlation:
                            cov = np.correlate(ni, ni, 'full') / len(sta[res_i])
                        else:
                            cov = covariance_function(ni) / len(sta[res_i])
                    middle = int(len(cov) / 2)
                    if save_covariance_function:
                        if toeplitz:
                            Cf[-1][i, i] = cov.copy() / np.max(cov.copy())
                        else:
                            Cf[-1][i, i] = cov.copy()
                        sts_CD['Cf_len'] = len(cov)
                    for k in range(n):
                        for l in range(k, n):
                            C[l + I, k + I] = C[k + I, l + I] = cov[middle + k - l] * dev[k] * dev[l]
    else:
        for i in idx:
            res_i = {0: 'res_Z', 1: 'res_N', 2: 'res_E'}[i]
            I = idx.index(i) * n
            for j in idx:
                res_j = {0: 'res_Z', 1: 'res_N', 2: 'res_E'}[j]
                J = idx.index(j) * n
                if i > j:
                    continue
                if toeplitz:
                    devi = np.ones(n)
                    devj = np.ones(n)
                    if correlation:
                        cov = np.correlate(sta[res_i], sta[res_j], 'full') / len(sta[res_i])
                    else:
                        cov = covariance_function(sta[res_i], sta[res_j]) / len(sta[res_i])
                else:
                    ni, devi = moving_average_scale(sta[res_i], int(len(sta[res_i]) / 16))
                    nj, devj = moving_average_scale(sta[res_j], int(len(sta[res_i]) / 16))
                    if correlation:
                        cov = np.correlate(ni, nj, 'full') / len(sta[res_i])
                    else:
                        cov = covariance_function(ni, nj) / len(sta[res_i])
                middle = int(len(cov) / 2)
                if save_covariance_function:
                    if toeplitz:
                        Cf[-1][i, j] = cov.copy() / np.max(cov.copy())
                    else:
                        Cf[-1][i, j] = cov.copy()
                    sts_CD['Cf_len'] = len(cov)
                for k in range(n):
                    if i == j:
                        for l in range(k, n):
                            C[l + I, k + I] = C[k + I, l + I] = cov[middle + k - l] * devi[k] * devi[l]
                    else:
                        for l in range(n):
                            C[l + J, k + I] = C[k + I, l + J] = cov[middle + k - l] * ((devi[k] + devj[k]) / 2) * \
                                                                ((devi[l] + devj[l]) / 2)
    for i in idx:  # add to diagonal 1% of its average
        I = idx.index(i) * n
        C[I:I + n, I:I + n] += np.diag(np.zeros(n) + np.average(C[I:I + n, I:I + n].diagonal()) * 0.01)

    sts_CD['Cf'] = Cf
    sts_CD['C'] = C

    return sts_CD


# todo: precalc green add option out as element seismogram or synthetic (sumasi element)
# def precalc_greens2(event, grid, targets, stations, comps_order='ZNE', store_dir=''):
#     """
#     :param event: earthquake coordinate location to get precalculated green's function
#     :param grid: grid shift x,y,z from event as precalculated green's function source
#     :param stations: list of station parameter
#     :param targets: list of station targets to get precalculated green's function
#     :param comps_order: component order 'ZNE'
#     :param store_dir: directory of pyrocko precalculated greens function store
#     :return: 6 elementary seismogram for each receiver components
#     """
#
#     engine = LocalEngine(store_superdirs=[store_dir])
#
#     depth = grid['z']
#     n_shft = grid['y']
#     e_shft = grid['x']
#     MT = grid['a']
#
#     mt_source1 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
#                           depth=depth, east_shift=e_shft, north_shift=n_shft,
#                           mnn=0, mee=0, mdd=0, mne=1, mnd=0, med=0)
#     mt_source2 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
#                           depth=depth, east_shift=e_shft, north_shift=n_shft,
#                           mnn=0, mee=0, mdd=0, mne=0, mnd=1, med=0)
#     mt_source3 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
#                           depth=depth, east_shift=e_shft, north_shift=n_shft,
#                           mnn=0, mee=0, mdd=0, mne=0, mnd=0, med=-1)
#     mt_source4 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
#                           depth=depth, east_shift=e_shft, north_shift=n_shft,
#                           mnn=-1, mee=0, mdd=1, mne=0, mnd=0, med=0)
#     mt_source5 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
#                           depth=depth, east_shift=e_shft, north_shift=n_shft,
#                           mnn=0, mee=-1, mdd=1, mne=0, mnd=0, med=0)
#     mt_source6 = MTSource(lat=float(event['lat']), lon=float(event['lon']),
#                           depth=depth, east_shift=e_shft, north_shift=n_shft,
#                           mnn=1, mee=1, mdd=1, mne=0, mnd=0, med=0)
#
#     mt_sources = [mt_source1, mt_source2, mt_source3, mt_source4, mt_source5, mt_source6]
#
#     mt_synthetic_traces = [None] * len(mt_sources)
#     for i, elem_source in zip(range(len(mt_sources)), mt_sources):
#         mt_response = engine.process(elem_source, targets)
#         mt_synthetic_traces[i] = mt_response.pyrocko_traces()
#
#     # sts_num = 0  # q_test
#     # src_num = 0
#     # store = engine.get_store(store_id)
#     # dist = targets[sts_num].distance_to(mt_sources[src_num])
#     # # targets[sts_num].lat
#     # depth = mt_sources[src_num].depth
#     # arrival_time = store.t('begin', (depth, dist))
#     # print('{:s} dist = {:.4f} km, arr = {:.2f}'.format(targets[sts_num].codes[1], dist / 1000, arrival_time))
#
#     elemse_all = []
#     for sts in stations:
#         stn = sts['code']
#         net = sts['network']
#         elemse = []
#         for elem_source in mt_synthetic_traces:
#             select_trace = [trc for trc in elem_source if trc.station == stn and trc.network == net]
#             if len(select_trace) == 3:
#                 streams = Stream()
#                 for comp in comps_order:
#                     streams += to_obspy_trace([trc for trc in select_trace if trc.channel == comp][0])
#                 elemse.append(streams)
#         elemse_all.append(elemse)
#
#     elemse_all = []
#     for sts in stations:
#         stn = sts['code']
#         net = sts['network']
#         elemse = []
#         streams = Stream()
#         for comp in comps_order:
#             for n, elem_source in zip(len(mt_synthetic_traces), mt_synthetic_traces):
#                 select_trace = [trc for trc in elem_source if trc.station == stn and trc.network == net and trc.channel == comp]
#                 if n == 0:
#                     streams += to_obspy_trace(select_trace[0])
#                 else:
#
#
#             if len(select_trace) == 3:
#                 streams = Stream()
#                     streams += to_obspy_trace([trc for trc in select_trace if trc.channel == comp][0])
#                 elemse.append(streams)
#         elemse_all.append(elemse)
#
#
#     return elemse_all
