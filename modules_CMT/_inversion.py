#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from pyproj import Geod
from pyrocko.gf import Target
from modules_CMT.extras import q_filter, find_exponent
from modules_CMT.MT_comps import a2mt, decompose, decompose_mopad
from modules_CMT.inverse_problem import invert, calc_taper_window, precalc_greens


def run_inversion(self):
    """
    Runs function :func:`invert` in parallel.

    Module :class:`multiprocessing` does not allow running function of the same class in parallel,
        so the function :func:`invert` cannot be method of class :class:`modules_CMT` and this wrapper is needed.
    """
    grid = self.grid
    todo = []
    for i in range(len(grid)):
        point_id = str(i).zfill(4)
        grid[i]['id'] = point_id
        if not grid[i]['err']:
            todo.append(i)

    # create norm_d[shift]
    norm_d = []
    for shift in range(len(self.d_shifts)):
        d_shift = self.d_shifts[shift]
        if self.Cd_inv:
            if self.fit_cov:
                idx = 0
                dCd_blocks = []
                for C in self.Cd_inv:
                    size = len(C)
                    dCd_blocks.append(np.dot(d_shift[idx:idx + size, :].T, C))
                    idx += size
                dCd = np.concatenate(dCd_blocks, axis=1)
                norm_d.append(np.dot(dCd, d_shift)[0, 0])
            else:  # for fit calculation on original waveform even use covariance inversion
                norm_d.append(0)
                for i in range(self.npts_slice * self.components):
                    norm_d[-1] += d_shift[i, 0] * d_shift[i, 0]
        else:
            norm_d.append(0)
            for i in range(self.npts_slice * self.components):
                norm_d[-1] += d_shift[i, 0] * d_shift[i, 0]

    targets = []

    for sts in self.stations:
        targets += [Target(quantity=self.seismogram, lat=float(sts['lat']), lon=float(sts['lon']),
                           store_id=sts['model'], tmin=self.t_min, tmax=(self.npts_elemse - 1) / self.samprate,
                           codes=(sts['network'], sts['code'], '', channel_code))
                    for channel_code in self.comps_order]
    self.targets = targets
    if self.threads > 1:  # parallel
        pool = mp.Pool(processes=self.threads)
        results = [pool.apply_async(invert, args=(self.event, grid[i], self.d_shifts, norm_d, self.Cd_inv, self.nr,
                                                  self.components, self.comps_order, self.stations, targets,
                                                  self.npts_elemse, self.npts_slice, self.elemse_start_origin,
                                                  self.GF_store_dir, self.deviatoric, self.decompose, self.fit_cov,
                                                  self.taper_perc))
                   for i in todo]
        output = [p.get() for p in results]
    else:  # serial
        output = []
        for i in todo:
            result = invert(self.event, grid[i], self.d_shifts, norm_d, self.Cd_inv, self.nr, self.components,
                            self.comps_order, self.stations, targets, self.npts_elemse, self.npts_slice,
                            self.elemse_start_origin, self.GF_store_dir, self.deviatoric, self.decompose, self.fit_cov,
                            self.taper_perc)
            output.append(result)
    min_misfit = output[0]['misfit']
    for i in todo:
        grid[i].update(output[todo.index(i)])
        grid[i]['shift_idx'] = grid[i]['shift']
        # grid[i]['shift'] = self.shift_min + grid[i]['shift']*self.SHIFT_step/self.max_samprate
        grid[i]['shift'] = self.shifts[grid[i]['shift']]
        min_misfit = min(min_misfit, grid[i]['misfit'])
    self.max_sum_c = self.max_c = self.sum_c = 0
    for i in todo:
        gp = grid[i]
        gp['sum_c'] = 0
        for idx in gp['shifts']:
            GP = gp['shifts'][idx]
            GP['c'] = np.sqrt(gp['det_Ca']) * np.exp(-0.5 * (GP['misfit'] - min_misfit))
            gp['sum_c'] += GP['c']
        gp['c'] = gp['shifts'][gp['shift_idx']]['c']
        # gp['c'] = np.sqrt(gp['det_Ca']) * np.exp(-0.5 * gp['misfit']-min_misfit)
        assert (gp['c'] == gp['shifts'][gp['shift_idx']]['c'])
        self.sum_c += gp['sum_c']
        self.max_c = max(self.max_c, gp['c'])
        self.max_sum_c = max(self.max_sum_c, gp['sum_c'])
    # print(self.max_c) # DEBUG


def find_best_grid_point(self):
    """
    Set ``self.centroid`` to a grid point with higher variance reduction --the best solution of the inverse problem.
    """
    self.centroid = max(self.grid, key=lambda v: v['VR'])  # best grid point
    x = self.centroid['x']
    y = self.centroid['y']
    # az = np.degrees(np.arctan2(x, y))
    az = np.degrees(np.arctan2(y, x))  # bug on original code?
    dist = np.sqrt(x ** 2 + y ** 2)
    g = Geod(ellps='WGS84')
    self.centroid['lon'], self.centroid['lat'], baz = g.fwd(self.event['lon'], self.event['lat'], az, dist)


def VR_all_components(self, standardized_data=False):
    """
    Calculates the variance reduction from each component and the variance reduction from a subset of stations.

    :param self:
    :param standardized_data: calculate VR in standardized data or true waveform
    :type standardized_data: bool
    Add the variance reduction to ``self.stations`` with keys ``VR``, ``VR_Z``, ``VR_N``, and ``VR_Z``.
    """
    npts = self.npts_slice
    data = self.data_shifts[self.centroid['shift_idx']]
    elemse = precalc_greens(self.event, self.centroid, self.targets, self.stations, self.comps_order,
                            self.GF_store_dir)
    for r in range(self.nr):
        tarr2 = UTCDateTime(0) + self.t_min + self.stations[r]['arr_time']
        tlen = calc_taper_window(self.stations[r]['dist'])
        for e in range(6):
            q_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'], tarr2, tlen, self.taper_perc)
            # q_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'])
            elemse[r][e].trim(UTCDateTime(0) + self.elemse_start_origin)

    COMPS_USED = 0
    for sta in range(self.nr):
        MISFIT = 0
        NORM_D = 0
        VR_sum = 0
        SYNT = {}
        for comp in range(3):
            SYNT[comp] = np.zeros(npts)
            for e in range(6):
                SYNT[comp] += elemse[sta][e][comp].data[0:npts] * self.centroid['a'][e, 0]
        comps_used = 0
        for comp in range(3):
            # if self.Cd_inv and not self.stations[sta][{0:'useZ', 1:'useN', 2:'useE'}[comp]]:
            #     self.stations[sta][{0:'VR_Z', 1:'VR_N', 2:'VR_E'}[comp]] = None
            #     continue
            synt = SYNT[comp]
            d = data[sta][comp][0:npts]
            if standardized_data and self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                if self.LT3:
                    d    = np.zeros(npts)
                    synt = np.zeros(npts)
                    x1 = -npts
                    for COMP in range(3):
                        if not self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[COMP]]:
                            continue
                        x1 += npts; x2 = x1+npts
                        y1 = comps_used*npts; y2 = y1+npts
                        d    += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
                        synt += np.dot(self.LT3[sta][y1:y2, x1:x2], SYNT[COMP])

                elif self.Cd_inv and self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    d    = np.dot(self.LT[sta][comp], d)
                    synt = np.dot(self.LT[sta][comp], synt)

                else:
                    pass
            comps_used += 1
            # res = d - synt
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
            self.stations[sta][{0: 'VR_Z', 1: 'VR_N', 2: 'VR_E'}[comp]] = VR
            # self.stations[sta][{0: 'res_Z', 1: 'res_N', 2: 'res_E'}[comp]] = res
            MISFIT += misfit
            NORM_D += norm_d
            VR_sum = 1 - MISFIT / NORM_D
            COMPS_USED += 1
            # print sta, comp, VR, VR_sum # DEBUG
        self.stations[sta]['VR'] = VR_sum  # q_add for find_optimum_params


def save_seismo(self, file_d, file_synt, elem=False, cholesky=False):
    """
    Saves observed and simulated seismograms into files.

    :param self:
    :param file_d: filename for observed seismogram
    :type file_d: string
    :param file_synt: filename for synthetic seismogram
    :type file_synt: string
    :param elem: save elementary seismogram or not
    :type elem: bool
    :param cholesky: save standarized data or not
    :type cholesky: bool

    Uses :func:`numpy.save`.
    """
    # todo: option cholesky to save standarized data not tested yet
    data = self.data_shifts[self.centroid['shift_idx']]
    npts = self.npts_slice
    # elemse = read_elemse(self.nr, self.npts_elemse, 'green/elemse'+self.centroid['id']+'.dat', self.stations) #
    elemse = precalc_greens(self.event, self.centroid, self.targets, self.stations, self.comps_order,
                            store_dir=self.GF_store_dir)
    for r in range(self.nr):
        tarr2 = UTCDateTime(0) + self.t_min + self.stations[r]['arr_time']
        tlen = calc_taper_window(self.stations[r]['dist'])
        for e in range(6):
            q_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'], tarr2, tlen, self.taper_perc)
            # q_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'])
            elemse[r][e].trim(UTCDateTime(0) + self.elemse_start_origin)

        SYNT = {}
        for comp in range(3):
            SYNT[comp] = np.zeros(npts)
            for e in range(6):
                SYNT[comp] += elemse[r][e][comp].data[0:npts] * self.centroid['a'][e, 0]

    synt = np.zeros((npts, self.nr * 3))
    d = np.empty((npts, self.nr * 3))
    synt_csq = np.zeros((npts, self.nr * 3))
    d_csq = np.empty((npts, self.nr * 3))
    for r in range(self.nr):
        for comp in range(3):
            comps_used = 0
            for e in range(6):
                synt[:, 3 * r + comp] += elemse[r][e][comp].data[0:npts] * self.centroid['a'][e, 0]
            d[:, 3 * r + comp] = data[r][comp][0:npts]

            if cholesky and self.stations[r][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                if self.LT3:
                    # print(r, comp) # DEBUG
                    # d_csq2 = np.zeros(npts)
                    # synt_csq2 = np.zeros(npts)
                    x1 = -npts
                    for COMP in range(3):
                        if not self.stations[r][{0: 'useZ', 1: 'useN', 2: 'useE'}[COMP]]:
                            continue
                        x1 += npts
                        x2 = x1 + npts
                        y1 = comps_used * npts
                        y2 = y1 + npts
                        # print(self.LT3[sta][y1:y2, x1:x2].shape, data[sta][COMP].data[0:npts].shape) # DEBUG
                        # d_csq2 = np.dot(self.LT3[r][y1:y2, x1:x2], data[r][COMP].data[0:npts])
                        d_csq[:, 3 * r + comp] += np.dot(self.LT3[r][y1:y2, x1:x2], d[:, 3 * r + comp])
                        synt_csq[:, 3 * r + comp] += np.dot(self.LT3[r][y1:y2, x1:x2], synt[:, 3 * r + comp])
                else:
                    d_csq[:, 3 * r + comp] = np.dot(self.LT[r][comp], d)
                    synt_csq[:, 3 * r + comp] = np.dot(self.LT[r][comp], synt)
                comps_used += 1
    if elem:
        for e in range(6):
            elem_synt = np.zeros((npts, self.nr * 3))
            for r in range(self.nr):
                for comp in range(3):
                    # for e in range(6):
                    elem_synt[:, 3 * r + comp] += elemse[r][e][comp].data[0:npts] * self.centroid['a'][e, 0]
                    # d[:, 3 * r + comp] = data[r][comp][0:npts]
            np.save('{:s}_MT{:d}'.format(file_synt, e + 1), elem_synt)
    np.save(file_d, d)
    np.save(file_synt, synt)
    if cholesky:
        np.save(f'{file_d}_cholesky', d_csq)
        np.save(f'{file_synt}_cholesky', synt_csq)


def print_solution(self):
    """
    Write into log the best solution ``self.centroid``.
    """
    C = self.centroid
    t = self.event['t'] + C['shift']
    self.log(
        '\nCentroid location:\n  Centroid time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   Depth {d:5.1f} km'.format(
            t=t.strftime('%Y-%m-%d %H:%M:%S'), lat=C['lat'], lon=C['lon'], d=C['z'] / 1e3))
    self.log(
        '  ({0:5.0f} m to the north and {1:5.0f} m to the east with respect to epicenter)'.format(C['x'],
                                                                                                  C['y']))
    if C['edge']:
        self.log('  Warning: the solution lies on the edge of the grid!')
    C['mt2'] = mt2 = a2mt(C['a'], system='USE')
    c = max(abs(min(mt2)), max(mt2))
    C['const'] = c = 10 ** np.floor(np.log10(c))
    C['MT2'] = MT2 = mt2 / c
    if C['shift'] >= 0:
        self.log('  time: {0:5.2f} s after origin time\n'.format(C['shift']))
    else:
        self.log('  time: {0:5.2f} s before origin time\n'.format(-C['shift']))
    if C['shift'] in (self.shifts[0], self.shifts[-1]):
        self.log('  Warning: the solution lies on the edge of the time-grid!')
    self.log('  VR: {0:4.0f} %\n  CN: {1:4.0f}'.format(C['VR'] * 100, C['CN']))
    # self.log('  VR: {0:8.4f} %\n  CN: {1:4.0f}'.format(C['VR']*100, C['CN'])) # DEBUG
    self.log(
        '  MT [ Mrr    Mtt    Mpp    Mrt    Mrp    Mtp ]:\n     [{1:5.2f}  {2:5.2f}  {3:5.2f}  {4:5.2f}  '
        '{5:5.2f}  {6:5.2f}] * {0:5.0e}'.format(c, *MT2))

    return C, t  # q_add


def write_solution(self, event_id, catalog_line, st_dev, quality=None, out_file=None):

    C = self.centroid
    t = self.event['t'] + C['shift']
    kop = 'id\torigin time\tlat\tlon\tdep\ttimeshift (s)\tepic east_shift (m)\tepic north_shift (m)\t' \
          'n_station used\tn_comp used\tVR (%)\tCN\tgap\tMO (Nm)\tMw\tDC perc\tCLVD perc\tISO perc\t' \
          'Strike1\tDip1\tRake1\tStrike2\tDip2\tRake2\tMrr\tMtt\tMpp\tMrt\tMrp\tMtp\texp\t' \
          'Uncertainty\tQuality\n'
    SOLUTION = '{n:s}\t'.format(n=str(event_id), cat=catalog_line)

    try:
        SOLUTION += '{t:s}\t{lat:.3f}\t{lon:.3f}\t{d:.1f}\t{ts:.2f}\t{ys:.0f}\t{xs:.0f}\t' \
                    '{sta:d}\t{comp:d}\t{vr:.1f}\t{cn:.1f}\t{gap:.1f}\t'. \
            format(t=t.strftime('%Y-%m-%dT%H:%M:%S'), lat=C['lat'], lon=C['lon'], d=C['z'] / 1e3,
                   ts=C['shift'], xs=C['x'], ys=C['y'], sta=self.num_sta, comp=self.components, vr=C['VR'] * 100,
                   cn=C['CN'], gap=self.azimuth_gap[0])
    except (TypeError, LookupError, IOError, OSError, KeyError, ValueError):
        pass

    try:
        SOLUTION += '{{mom:5.2e}}\t{{Mw:.1f}}\t{{dc_perc:{0:s}f}}\t{{clvd_perc:{0:s}f}}\t{{iso_perc:{0:s}f}}\t' \
                    '{{s1:{0:s}f}}\t{{d1:{0:s}f}}\t{{r1:{0:s}f}}\t{{s2:{0:s}f}}\t{{d2:{0:s}f}}\t{{r2:{0:s}f}}\t'.\
            format('3.1').format(**self.mt_decomp)
    except (TypeError, LookupError, IOError, OSError, KeyError, ValueError):
        pass

    try:
        mt = np.array(a2mt(self.centroid['a'], system='USE'))
        exp = find_exponent(mt)
        SOLUTION += '{{:.2f}}\t{{:.2f}}\t{{:.2f}}\t{{:.2f}}\t{{:.2f}}\t{{:.2f}}\t{{:d}}\t'.\
            format('3.1').format(*mt/10.**exp, exp)
    except (TypeError, LookupError, IOError, OSError, KeyError, ValueError):
        pass

    try:
        SOLUTION += 'dc: {dc:.2f}, clvd: {clvd:.2f}, iso: {iso:.2f}, Mw: {Mw:.2f}, t: {t:.2f}, ' \
                    'ew: {y:.2f}, ns: {x:.2f}, z: {z:.2f}'.format(**st_dev)
        # SOLUTION += 'dc: {dc:.2f}, clvd: {clvd:.2f}, iso: {iso:.2f}, Mw: {Mw:.2f}, t: {t:.2f}, x: {x:.2f}, ' \
        #             'y: {y:.2f}, z: {z:.2f}, mom: {mom:.2f}, S: {S:.2f}, D: {D:.2f}, R: {R:.2f}'.format(**st_dev)
    except (NameError, TypeError, LookupError, IOError, OSError, KeyError, ValueError):
        SOLUTION += 'No uncertainty evaluation'

    if quality is None:
        SOLUTION += '\n'
    else:
        SOLUTION += '\t{q:s}\n'.format(q=quality)

    if out_file:
        if os.path.exists(out_file):  # write finished solution to solution catalog
            with open(out_file, 'a') as f:
                f.write(SOLUTION)
        else:
            with open(out_file, 'w') as f:
                f.write(kop)
                f.write(SOLUTION)
    else:
        return SOLUTION


def decompose_fault_planes(self, tool=''):
    """
    Decompose the moment tensor of the best grid point by :func:`decompose` and writes the result to the log.

    :param self:
    :param tool: tool for the decomposition, `mopad` for :func:`decompose_mopad`, otherwise :func:`decompose` is used
    """
    mt = a2mt(self.centroid['a'])
    if tool == 'mopad':
        self.mt_decomp = decompose_mopad(mt)
    else:
        self.mt_decomp = decompose(mt)


def print_fault_planes(self, precision='3.0'):
    """
    Decompose the moment tensor of the best grid point by :func:`decompose` and writes the result to the log.

    :param self:
    :param precision: floats precision, like ``5.1`` for 5 letters width and 1 decimal place (default ``3.0``)
    :type precision: string, optional
    """
    self.log('''\nScalar Moment: M0 = {{mom:5.2e}} Nm (Mw = {{Mw:3.1f}})
DC component: {{dc_perc:{0:s}f}} %,   CLVD component: {{clvd_perc:{0:s}f}} %,   ISOtropic component: {{iso_perc:{0:s}f}} %
Fault plane 1: strike = {{s1:{0:s}f}}, dip = {{d1:{0:s}f}}, slip-rake = {{r1:{0:s}f}}
Fault plane 2: strike = {{s2:{0:s}f}}, dip = {{d2:{0:s}f}}, slip-rake = {{r2:{0:s}f}}'''.format(precision).format(
        **self.mt_decomp))


def unused_sts_filter(self):
    for i in range(len(self.data_raw) - 1, -1, -1):
        st2 = self.data_raw[i]
        stats = st2[0].stats
        sta = stats.station
        net = stats.network
        loc = stats.location
        # sts = self.stations[i]
        sts_idx = '_'.join([net, sta, loc, stats.channel[0:2]])
        if self.stations_index[sts_idx]['useZ'] is False and self.stations_index[sts_idx]['useN'] is False and \
                self.stations_index[sts_idx]['useE'] is False:
            del self.data_raw[i]
            del self.stations[i]
            del self.data_shifts[self.centroid['shift_idx']][i]
            if self.use_cov_residual or self.use_cov_noise:
                if self.LT3:
                    del self.LT3[i]
                else:
                    del self.LT[i]
    self.create_station_index()
    targets = []
    for sts in self.stations:
        targets += [Target(quantity=self.seismogram, lat=float(sts['lat']), lon=float(sts['lon']),
                           store_id=sts['model'], tmin=self.t_min, tmax=(self.npts_elemse - 1) / self.samprate,
                           codes=(sts['network'], sts['code'], '', channel_code))
                    for channel_code in self.comps_order]
    self.targets = targets


def solution_quality_classification(self, stdev=None, quality_class_1=('A', 'B', 'C', 'D'),
                                    quality_class_2=('1', '2', '3', '4'), quality_color=('g', 'b', 'orange', 'r'),
                                    outfile=''):
    # alphabetical part
    C = self.centroid

    if C['VR'] * 100 >= 60:
        if self.num_sta >= 4:
            qi_1 = 0
        else:
            qi_1 = 1
    elif C['VR'] * 100 >= 40:
        qi_1 = 1
    elif C['VR'] * 100 >= 20:
        qi_1 = 2
    else:
        qi_1 = 3

    # numerical part
    if C['CN'] < 5:
        qi_2 = 0
    elif C['CN'] < 8:
        qi_2 = 1
    elif C['CN'] < 10:
        qi_2 = 2
    else:
        qi_2 = 3

    if stdev:
        if self.dc_interest_eq:
            unc = abs(C['clvd_perc']) / 50 + self.azimuth_gap[0] / 180 + stdev['dc'] + stdev['clvd'] + stdev['iso'] + \
                  stdev['Mw'] / 0.2 + stdev['t'] + ((stdev['x'] + stdev['y'] + stdev['z']) / 6)
        else:
            unc = 1 + self.azimuth_gap[0] / 180 + stdev['dc'] + stdev['clvd'] + stdev['iso'] + \
                  stdev['Mw'] / 0.2 + stdev['t'] + ((stdev['x'] + stdev['y'] + stdev['z']) / 6)
        if unc < 3:
            qi_2 += 0
        elif unc < 4:  # 3.5:
            qi_2 += 1
        elif unc < 5:  # 4:
            qi_2 += 2
        else:
            qi_2 += 3
    else:
        if self.dc_interest_eq:
            unc = abs(C['clvd_perc']) / 50 + self.azimuth_gap[0] / 180
        else:
            unc = 1 + self.azimuth_gap[0] / 180
        if unc < 1:
            qi_2 += 0
        elif unc < 2:
            qi_2 += 1
        elif unc < 3:
            qi_2 += 2
        else:
            qi_2 += 3

    if qi_2 > len(quality_class_2) - 1:
        qi_2 = len(quality_class_2) - 1
    if outfile:
        plt.text(0, 0, f'{quality_class_1[qi_1]}{quality_class_2[qi_2]}', ha='center', va='center',
                 fontsize=200, weight='bold', color=quality_color[qi_1])
        plt.xlim([-1, 1])
        plt.ylim([-0.9, 1])
        plt.axis('off')
        plt.savefig(outfile, bbox_inches='tight', dpi=500)
        plt.clf()
        plt.close()

    return f'{quality_class_1[qi_1]}{quality_class_2[qi_2]}'
