#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from obspy import UTCDateTime
from synt_CMT.extras import q_filter
from synt_CMT.MT_comps import decompose, a2mt
from synt_CMT.forward_problem import precalc_greens


def invert(event, grid, d_shifts, norm_d, Cd_inv, nr, comps, comps_order, stations, target_sta, npts_elemse,
           npts_slice, elemse_start_origin, gf_store_dir, deviatoric=False, decomp=True, fit_cov=True, taper_perc=None):
    """
    Solves inverse problem in a single grid point for multiple time shifts.

    :param fit_cov: calculate fitting on standarized data or original waveform even use covariance matrix
    :param event: event parameter
    :param grid: grid shifting parameter
    :param d_shifts: list of shifted data vectors :math:`d`
    :type d_shifts: list of :class:`~numpy.ndarray`
    :param norm_d: list of norms of vectors :math:`d`
    :type norm_d: list of floats
    :param Cd_inv: inverse of the data covariance matrix :math:`C_D^{-1}` saved block-by-block
    :type Cd_inv: list of :class:`~numpy.ndarray`
    :param nr: number of receivers
    :type nr: integer
    :param comps: number of components used in inversion
    :type comps: integer
    :param comps_order: order of stream components (ex: 'ZNE')
    :type comps_order: string
    :param stations: 2DO popsat
    :type stations: 2DO popsat
    :param target_sta: list of target sta to calculate GF
    :type target_sta: pyrocko gf target
    :param npts_elemse: number of points of elementary seismograms
    :type npts_elemse: integer
    :param npts_slice: number of points of seismograms used in inversion (npts_slice <= npts_elemse)
    :type npts_slice: integer
    # :param fmin: lower frequency for filtering elementary seismogram
    # :type fmin: float
    # :param fmax: higher frequency for filtering elementary seismogram
    # :type fmax: float
    :param elemse_start_origin: time between elementary seismogram start and elementary seismogram origin time
    :type elemse_start_origin: float
    :param gf_store_dir: pyrocko gf stores dir
    :param deviatoric: if ``True``, invert only deviatoric part of moment tensor (5 components),
        otherwise full moment tensor (6 components)
    :type deviatoric: bool, optional
    :param decomp: if ``True``, decomposes found moment tensor in each grid point
    :type decomp: bool, optional
    :param taper_perc: taper precentation if used tapered data in inversion
    :type taper_perc: float
    :returns: Dictionary {'shift': order of `d_shift` item, 'a': coeficients of the elementary seismograms,
        'VR': variance reduction, 'CN' condition number, and moment tensor decomposition
        (keys described at function :func:`decompose`)}

    It reads elementary seismograms for specified grid point, filter them and creates matrix :math:`G`.
    Calculates :math:`G^T`, :math:`G^T G`, :math:`(G^T G)^{-1}`, and condition number of :math:`G^T G`
    (using :func:`~np.linalg.cond`)
    Then reads shifted vectors :math:`d` and for each of them calculates :math:`G^T d` and the solution
    :math:`(G^T G)^{-1} G^T d`. Calculates variance reduction (VR) of the result.

    Finally chooses the time shift where the solution had the best VR and returns its parameters.

    Remark: because of parallelisation, this wrapper cannot be part of class :class:`synt_CMT`.
    """

    # params: grid[i]['id'], self.d_shifts, self.Cd_inv, self.nr, self.components, self.stations, self.npts_elemse,
    # self.npts_slice, self.elemse_start_origin, self.deviatoric, self.decompose
    if deviatoric:
        ne = 5
    else:
        ne = 6

    elemse = precalc_greens(event, grid, target_sta, stations, comps_order, gf_store_dir)

    for r in range(nr):
        tarr2 = UTCDateTime(0) + stations[r]['arr_time']
        tlen = calc_taper_window(stations[r]['dist'])
        for i in range(ne):
            q_filter(elemse[r][i], stations[r]['fmin'], stations[r]['fmax'], tarr2, tlen, taper_perc)
            # q_filter(elemse[r][i], stations[r]['fmin'], stations[r]['fmax'])

    if npts_slice != npts_elemse:
        dt = elemse[0][0][0].stats.delta
        for st6 in elemse:
            for st in st6:
                # st.trim(UTCDateTime(0)+dt*elemse_start_origin, UTCDateTime(0)+dt*npts_slice+dt*elemse_start_origin+1)
                st.trim(UTCDateTime(0) + elemse_start_origin)
        npts = npts_slice
    else:
        npts = npts_elemse

    # old G matrix
    # c = 0
    # G = np.empty((comps * npts, ne))
    # G2 = np.empty((comps * npts, ne))
    # for r in range(nr):
    #     for comp in range(3):
    #         if stations[r][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # this component has flag 'use in inversion'
    #             weight = stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]
    #             for i in range(npts):
    #                 for e in range(ne):
    #                     G[c * npts + i, e] = elemse[r][e][comp].data[i] * weight
    #                     G2[c * npts + i, e] = elemse2[r][e][comp].data[i] * weight
    #             c += 1
    c = 0
    G = np.empty((comps * npts, ne))
    for r in range(nr):
        for comp in range(3):
            if stations[r][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # this component has flag 'use in inversion'
                weight = stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]
                for e in range(ne):
                    G[c*npts:(c+1)*npts, e] = elemse[r][e][comp].data[0:npts] * weight
                c += 1
    # test new G matrix
    # plt.figure()
    # plt.plot(G[:, 0])
    # plt.plot(Gn[:, 0])
    # plt.xlim([0, 200])

    if Cd_inv:
        # evaluate G^T C_D^{-1}
        # G^T C_D^{-1} is in ``GtCd`` saved block-by-block, in ``Gt`` in one ndarray
        idx = 0
        GtCd = [None] * len(Cd_inv)
        # print('\nINVERT')
        for i, C in zip(range(len(Cd_inv)), Cd_inv):
            size = len(C)
            # print(G.shape, size, idx, G[idx:idx+size, : ].T.shape, C.shape) # DEBUG
            GtCd[i] = np.dot(G[idx:idx + size, :].T, C)
            idx += size
        Gt = np.concatenate(GtCd, axis=1)
    else:
        Gt = G.transpose()
    GtG = np.dot(Gt, G)
    CN = np.sqrt(np.linalg.cond(GtG))  # condition number
    GtGinv = np.linalg.inv(GtG)
    det_Ca = np.linalg.det(GtGinv)
    # print('det', det_Ca) # DEBUG

    res = {}
    sum_c = 0
    for shift in range(len(d_shifts)):
        d_shift = d_shifts[shift]
        # d : vector of shifted data
        #   shift>0 means that elemse start `shift` samples after data zero time

        # Gtd
        Gtd = np.dot(Gt, d_shift)

        # result : coeficients of elementary seismograms
        a = np.dot(GtGinv, Gtd)
        # a[0] = 1.; a[1] = 2.; a[2] = 3.; a[3] = 4.; a[4] = 5.; a[5] = 6.
        if deviatoric:
            a = np.append(a, [[0.]], axis=0)

        if Cd_inv:
            dGm = d_shift - np.dot(G, a[:ne])  # dGm = d_obs - G m
            idx = 0
            dGmCd_blocks = [None] * len(Cd_inv)
            # dGm_blocks = [None] * len(Cd_inv)
            # d_shift_blocks = [None] * len(Cd_inv)
            # dCd_blocks = [None] * len(Cd_inv)

            if fit_cov:
                for i, C in zip(range(len(Cd_inv)), Cd_inv):
                    size = len(C)
                    dGmCd_blocks[i] = np.dot(dGm[idx:idx + size, :].T, C)
                    # dCd_blocks[i] = np.dot(d_shift[idx:idx + size, :].T, C)
                    # dGm_blocks[i] = dGm[idx:idx + size, :]
                    # d_shift_blocks[i] = d_shift[idx:idx + size, :]
                    idx += size

                dGmCd = np.concatenate(dGmCd_blocks, axis=1)
                misfit = np.dot(dGmCd, dGm)[0, 0]
            else:
                synt = np.zeros(comps * npts)
                for i in range(ne):
                    synt += G[:, i] * a[i]
                misfit = np.sum((d_shift[:, 0]-synt)**2)

        else:
            synt = np.zeros(comps * npts)
            for i in range(ne):
                synt += G[:, i] * a[i]
            misfit = np.sum((d_shift[:, 0]-synt)**2)

        VR = 1 - misfit / norm_d[shift]
        res[shift] = {}
        res[shift]['a'] = a.copy()
        res[shift]['misfit'] = misfit
        res[shift]['VR'] = VR
        # res[shift]['comp_VR'] = VRs

    shift = max(res, key=lambda s: res[s]['VR'])  # best shift

    r = {'shift': shift, 'a': res[shift]['a'].copy(), 'VR': res[shift]['VR'], 'misfit': res[shift]['misfit'],
         'CN': CN, 'GtGinv': GtGinv, 'det_Ca': det_Ca, 'shifts': res}  # , 'comp_VR': res[shift]['comp_VR']}
    if decomp:
        r.update(decompose(a2mt(r['a'])))  # add MT decomposition to dict `r`

    return r


def calc_taper_window(distance, max_t_v=3000, min_length=120):
    """
    Compute len windows for tapering process

    :param distance: station distance

    :return: time length of taper window
    """
    # todo: fix taper plateu length formulation, total length taar + taper slope not exceed OT + t_after
    # return distance / 1000 * 0.2 + 25  # 0.2 * dist + 30    # 600 km 150 sec    # 100 km 50 sec
    # return distance / 1000 * 0.2 + 60
    # return distance / 1000 * 0.2 + 30
    if int(np.ceil(distance / max_t_v) + 30) > min_length:
        return int(np.ceil(distance / max_t_v) + 30)
    else:
        return int(min_length)
