#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import multiprocessing as mp
from scipy import signal, stats
from copy import deepcopy
from modules_CMT.extras import decimate, moving_average_scale, covariance_function, correlation_function
from modules_CMT.multiprocessings import covariance_matrix_prenoise_mp, covariance_matrix_residual_mp


def covariance_matrix(self, crosscovariance=False, n_toeplitz=False, r_toeplitz=False,
                      correlation=False, save_non_inverted=False, save_covariance_function=False,
                      plot_each_cov_function=True, init=False, normalize=False):
    """Creates total covariance matrix :math:`C_D` from covariance data and/or covariance theoritic"""

    CfT = []
    CdT = []
    CfD = []
    CdD = []
    n = self.npts_slice
    self.correlation = correlation

    if self.use_cov_noise:
        try:
            CfD, CdD = self.covariance_matrix_prenoise(crosscovariance=crosscovariance, toeplitz=n_toeplitz,  # TEST1
                                                       save_non_inverted=save_non_inverted, correlation=correlation,
                                                       save_covariance_function=save_covariance_function)
            self.Cf_len = self.Cfd_len
        except ValueError:
            self.use_cov_noise = False

    if self.use_cov_residual and not init:
        try:
            CfT, CdT = self.covariance_matrix_residual(crosscovariance=crosscovariance, toeplitz=r_toeplitz,
                                                       save_non_inverted=save_non_inverted, correlation=correlation,
                                                       save_covariance_function=save_covariance_function)
            self.Cf_len = self.Cft_len
        except ValueError:
            self.use_cov_residual = False

    if self.use_cov_residual and self.use_cov_noise and not init:
        # if toeplitz:
        self.log('\nCreating total covariance matrix (data + theory error)')
        # else:
        #     self.log('\nCreating total non-Toeplitz covariance matrix (data + theory error)')

#        Cd_max = 0
#        Ct_max = 0
#        if normalize:
#            for Cd, Ct in zip(CdD, CdT):
#                if Cd.max() > Cd_max:
#                    Cd_max = Cd.max()
#                if Ct.max() > Ct_max:
#                    Ct_max = Ct.max()
#            if Cd_max != 0 and Ct_max != 0:
#                if Cd_max > Ct_max:
#                    for i in range(len(CdT)):
#                        CdT[i] = CdT[i] * (Cd_max * CdT[i].max() / Ct_max) / Ct_max
#                elif Ct_max > Cd_max:
#                    for i in range(len(CdD)):
#                        CdD[i] = CdD[i] * (Ct_max * CdD[i].max() / Cd_max) / Cd_max

        if plot_each_cov_function:
            self.Cf = CfD
            self.Cd = CdD
            self.plot_covariance_function(outfile=os.path.join(self.outdir, 'covariance_function_data.png'))
            self.plot_covariance_matrix(outfile=os.path.join(self.outdir, 'covariance_matrix_data.png'),
                                        colorbar=True)
            self.Cf = CfT
            self.Cd = CdT
            self.plot_covariance_function(outfile=os.path.join(self.outdir, 'covariance_function_theoritic.png'))
            self.plot_covariance_matrix(outfile=os.path.join(self.outdir, 'covariance_matrix_theoritic.png'),
                                        colorbar=True)
        Cd_inv = []
        LT = []
        LT3 = []
        for r, Cd, Ct in zip(range(len(self.stations)), CdD, CdT):
            C = Cd + Ct
            sta = self.stations[r]
            idx = []
            if sta['useZ']:
                idx.append(0)
            if sta['useN']:
                idx.append(1)
            if sta['useE']:
                idx.append(2)

            if crosscovariance and len(C):
                try:
                    C_inv = np.linalg.inv(C)
                    # C_inv = np.linalg.pinv(C)
                    Cd_inv.append(C_inv)
                    LT3.append(np.linalg.cholesky(C_inv).T)
                except:
                    w, v = np.linalg.eig(C)
                    print('Minimal eigenvalue C[{0:1d}]: {1:6.1e}, clipping'.format(r, min(w)))
                    w = w.real.clip(min=0)  # set non-zero eigenvalues to zero and
                    # remove complex part (numerc artefacts)
                    # mx = max(w)
                    # w = w.real.clip(min=w*1e-18) # set non-zero eigenvalues to almost-zero and
                    # remove complex part (both are numerical artefacts)
                    v = v.real  # remove complex part of eigenvectors
                    C = v.dot(np.diag(w)).dot(v.T)
                    # C = nearcorr(C)
                    C_inv = np.linalg.inv(C)
                    # C_inv = np.linalg.pinv(C)
                    w, v = np.linalg.eig(C_inv)  # DEBUG
                    if min(w) < 0:
                        print('Minimal eigenvalue C_inv: {1:6.1e}, CLIPPING'.format(r, min(w)))
                        w = w.real.clip(min=0)  # DEBUG
                        v = v.real  # DEBUG
                        C_inv = v.dot(np.diag(w)).dot(v.T)
                    Cd_inv.append(C_inv)
                    LT.append([1, 1, 1])
                    LT3.append(np.diag(np.sqrt(w)).dot(v.T))
            elif crosscovariance:  # C is zero-size matrix
                Cd_inv.append(C)
                LT.append([1, 1, 1])
                LT3.append(1)
            else:
                C_inv = np.linalg.inv(C)
                # C_inv = np.linalg.pinv(C)
                Cd_inv.append(C_inv)
                LT.append([1, 1, 1])
                for i in idx:
                    I = idx.index(i) * n
                    try:
                        LT[-1][i] = np.linalg.cholesky(C_inv[I:I + n, I:I + n]).T
                    except:
                        # w,v = np.linalg.eig(C[I:I+n, I:I+n])
                        # mx = max(w)
                        # print ('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, clipping'.format(r,i,min(w)))
                        # w = w.real.clip(min=0)
                        # v = v.real
                        # C[I:I+n, I:I+n] = v.dot(np.diag(w)).dot(v.T)
                        # C_inv[I:I+n, I:I+n] = np.linalg.inv(C[I:I+n, I:I+n])
                        w, v = np.linalg.eig(C_inv[I:I + n, I:I + n])
                        print('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, CLIPPING'.format(r, i, min(w)))
                        w = w.real.clip(min=0)
                        v = v.real
                        LT[-1][i] = np.diag(np.sqrt(w)).dot(v.T)

        self.Cf = deepcopy(CfD)
        for i in range(len(CfD)):
            for j in range(len(CfD[i])):
                for k in range(len(CfD[i][j])):
                    if CfD[i][j][k] is not None:
                        len_CfD = len(CfD[i][j][k])
                        if CfT[i][j][k] is not None:
                            len_CfT = len(CfT[i][j][k])
                        else:
                            self.Cf[i][j][k] = CfD[i][j][k]
                            self.Cf_len = len_CfD
                            break
                        if len_CfD > len_CfT:
                            CfT_pad = np.zeros(CfD[i][j][k].shape)
                            idx_start = int((len_CfD - len_CfT)/2)
                            idx_stop = idx_start + len_CfT
                            CfT_pad[idx_start:idx_stop, ] = CfT[i][j][k]
                            self.Cf[i][j][k] = (CfD[i][j][k] + CfT_pad) / max(CfD[i][j][k] + CfT_pad)
                            self.Cf_len = len_CfD
                        elif len_CfT > len_CfD:
                            CfD_pad = np.zeros(CfT[i][j][k].shape)
                            idx_start = int((len_CfT - len_CfD)/2)
                            idx_stop = idx_start + len_CfD
                            CfD_pad[idx_start:idx_stop, ] = CfD[i][j][k]
                            self.Cf[i][j][k] = (CfT[i][j][k] + CfD_pad) / max(CfT[i][j][k] + CfD_pad)
                            self.Cf_len = len_CfT
                        else:
                            self.Cf[i][j][k] = (CfD[i][j][k] + CfT[i][j][k]) / max(CfD[i][j][k] + CfT[i][j][k])
                    else:
                        if CfT[i][j][k] is not None:
                            len_CfT = len(CfT[i][j][k])
                            self.Cf[i][j][k] = CfT[i][j][k]
                            self.Cf_len = len_CfT

        # import matplotlib as mpl
        # mpl.use('Qt5Agg')
        # from matplotlib import pyplot as plt
        # plt.plot(CfT[i][j][k])

        self.Cd = CdD
        for Cdd, Cdt in zip(self.Cd, CdT):
            Cdd += Cdt
        self.Cd_inv = Cd_inv
        self.LT = LT
        self.LT3 = LT3

    elif self.use_cov_residual and not init:  # test5

        Cd_inv = []
        LT = []
        for r, C in zip(range(len(self.stations)), CdT):
            sta = self.stations[r]
            idx = []
            if sta['useZ']:
                idx.append(0)
            if sta['useN']:
                idx.append(1)
            if sta['useE']:
                idx.append(2)
            # Inversion of matrix C
            C_inv = np.linalg.inv(C)
            # C_inv = np.linalg.pinv(C)
            Cd_inv.append(C_inv)
            LT.append([1, 1, 1])
            for i in idx:
                I = idx.index(i) * n
                try:
                    LT[-1][i] = np.linalg.cholesky(C_inv[I:I + n, I:I + n]).T
                except:
                    w, v = np.linalg.eig(C_inv[I:I + n, I:I + n])
                    print('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, CLIPPING'.format(r, i, min(w)))
                    w = w.real.clip(min=0)
                    v = v.real
                    LT[-1][i] = np.diag(np.sqrt(w)).dot(v.T)
        self.Cf = CfT
        self.Cd = CdT
        self.Cd_inv = Cd_inv
        self.LT = LT

    elif self.use_cov_noise:

        Cd_inv = []
        LT = []
        LT3 = []
        for r, C in zip(range(len(self.stations)), CdD):
            sta = self.stations[r]
            idx = []
            if sta['useZ']:
                idx.append(0)
            if sta['useN']:
                idx.append(1)
            if sta['useE']:
                idx.append(2)

            if crosscovariance and len(C):
                try:
                    C_inv = np.linalg.inv(C)
                    # C_inv = np.linalg.pinv(C)
                    Cd_inv.append(C_inv)
                    LT3.append(np.linalg.cholesky(C_inv).T)
                except np.linalg.LinAlgError:
                    w, v = np.linalg.eig(C)
                    print('Minimal eigenvalue C[{0:1d}]: {1:6.1e}, clipping'.format(r, min(w)))
                    w = w.real.clip(min=0)  # set non-zero eigenvalues to zero and
                    v = v.real  # remove complex part of eigenvectors
                    C = v.dot(np.diag(w)).dot(v.T)
                    # C = nearcorr(C)
                    C_inv = np.linalg.inv(C)
                    # C_inv = np.linalg.pinv(C)
                    w, v = np.linalg.eig(C_inv)  # DEBUG
                    if min(w) < 0:
                        print('Minimal eigenvalue C_inv: {1:6.1e}, CLIPPING'.format(r, min(w)))
                        w = w.real.clip(min=0)  # DEBUG
                        v = v.real  # DEBUG
                        C_inv = v.dot(np.diag(w)).dot(v.T)
                    Cd_inv.append(C_inv)
                    LT.append([1, 1, 1])
                    LT3.append(np.diag(np.sqrt(w)).dot(v.T))
            elif crosscovariance:  # C is zero-size matrix
                Cd_inv.append(C)
                LT.append([1, 1, 1])
                LT3.append(1)
            else:
                C_inv = np.linalg.inv(C)
                # C_inv = np.linalg.pinv(C)
                Cd_inv.append(C_inv)
                LT.append([1, 1, 1])
                for i in idx:
                    I = idx.index(i) * n
                    try:
                        LT[-1][i] = np.linalg.cholesky(C_inv[I:I + n, I:I + n]).T
                    except:
                        # w,v = np.linalg.eig(C[I:I+n, I:I+n])
                        # mx = max(w)
                        # print ('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, clipping'.format(r,i,min(w)))
                        # w = w.real.clip(min=0)
                        # v = v.real
                        # C[I:I+n, I:I+n] = v.dot(np.diag(w)).dot(v.T)
                        # C_inv[I:I+n, I:I+n] = np.linalg.inv(C[I:I+n, I:I+n])
                        w, v = np.linalg.eig(C_inv[I:I + n, I:I + n])
                        print('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, CLIPPING'.format(r, i, min(w)))
                        w = w.real.clip(min=0)
                        v = v.real
                        LT[-1][i] = np.diag(np.sqrt(w)).dot(v.T)

        self.Cf = CfD
        self.Cd = CdD
        self.Cd_inv = Cd_inv
        self.LT = LT
        self.LT3 = LT3
    else:
        return


def covariance_matrix_prenoise(self, crosscovariance=False, toeplitz=True, correlation=False, save_non_inverted=False,
                               save_covariance_function=False):
    """
    Creates covariance matrix :math:`C_d` from ``self.noise``.
    Modified from BayesISOLA https://geo.mff.cuni.cz/~vackar/BayesISOLA/ https://github.com/vackar/BayesISOLA

    :param self
    :param crosscovariance: Set ``True`` to calculate crosscovariance between components. If ``False``,
        it assumes that noise at components is not correlated, so non-diagonal blocks are identically zero.
    :type crosscovariance: bool, optional
    :param toeplitz: If ``False``, scalling covariance function following dettmer, 2007.
    :type toeplitz: bool, optional
    :param correlation: If ``True``, using correlation function else using covariance function following dettmer, 2007.
    :type correlation: bool
    :param save_non_inverted: If ``True``, save also non-inverted matrix, which can be plotted later.
    :type save_non_inverted: bool, optional
    :param save_covariance_function: If ``True``, save also the covariance function matrix,
        which can be plotted later.
    :type save_covariance_function: bool, optional
    """
    if correlation:
        if toeplitz:
            if crosscovariance:
                self.log('\nCreating Toeplitz cross-correlation matrix from pre-event noise')
            else:
                self.log('\nCreating Toeplitz correlation matrix from pre-event noise')
        else:
            if crosscovariance:
                self.log('\nCreating non-Toeplitz cross-correlation matrix from pre-event noise')
            else:
                self.log('\nCreating non-Toeplitz correlation matrix from pre-event noise')
    else:
        if toeplitz:
            if crosscovariance:
                self.log('\nCreating Toeplitz cross-covariance matrix from pre-event noise')
            else:
                self.log('\nCreating Toeplitz covariance matrix from pre-event noise')
        else:
            if crosscovariance:
                self.log('\nCreating non-Toeplitz cross-covariance matrix from pre-event noise')
            else:
                self.log('\nCreating non-Toeplitz covariance matrix from pre-event noise')
    if not self.noise:
        self.log('No noise slice to generate covariance matrix. Some of records probably too short or noise slices '
                 'not generated [param noise_slice at func trim_filter_data()]. Exiting...', printcopy=True)
        raise ValueError('No noise slice to generate covariance matrix.')

    Cf = []
    Cd = []

    if self.threads > 1:
        pool = mp.Pool(processes=self.threads)
        results = [pool.apply_async(covariance_matrix_prenoise_mp,
                                    args=(i, self.stations, self.npts_slice, self.noise[i], save_covariance_function,
                                          crosscovariance, toeplitz, correlation))
                   for i in range(len(self.stations))]
        output = [p.get() for p in results]
    else:
        output = []
        for i in range(len(self.stations)):
            result = covariance_matrix_prenoise_mp(i, self.stations, self.npts_slice, self.noise[i],
                                                   save_covariance_function=save_covariance_function,
                                                   crosscovariance=crosscovariance, toeplitz=toeplitz,
                                                   correlation=correlation)
            output.append(result)

    sts_cov = sorted(output, key=lambda d: d['dist'])

    lst_cflen = []
    for cov in sts_cov:
        if 'Cf_len' in cov:
            lst_cflen.append(cov['Cf_len'])
        if save_non_inverted:
            Cf.append(cov['Cf'][0])
        Cd.append(cov['C'])

    self.Cfd_len = stats.mode(lst_cflen)[0]

    return Cf, Cd


def covariance_matrix_residual(self, crosscovariance=False, toeplitz=False, correlation=False, save_non_inverted=False,
                               save_covariance_function=False):
    """
    Creates covariance matrix :math:`C_t` from ``residual between synthetic seismogram each velocity model``.

    :param self
    :type
    :param crosscovariance: Set ``True`` to calculate crosscovariance between components. If ``False``,
        it assumes that residual error at components is not correlated, so non-diagonal blocks are identically zero.
    :type crosscovariance: bool, optional
    :param toeplitz: If ``False``, scalling covariance function following dettmer, 2007.
    :type toeplitz: bool, optional
    :param correlation: If ``True``, using correlation function else using covariance function following dettmer, 2007.
    :type correlation: bool
    :param save_non_inverted: If ``True``, save also non-inverted matrix, which can be plotted later.
    :type save_non_inverted: bool, optional
    :param save_covariance_function: If ``True``, save also the covariance function matrix,
        which can be plotted later.
    :type save_covariance_function: bool, optional

    Author: Dettmer 2007

    """
    if correlation:
        if toeplitz:
            if crosscovariance:
                self.log('\nCreating Toeplitz cross-correlation matrix from initial inversion residual')
            else:
                self.log('\nCreating Toeplitz correlation matrix from initial inversion residual')
        else:
            if crosscovariance:
                self.log('\nCreating non-Toeplitz cross-correlation matrix from initial inversion residual')
            else:
                self.log('\nCreating non-Toeplitz correlation matrix from initial inversion residual')
    else:
        if toeplitz:
            if crosscovariance:
                self.log('\nCreating Toeplitz cross-covariance matrix from initial inversion residual')
            else:
                self.log('\nCreating Toeplitz covariance matrix from initial inversion residual')
        else:
            if crosscovariance:
                self.log('\nCreating non-Toeplitz cross-covariance matrix from initial inversion residual')
            else:
                self.log('\nCreating non-Toeplitz covariance matrix from initial inversion residual')
    if 'res_Z' not in self.stations[0]:
        self.log('No residual data to generate covariance matrix. Exiting...', printcopy=True)
        raise ValueError('No residual data to generate covariance matrix.')

    Cf = []
    Cd = []

    if self.threads > 1:
        pool = mp.Pool(processes=self.threads)
        results = [pool.apply_async(covariance_matrix_residual_mp,
                                    args=(i, self.stations, self.npts_slice, save_covariance_function, crosscovariance,
                                          toeplitz, correlation))
                   for i in range(len(self.stations))]
        output = [p.get() for p in results]
    else:
        output = []
        for i in range(len(self.stations)):
            result = covariance_matrix_residual_mp(i, self.stations, self.npts_slice,
                                                   save_covariance_function=save_covariance_function,
                                                   crosscovariance=crosscovariance,toeplitz=toeplitz,
                                                   correlation=correlation)
            output.append(result)

    sts_cov = sorted(output, key=lambda d: d['dist'])

    lst_cflen = []
    for cov in sts_cov:
        if 'Cf_len' in cov:
            lst_cflen.append(cov['Cf_len'])
        if save_non_inverted:
            Cf.append(cov['Cf'][0])
        Cd.append(cov['C'])

    self.Cft_len = stats.mode(lst_cflen)[0]
    return Cf, Cd
