#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pygmt
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # for test_module
# mpl.use('Qt5Agg')  # for test_module
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from os.path import join, exists
from math import sin, cos, radians
from matplotlib.lines import Line2D
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta as td
from obspy import UTCDateTime
from obspy.core import stream
from obspy.imaging.beachball import beach
from obspy.imaging.mopad_wrapper import beach as beach2
from scipy import signal, interpolate
from synt_CMT.MT_comps import find_exponent, a2mt, decompose
from synt_CMT.inverse_problem import precalc_greens, calc_taper_window
from synt_CMT.extras import histogram, q_filter, taper_wdw, format_timedelta, align_yaxis


def plot_gmt(self, inputdir, outfile, quality, add_waveform=False, time_zone="UTC", plot_DEV=False):

    dic_tz = {'UTC': 0, 'WIB': 7, 'WITA': 8, 'WIT': 9}

    fig = pygmt.Figure()

    C = self.centroid
    E = self.event
    fp = C['faultplanes']
    lat = C['lat']
    lon = C['lon']
    dep = C['z'] / 1e3
    mag = C['Mw']
    time = E['t'] + C['shift']

    use   = self.used_a
    p_use = self.used_p
    unuse = self.used_n
    out_dir = os.path.dirname(outfile)
    dict_q  = {'A': 'blue', 'B': 'green', 'C': 'gray', 'D': 'red'}

    # margin_lat = 3
    # margin_lon = 4
    # x_rose =
    # y_rose =
    # x_scale =
    # y_scale =

    margin_lat = 3.75
    margin_lon = 5
    region     = [lon - margin_lon, lon + margin_lon, lat - margin_lat, lat + margin_lat]
    rectangle  = [[region[0], region[2], region[1], region[3]]]

    title_fit1 = ''
    title_fit2 = ''

    fig.image(imagefile=join(inputdir, 'w_bg.jpg'), position="x0/0+w23c/22c", box=False)

    if add_waveform and exists(join(out_dir, 'fit_true_waveform.png')) or \
        add_waveform and exists(join(out_dir, 'fit_cova_waveform.png')):
        fig.coast(region=region, projection="M16c", land="245/245/200", water="120/210/255",
                  shorelines=True, resolution="f", frame=["WSne", "af2"], xshift=1.25, yshift=1.65)
    else:
        fig.coast(region=region, projection="M16c", land="245/245/200", water="120/210/255",
                  shorelines=True, resolution="f", frame=["WSne", "af2"], xshift=3.65, yshift=1.65)
    # pygmt.config(MAP_FRAME_PEN="blue", MAP_TICK_PEN="blue", FONT_ANNOT_PRIMARY="blue", FONT_LABEL="8p,Helvetica,black", )
    pygmt.config(FONT_ANNOT_PRIMARY="10p,Helvetica,black")
    with pygmt.config(MAP_FRAME_TYPE="plain"):
        fig.basemap(rose=f"g{region[0]+0.9}/{region[3]-1.5}+f2+w2c+jCB",
                    map_scale=f"g{region[0]+0.9}/{region[3]-1.6}+w150k+ar+c0+jCT+lkm")

    if not unuse.empty:
        fig.plot(x=unuse.lon, y=unuse.lat, style="i0.5c", color="black", pen="black", label='"Stations Not Used"')
        fig.text(x=unuse.lon, y=unuse.lat-0.15, text=unuse.code, justify="CM")
    if not p_use.empty:
        fig.plot(x=p_use.lon, y=p_use.lat, style="i0.5c", color="gray", pen="black", label='"Some Comp Used"')
        fig.text(x=p_use.lon, y=p_use.lat-0.15, text=p_use.code, justify="CM")
    if not use.empty:
        fig.plot(x=use.lon, y=use.lat, style="i0.5c", color="red", pen="black", label='"Stations Used"')
        fig.text(x=use.lon, y=use.lat-0.15, text=use.code, justify="CM")

    fig.legend()

    if plot_DEV:
        mt = np.array(a2mt(C['a'], 'USE')) * 10 ** 7
        exp = find_exponent(mt)
        mt = mt / 10. ** exp
        DEV = dict(mrr=mt[0], mtt=mt[1], mff=mt[2], mrt=mt[3], mrf=mt[4], mtf=mt[5], exponent=exp)
        fig.meca(DEV, component="deviatoric", scale="2c", longitude=lon + 0.5, latitude=lat + 1.5, depth=dep)  ##
    else:
        DC = dict(strike=C['s1'], dip=C['d1'], rake=C['r1'], magnitude=mag)
        fig.meca(DC, scale="2c", longitude=lon, latitude=lat, depth=dep,  G=dict_q[quality])

    with fig.inset(position=f"g{region[0]}/{region[2]}+w6c/2.4", box="+pblack"):
        fig.coast(region=[90, 144, -12, 10], projection="M6c", land="245/245/230", water="120/240/255",
                  borders=[1, 2], shorelines="0.1", resolution="h")
        fig.plot(data=rectangle, style="r+s", pen="1.5p,red")

    fig.image(imagefile=join(inputdir, 'wm_QCMT.png'), position="x14.5/0.4+w1.8c+jCB", box=False)

    if add_waveform:
        if exists(join(out_dir, 'fit_true_waveform.png')) and exists(join(out_dir, 'fit_cova_waveform.png')):
            fig.image(imagefile=join(out_dir, 'fit_true_waveform.png'), position="x16.4/11.84+w4.8c+jLT", box=True)
            fig.image(imagefile=join(out_dir, 'fit_cova_waveform.png'), position="x16.4/6+w4.8c+jLT", box=True)
            title_fit1 = 'Fitting true waveform'
            title_fit2 = 'Fitting standardized data'
        elif exists(join(out_dir, 'fit_true_waveform.png')):
            fig.image(imagefile=join(out_dir, 'fit_true_waveform.png'), position="x16.4/11.84+w4.8c+jLT", box=True)
            title_fit1 = 'Fitting true waveform'
        elif exists(join(out_dir, 'fit_cova_waveform.png')):
            fig.image(imagefile=join(out_dir, 'fit_cova_waveform.png'), position="x16.4/11.84+w4.8c+jLT", box=True)
            title_fit1 = 'Fitting standardized data'

    title_text1a   = f"Automatic Regional Moment Tensor"
    title_text1b   = f"Ambon Geophysical Station - BMKG"
    title_text2a  = f"Event {E['id']}, {E['remark']}:"
    title_text2b  = f"   Origin Time"
    title_text2bt = f": {(E['t'] + td(hours=dic_tz[time_zone])).strftime('%d %b %Y %H:%M:%S')} {time_zone}"
    title_text2c  = f"   Epicenter"
    title_text2ct = f": {E['lat']:.2f}, {E['lon']:.2f}"
    title_text2d  = f"   Depth"
    title_text2dt = f": {E['depth'] / 1e3:.0f} km"
    title_text2e  = f"   Magnitude"
    title_text2et = f": {E['mag']:.1f}"
    title_text3a  = f"Moment tensor solution:"
    title_text3b  = f"   Centroid Time"
    title_text3bt = f": {(time + td(hours=dic_tz[time_zone])).strftime('%d %b %Y %H:%M:%S')} {time_zone}"
    title_text3c  = f"   Centroid Loc"
    title_text3ct = f": {lat:.2f}, {lon:.2f}"
    title_text3d = f"   Centroid Depth"
    title_text3dt = f": {dep:.0f} km"
    title_text3e = f"   Magnitude (Mw)"
    title_text3et = f": {mag:.1f}"
    title_text3f = f'Solution Quality: "{quality}"'
    # title_text3ft1 = f':'
    # title_text3ft = f'"{quality}"'
    title_text4 = f"Fault Plane 1: strike {fp[0][0]:.0f}, dip {fp[0][1]:.0f}, rake {fp[0][2]:.0f}"
    title_text5 = f"Fault Plane 2: strike {fp[1][0]:.0f}, dip {fp[1][1]:.0f}, rake {fp[1][2]:.0f}"

    if add_waveform and exists(join(out_dir, 'fit_true_waveform.png')) or \
            add_waveform and exists(join(out_dir, 'fit_cova_waveform.png')):
        x_shift = 10.3
        x_shiftb = 12.5
        x_shift2 = 2.2
        x_shift3 = 3
        x_shift4 = 0.2
    else:
        x_shift = 7.9
        x_shiftb = 7.9
        x_shift2 = 2.2
        x_shift3 = 3
        x_shift4 = 0.2
    fig.text(x=region[0], y=region[3]+3.17, text=title_text2a, justify="LB", no_clip=True,
             font="12p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.8, text=title_text2b, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.45, text=title_text2c, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.1, text=title_text2d, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+1.75, text=title_text2e, justify="LB", no_clip=True,
             font="10p,Helvetica,black")

    fig.text(x=region[0], y=region[3]+2.8, text=title_text2bt, justify="LB", no_clip=True,
             font="10p,Helvetica,black", xshift=x_shift2)
    fig.text(x=region[0], y=region[3]+2.45, text=title_text2ct, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.1, text=title_text2dt, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+1.75, text=title_text2et, justify="LB", no_clip=True,
             font="10p,Helvetica,black")

    fig.text(x=region[0], y=region[3]+3.17, text=title_text3a, justify="LB", no_clip=True, xshift=x_shiftb-x_shift2,
             font="12p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.8, text=title_text3b, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.45, text=title_text3c, justify="LB", no_clip=True,
                 font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.1, text=title_text3d, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+1.75, text=title_text3e, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    # fig.text(x=region[0], y=region[3]+0.85, text=title_text3f, justify="LB", no_clip=True,
    #          font="10p,Helvetica,black")

    fig.text(x=region[0], y=region[3]+2.8, text=title_text3bt, justify="LB", no_clip=True, xshift=x_shift3,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.45, text=title_text3ct, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+2.1, text=title_text3dt, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    fig.text(x=region[0], y=region[3]+1.75, text=title_text3et, justify="LB", no_clip=True,
             font="10p,Helvetica,black")
    # fig.text(x=region[0], y=region[3]+1.05, text=title_text3ft1, justify="LB", no_clip=True,
    #          font="10p,Helvetica,black")
    # fig.text(x=region[0], y=region[3]+1.05, text=title_text3ft, justify="LB", no_clip=True,
    #          font=f"10p,Helvetica-Bold,{dict_q[quality]}", xshift=x_shift4)

    fig.text(x=region[0], y=region[3]+1.15, text=title_text3f, justify="CB", no_clip=True,
             xshift=-x_shift3-x_shift4-x_shiftb+x_shift,
             font=f"11.5p,Helvetica-Bold,{dict_q[quality]}")
    fig.text(x=region[0], y=region[3]+0.72, text=title_text4, justify="CB", no_clip=True,
             font="11p,Helvetica-Oblique,black")
    fig.text(x=region[0], y=region[3]+0.35, text=title_text5, justify="CB", no_clip=True,
             font="11p,Helvetica-Oblique,black")
    fig.text(x=region[0], y=region[3]+3.72, text=title_text1b, justify="CB", no_clip=True,
             font="14p,Helvetica-Bold,red")
    fig.text(x=region[0], y=region[3]+4.12, text=title_text1a, justify="CB", no_clip=True,
             font="14p,Helvetica-Bold,red")

    if title_fit1:
        fig.text(x=region[1], y=region[3]+0.08, text=title_fit1, justify="CB", no_clip=True,
                 font="9p,Helvetica-Oblique,black", xshift=2.6-x_shift)

    if title_fit2:
        fig.text(x=region[1], y=region[3] - margin_lat+0.12, text=title_fit2, justify="CB", no_clip=True,
                 font="9p,Helvetica-Oblique,black")

    fig.savefig(outfile, dpi=300)


@staticmethod
def plot_snr(st, outfile='', distance=5e3, ylabel='', arr_t=None, xmin=100, xmax=300, legend=False,
             title="{net:s}:{sta:s}{ch:s}, {date:s} {time:s}"):
    if type(st) == stream.Stream:
        npts = st[0].stats.npts
        shifts = {st[0].stats.channel[2]: 0, st[1].stats.channel[2]: distance, st[2].stats.channel[2]: -distance}
        samprate = st[0].stats.sampling_rate
    elif type(st) == stream.Trace:
        tr = st
        npts = tr.stats.npts
        samprate = tr.stats.sampling_rate
    t = np.arange(0, (npts-0.5) / samprate, 1 / samprate)
    colors = {'N': 'r', 'E': 'g', 'Z': 'b'}
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(12, 6))
    if type(st) == stream.Stream:
        for tr in st:
            ch = tr.stats.channel[2]
            plt.plot(t, tr.data + shifts[ch], colors[ch], label={'Z': 'raw record', 'N': '', 'E': ''}[ch])
            if arr_t:
                plt.plot(arr_t, 0, '|k', markersize=300)
            plt.text(xmin+(xmax-xmin)*0.05, shifts[ch]+distance*0.2, ch, color=colors[ch])
        plt.title(title.format(net=st[0].stats.network, sta=st[0].stats.station,
                               date=st[0].stats.starttime.strftime("%Y-%m-%d"),
                               time=st[0].stats.starttime.strftime("%H:%M:%S"), ch=''))
    elif type(st) == stream.Trace:
        ch = tr.stats.channel[2]
        plt.plot(t, tr.data, colors[ch], label='raw record')
        if arr_t:
            plt.plot(arr_t, 0, '|k', markersize=300)
        plt.title(title.format(net=tr.stats.network, sta=tr.stats.station,
                               date=tr.stats.starttime.strftime("%Y-%m-%d"),
                               time=tr.stats.starttime.strftime("%H:%M:%S"), ch=' (channel '+ch+')'))
    plt.xlim(xmin, xmax)
    if legend:
        plt.legend(loc='upper right')
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel('time [s]')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()


def plot_MT(self, outfile='output/centroid.png', facecolor='red'):
    """
    Plot the beachball of the best solution ``self.centroid``.

    :param outfile: path to the file where to plot; if ``None``, plot to the screen
    :type outfile: string, optional
    :param facecolor: color of the colored quadrants/parts of the beachball
    """
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    lw = 2
    plt.xlim(-100 - lw / 2, 100 + lw / 2)
    plt.ylim(-100 - lw / 2, 100 + lw / 2)

    a = self.centroid['a']
    mt2 = a2mt(a, system='USE')
    # beachball(mt2, outfile=outfile)
    full = beach(mt2, linewidth=lw, facecolor=facecolor, edgecolor='black', zorder=1)
    ax.add_collection(full)
    if self.decompose:
        dc = beach((self.centroid['s1'], self.centroid['d1'], self.centroid['r1']), nofill=True,
                   linewidth=lw / 2, zorder=2)
        ax.add_collection(dc)
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()
    # mt = [-a[3,0]+a[5,0], -a[4,0]+a[5,0], a[3,0]+a[4,0]+a[5,0], a[0,0], a[1,0], -a[2,0]]
    # [M11, M22, M33, M12, M13, M23] in NEZ system
    # beachball(mt, mopad_basis='NED')
    # mt = [mt[2], mt[0], mt[1], mt[4], -mt[5], -mt[3]]
    # beachball(mt)
    # beachball2(mt)


def plot_uncertainty(self, outfile='output/uncertainty.png', n=200, reference=None, best=True, fontsize=None,
                     save_ppdf=False):
    """
    Generates random realizations based on the calculated solution and its uncertainty and plots these mechanisms
     and histograms of its parameters.

    :param outfile: Path to the file where to plot. If ``None``, plot to the screen. Because more figures are
        plotted, inserts an identifier before the last dot (`uncertainty.png` -> `uncertainty_MT.png`, `
        intertanty_time-shift.png`, etc.).
    :type outfile: string, optional
    :param n: number of realizations
    :type n: integer, optional
    :param reference: plot a given reference solution too; the value should be length 6 array of moment tensor
        in 'NEZ' coordinates or a moment tensor decomposition produced by :func:`decompose`
    :type reference: array or dictionary
    :param best: show the best solutions together too
    :type best: boolean, optional
    :param fontsize: fontsize for histogram annotations
    :type fontsize: scalar, optional
    :param save_ppdf: save uncertainty ppdf to file or not
    :type save_ppdf: boolean
    """

    # Generate mechanisms
    shift = []
    depth = []
    NS = []
    EW = []
    n_sum = 0
    A = []
    c = self.centroid
    for gp in self.grid:
        if gp['err']:
            continue
        for i in gp['shifts']:
            GP = gp['shifts'][i]
            n_GP = int(round(GP['c'] / self.sum_c * n))
            if n_GP == 0:
                continue
            n_sum += n_GP
            a = GP['a']
            if self.deviatoric:
                a = a[:5]
            cov = gp['GtGinv']
            A2 = np.random.multivariate_normal(a.T[0], cov, n_GP)
            for a in A2:
                a = a[np.newaxis].T
                if self.deviatoric:
                    a = np.append(a, [[0.]], axis=0)
                A.append(a)
            shift += [self.shifts[i]] * n_GP
            depth += [gp['z'] / 1e3] * n_GP
            NS += [gp['x'] / 1e3] * n_GP
            EW += [gp['y'] / 1e3] * n_GP
    if n_sum <= 1:
        self.log('\nUncertainty evaluation: nothing plotted. Posterior probability density function too wide or '
                 'prefered number of mechanism ({0:d}) too low.'.format(n))
        return None
    # Process mechanisms
    dc_perc = []
    clvd_perc = []
    iso_perc = []
    moment = []
    Mw = []
    strike = []
    dip = []
    rake = []
    # strike1 = []
    # dip1 = []
    # rake1 = []
    # strike2 = []
    # dip2 = []
    # rake2 = []
    for a in A:
        mt = a2mt(a)
        MT = decompose(mt)
        dc_perc.append(MT['dc_perc'])
        clvd_perc.append(MT['clvd_perc'])
        iso_perc.append(MT['iso_perc'])
        moment.append(MT['mom'])
        Mw.append(MT['Mw'])
        strike += [MT['s1'], MT['s2']]
        dip += [MT['d1'], MT['d2']]
        rake += [MT['r1'], MT['r2']]
        # strike1.append(min(MT['s1'], MT['s2']))
        # dip1.append(min(MT['d1'], MT['d2']))
        # rake1.append(min(MT['r1'], MT['r2']))
        # strike2.append(max(MT['s1'], MT['s2']))
        # dip2.append(max(MT['d1'], MT['d2']))
        # rake2.append(max(MT['r1'], MT['r2']))

    # Compute standard deviation
    stdev = {'dc': np.std(dc_perc) / 100, 'clvd': np.std(clvd_perc) / 100, 'iso': np.std(iso_perc) / 100,
             'Mw': np.std(Mw), 't': np.std(shift), 'x': np.std(EW), 'y': np.std(NS), 'z': np.std(depth)}
             # 'Mw': np.std(Mw) / 0.2, 't': np.std(shift), 'x': np.std(NS), 'y': np.std(EW), 'z': np.std(depth)}
    # stdev = {'dc': np.std(dc_perc) / 100, 'clvd': np.std(clvd_perc) / 100, 'iso': np.std(iso_perc) / 100,
    #          'Mw': np.std(Mw) / 0.2, 't': np.std(shift), 'x': np.std(NS), 'y': np.std(EW), 'z': np.std(depth),
    #          'mom': np.std(moment), 'S': (np.std(strike1) + np.std(strike2)) / 2,
    #          'D': (np.std(dip1) + np.std(dip2)) / 2, 'R': (np.std(rake1) + np.std(rake2)) / 2}
    # todo: fixed fault geometry uncertainty

    # Plot centroid uncertainty
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    lw = 0.5
    plt.xlim(-100 - lw / 2, 100 + lw / 2)
    plt.ylim(-100 - lw / 2, 100 + lw / 2)
    for a in A:
        mt2 = a2mt(a, system='USE')
        try:
            full = beach(mt2, linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
            ax.add_collection(full)
        except:
            try:
                full = beach2(mt2, linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
                ax.add_collection(full)
            except:
                print('plotting this moment tensor failed: ', mt2)
    if best:
        mt2 = a2mt(c['a'], system='USE')
        full = beach(mt2, linewidth=lw * 3, nofill=True, edgecolor=(1., 0, 0))
        ax.add_collection(full)
    if reference and len(reference) == 6:
        ref = decompose(reference)
        mt2 = (reference[2], reference[0], reference[1], reference[4], -reference[5], -reference[3])
        full = beach(mt2, linewidth=lw * 3, nofill=True, edgecolor='red')
        ax.add_collection(full)
    elif reference:
        ref = reference
        if 'mom' in ref and not 'Mw' in ref:
            ref['Mw'] = 2. / 3. * np.log10(ref['mom']) - 18.1 / 3.
        elif 'Mw' in ref and not 'mom' in ref:
            ref['mom'] = 10 ** ((ref['Mw'] + 18.1 / 3.) * 1.5)
    else:
        ref = {'dc_perc': None, 'clvd_perc': None, 'iso_perc': None, 'mom': None, 'Mw': None,
               's1': 0, 's2': 0, 'd1': 0, 'd2': 0, 'r1': 0, 'r2': 0}
    k = outfile.rfind(".")
    s1 = outfile[:k] + '_'
    s2 = outfile[k:]
    if outfile:
        plt.savefig(s1 + 'MT' + s2, bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    lw = 0.5
    plt.xlim(-100 - lw / 2, 100 + lw / 2)
    plt.ylim(-100 - lw / 2, 100 + lw / 2)
    for i in range(0, len(strike), 2):
        try:
            dc = beach((strike[i], dip[i], rake[i]), linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
            ax.add_collection(dc)
        except:
            try:
                dc = beach2((strike[i], dip[i], rake[i]), linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
                ax.add_collection(dc)
            except:
                print('plotting this moment strike / dip / rake failed: ', (strike[i], dip[i], rake[i]))
    if best and self.decompose:
        dc = beach((c['s1'], c['d1'], c['r1']), nofill=True, linewidth=lw * 3, edgecolor=(1., 0, 0))
        ax.add_collection(dc)
    if reference:
        dc = beach((ref['s1'], ref['d1'], ref['r1']), linewidth=lw * 3, nofill=True, edgecolor='red')
        ax.add_collection(dc)
    if outfile:
        plt.savefig(s1 + 'MT_DC' + s2, bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()

    # Plot histograms
    histogram(dc_perc, s1 + 'comp-1-DC' + s2, bins=(10, 100), range=(0, 100), xlabel='DC %',
              reference=ref['dc_perc'], reference2=(None, c['dc_perc'])[best], fontsize=fontsize)
    histogram(clvd_perc, s1 + 'comp-2-CLVD' + s2, bins=(20, 200), range=(-100, 100), xlabel='CLVD %',
              reference=ref['clvd_perc'], reference2=(None, c['clvd_perc'])[best], fontsize=fontsize)
    if not self.deviatoric:
        histogram(iso_perc, s1 + 'comp-3-ISO' + s2, bins=(20, 200), range=(-100, 100), xlabel='ISO %',
                  reference=ref['iso_perc'], reference2=(None, c['iso_perc'])[best], fontsize=fontsize)
        if save_ppdf:
            with open(os.path.join(os.path.dirname(outfile), 'unc_iso.txt'), 'w') as output:
                for row in iso_perc:
                    output.write(str(row) + '\n')
    # histogram(moment,    s1+'mech-0-moment'+s2, bins=20, range=(self.mt_decomp['mom']*0.7,
    # self.mt_decomp['mom']*1.4), xlabel='scalar seismic moment [Nm]', reference=ref['mom'], fontsize=fontsize)
    histogram(moment, s1 + 'mech-0-moment' + s2, bins=20,
              range=(self.mt_decomp['mom'] * 0.7 / 2, self.mt_decomp['mom'] * 1.4 * 2),
              xlabel='scalar seismic moment [Nm]', reference=ref['mom'], reference2=(None, c['mom'])[best],
              fontsize=fontsize)
    # histogram(Mw,        s1+'mech-0-Mw'+s2,     bins=20, range=(self.mt_decomp['Mw']-0.1,
    # self.mt_decomp['Mw']+0.1), xlabel='moment magnitude $M_W$', reference=ref['Mw'], fontsize=fontsize)
    histogram(Mw, s1 + 'mech-0-Mw' + s2, bins=20,
              range=(self.mt_decomp['Mw'] - 0.1 * 3, self.mt_decomp['Mw'] + 0.1 * 3),
              xlabel='moment magnitude $M_W$', reference=ref['Mw'], reference2=(None, c['Mw'])[best],
              fontsize=fontsize)
    histogram(strike, s1 + 'mech-1-strike' + s2, bins=72, range=(0, 360), xlabel=u'strike [°]', multiply=2,
              reference=((ref['s1'], ref['s2']), None)[reference == None],
              reference2=(None, (c['s1'], c['s2']))[best], fontsize=fontsize)
    histogram(dip, s1 + 'mech-2-dip' + s2, bins=18, range=(0, 90), xlabel=u'dip [°]', multiply=2,
              reference=((ref['d1'], ref['d2']), None)[reference == None],
              reference2=(None, (c['d1'], c['d2']))[best], fontsize=fontsize)
    histogram(rake, s1 + 'mech-3-rake' + s2, bins=72, range=(-180, 180), xlabel=u'rake [°]', multiply=2,
              reference=((ref['r1'], ref['r2']), None)[reference == None],
              reference2=(None, (c['r1'], c['r2']))[best], fontsize=fontsize)
    # save centroid, mt_decomp and uncertainty pdf
    if save_ppdf:
        with open(os.path.join(os.path.dirname(outfile), 'centroid'), 'wb') as f:
            pickle.dump(self.centroid, f)
        with open(os.path.join(os.path.dirname(outfile), 'mt_decomp'), 'wb') as f:
            pickle.dump(self.mt_decomp, f)
        with open(os.path.join(os.path.dirname(outfile), 'shifts'), 'wb') as f:
            pickle.dump(self.shifts, f)
        with open(os.path.join(os.path.dirname(outfile), 'depths'), 'wb') as f:
            pickle.dump(self.depths, f)
        with open(os.path.join(os.path.dirname(outfile), 'steps_x'), 'wb') as f:
            pickle.dump(self.steps_x, f)
        with open(os.path.join(os.path.dirname(outfile), 'A'), 'wb') as f:
            pickle.dump(A, f)
        with open(os.path.join(os.path.dirname(outfile), 'unc_dc.txt'), 'w') as output:
            for row in dc_perc:
                output.write(str(row) + '\n')
        with open(os.path.join(os.path.dirname(outfile), 'unc_clvd.txt'), 'w') as output:
            for row in clvd_perc:
                output.write(str(row) + '\n')
        with open(os.path.join(os.path.dirname(outfile), 'unc_moment.txt'), 'w') as output:
            for row in moment:
                output.write(str(row) + '\n')
        with open(os.path.join(os.path.dirname(outfile), 'unc_mw.txt'), 'w') as output:
            for row in Mw:
                output.write(str(row) + '\n')
        with open(os.path.join(os.path.dirname(outfile), 'unc_strike.txt'), 'w') as output:
            for row in strike:
                output.write(str(row) + '\n')
        with open(os.path.join(os.path.dirname(outfile), 'unc_dip.txt'), 'w') as output:
            for row in dip:
                output.write(str(row) + '\n')
        with open(os.path.join(os.path.dirname(outfile), 'unc_rake.txt'), 'w') as output:
            for row in rake:
                output.write(str(row) + '\n')
    if len(self.shifts) > 1:
        shift_step = self.SHIFT_step / self.max_samprate
        histogram(shift, s1 + 'time-shift' + s2, bins=len(self.shifts),
                  range=(self.shifts[0] - shift_step / 2., self.shifts[-1] + shift_step / 2.),
                  xlabel='time shift [s]',
                  reference=[0., None][reference == None], reference2=(None, c['shift'])[best], fontsize=fontsize)
        if save_ppdf:
            with open(os.path.join(os.path.dirname(outfile), 'unc_tshift.txt'), 'w') as output:
                for row in shift:
                    output.write(str(row) + '\n')
                output.write(str(shift_step))
    if len(self.depths) > 1:
        min_depth = (self.depths[0] - self.step_z / 2.) / 1e3
        max_depth = (self.depths[-1] + self.step_z / 2.) / 1e3
        histogram(depth, s1 + 'place-depth' + s2, bins=len(self.depths), range=(min_depth, max_depth),
                  xlabel='centroid depth [km]', reference=[self.event['depth'] / 1e3, None][reference == None],
                  reference2=(None, c['z'] / 1e3)[best], fontsize=fontsize)
        if save_ppdf:
            with open(os.path.join(os.path.dirname(outfile), 'unc_depth.txt'), 'w') as output:
                for row in depth:
                    output.write(str(row) + '\n')
                output.write(str(self.event['depth']) + '\n')
                output.write(str(min_depth) + ', ' + str(max_depth))
    if len(self.grid) > len(self.depths):
        x_lim = (self.steps_x[-1] + self.step_x / 2.) / 1e3
        histogram(NS, s1 + 'place-NS' + s2, bins=len(self.steps_x), range=(-x_lim, x_lim),
                  xlabel=u'← to south : centroid place [km] : to north →', reference=[0., None][reference == None],
                  reference2=(None, c['x'] / 1e3)[best], fontsize=fontsize)
        histogram(EW, s1 + 'place-EW' + s2, bins=len(self.steps_x), range=(-x_lim, x_lim),
                  xlabel=u'← to west : centroid place [km] : to east →', reference=[0., None][reference == None],
                  reference2=(None, c['y'] / 1e3)[best], fontsize=fontsize)
        if save_ppdf:
            with open(os.path.join(os.path.dirname(outfile), 'unc_NS.txt'), 'w') as output:
                for row in NS:
                    output.write(str(row) + '\n')
                output.write(str(x_lim))
            with open(os.path.join(os.path.dirname(outfile), 'unc_EW.txt'), 'w') as output:
                for row in EW:
                    output.write(str(row) + '\n')
                output.write(str(x_lim))

    self.log('\nUncertainty evaluation: plotted {0:d} mechanism of {1:d} requested.'.format(n_sum, n))
    self.log('Standard deviation : dc: {dc:4.2f}, clvd: {clvd:4.2f}, iso: {iso:4.2f}, Mw: {Mw:4.2f}, t: {t:4.2f}, '
             'ew: {y:4.2f}, ns: {x:4.2f}, z: {z:4.2f}'.format(**stdev))
    # self.log('Standard deviation : dc: {dc:4.2f}, clvd: {clvd:4.2f}, iso: {iso:4.2f}, Mw: {Mw:4.2f}, t: {t:4.2f}, '
    #          'x: {x:4.2f}, y: {y:4.2f}, z: {z:4.2f}, mom: {mom:4.2f}, '
    #          'S: {S:4.2f}, D: {D:4.2f}, R: {R:4.2f}'.format(**stdev))
    return stdev


def plot_MT_uncertainty_centroid(self, outfile='output/MT_uncertainty_centroid.png', n=100):
    """
    Similar as :func:`plot_uncertainty`, but only the best point of the space-time grid is taken into account,
    so the uncertainties should be Gaussian.
    """
    a = self.centroid['a']
    if self.deviatoric:
        a = a[:5]
    cov = self.centroid['GtGinv']
    A = np.random.multivariate_normal(a.T[0], cov, n)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    lw = 0.5
    plt.xlim(-100 - lw / 2, 100 + lw / 2)
    plt.ylim(-100 - lw / 2, 100 + lw / 2)

    for a in A:
        a = a[np.newaxis].T
        if self.deviatoric:
            a = np.append(a, [[0.]], axis=0)
        mt2 = a2mt(a, system='USE')
        # full = beach(mt2, linewidth=lw, nofill=True, edgecolor='black')
        try:
            full = beach(mt2, linewidth=lw, nofill=True, edgecolor='black')
        except:
            print(a)
            print(mt2)
        ax.add_collection(full)
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()


def plot_maps(self, outfile='output/map.png', beachball_size_c=False):
    """
    Plot figures showing how the solution is changing across the grid.

    :param outfile: Path to the file where to plot. If ``None``, plot to the screen. Because one figure is plotted
      for each depth, inserts an identifier before the last dot (`map.png` -> `map_1000.png`, `map_2000.png`, etc.).
    :type outfile: string, optional
    :param beachball_size_c: If ``True``, the sizes of the beachballs correspond to the posterior
      probability density function (PPD) instead of the variance reduction VR
    :type beachball_size_c: bool, optional

    Plot top view to the grid at each depth. The solutions in each grid point (for the centroid time with
    the highest VR) are shown by beachballs. The color of the beachball corresponds to its DC-part.
    The inverted centroid time is shown by a contour in the background and the condition number by contour lines.
    """
    r = self.grid_radius * 1e-3 * 1.1  # to km, *1.1
    if beachball_size_c:
        max_width = np.sqrt(self.max_sum_c)
    for z in self.depths:
        # prepare data points
        x = []
        y = []
        s = []
        CN = []
        MT = []
        color = []
        width = []
        highlight = []
        for gp in self.grid:
            if gp['z'] != z or gp['err']:
                continue
            x.append(gp['x'] / 1e3)
            y.append(gp['y'] / 1e3)
            s.append(gp['shift'])
            CN.append(gp['CN'])  # NS is x
            # coordinate, so switch it with y to be vertical
            MT.append(a2mt(gp['a'], system='USE'))
            VR = max(gp['VR'], 0)
            if beachball_size_c:
                width.append(self.step_x / 1e3 * np.sqrt(gp['sum_c']) / max_width)
            else:
                width.append(self.step_x / 1e3 * VR)
            if self.decompose:
                dc = float(gp['dc_perc']) / 100
                color.append((dc, 0, 1 - dc))
            else:
                color.append('black')
            highlight.append(self.centroid['id'] == gp['id'])
        if outfile:
            k = outfile.rfind(".")
            filename = outfile[:k] + "_{0:0>5.0f}".format(z) + outfile[k:]
        else:
            filename = None
        self.plot_map_backend(x, y, s, CN, MT, color, width, highlight, -r, r, -r, r, xlabel='west - east [km]',
                              ylabel='south - north [km]', title='depth {0:5.2f} km'.format(z / 1000),
                              beachball_size_c=beachball_size_c, outfile=filename)


def plot_slices(self, outfile='output/slice.png', point=None, beachball_size_c=False):
    """
    Plot vertical slices through the grid of solutions in point `point`.
    If `point` not specified, use the best solution as a point.

    :param outfile: Path to the file where to plot. If ``None``, plot to the screen.
    :type outfile: string, optional
    :param point: `x` and `y` coordinates (with respect to the epicenter) of a grid point where the slices are
        placed through. If ``None``, uses the coordinates of the inverted centroid.
    :type point: tuple, optional
    :param beachball_size_c: If ``True``, the sizes of the beachballs correspond to the posterior probability
        density function (PPD) instead of the variance reduction VR
    :type beachball_size_c: bool, optional

    The legend is the same as at :func:`plot_maps`.
    """
    if point:
        x0, y0 = point
    else:
        x0 = self.centroid['x']
        y0 = self.centroid['y']
    depth_min = self.depth_min / 1000
    depth_max = self.depth_max / 1000
    depth = depth_max - depth_min
    r = self.grid_radius * 1e-3 * 1.1  # to km, *1.1
    if beachball_size_c:
        max_width = np.sqrt(self.max_sum_c)
    for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE'):
        x = []
        y = []
        s = []
        CN = []
        MT = []
        color = []
        width = []
        highlight = []
        for gp in self.grid:
            if slice == 'N-S':
                X = -gp['x'];    Z = gp['y'] - y0
            elif slice == 'W-E':
                X = gp['y'];    Z = gp['x'] - x0
            elif slice == 'NW-SE':
                X = (gp['y'] - gp['x']) * 1 / np.sqrt(2);    Z = gp['x'] + gp['y'] - y0 - x0
            elif slice == 'SW-NE':
                X = (gp['y'] + gp['x']) * 1 / np.sqrt(2);    Z = gp['x'] - gp['y'] + y0 - x0
            Y = gp['z']
            if abs(Z) > 0.001 or gp['err']:
                continue
            x.append(X / 1e3)
            y.append(Y / 1e3)
            s.append(gp['shift'])
            CN.append(gp['CN'])
            MT.append(a2mt(gp['a'], system='USE'))
            VR = max(gp['VR'], 0)
            if beachball_size_c:
                width.append(self.step_x / 1e3 * np.sqrt(gp['sum_c']) / max_width)
            else:
                width.append(self.step_x / 1e3 * VR)
            if self.decompose:
                dc = float(gp['dc_perc']) / 100
                color.append((dc, 0, 1 - dc))
            else:
                color.append('black')
            highlight.append(self.centroid['id'] == gp['id'])
        if outfile:
            k = outfile.rfind(".")
            filename = outfile[:k] + '_' + slice + outfile[k:]
        else:
            filename = None
        xlabel = {'N-S': 'north - south', 'W-E': 'west - east', 'NW-SE': 'north-west - south-east',
                  'SW-NE': 'south-west - north-east'}[slice] + ' [km]'
        self.plot_map_backend(x, y, s, CN, MT, color, width, highlight, -r, r, depth_max + depth * 0.05,
                              depth_min - depth * 0.05, xlabel, 'depth [km]', title='vertical slice',
                              beachball_size_c=beachball_size_c, outfile=filename)


def plot_maps_sum(self, outfile='output/map_sum.png'):
    """
    Plot map and vertical slices through the grid of solutions showing the posterior probability
        density function (PPD).
    Contrary to :func:`plot_maps` and :func:`plot_slices`, the size of the beachball correspond not only to the
     PPD of grid-point through which is a slice placed, but to a sum of all grid-points which are before and behind.

    :param outfile: Path to the file where to plot. If ``None``, plot to the screen.
    :type outfile: string, optional

    The legend and properties of the function are similar as at function :func:`plot_maps`.
    """
    if not self.Cd_inv:
        return False  # if the data covariance matrix is unitary, we have no estimation of data errors,
        # so the PDF has good sense
    r = self.grid_radius * 1e-3 * 1.1  # to km, *1.1
    depth_min = self.depth_min * 1e-3
    depth_max = self.depth_max * 1e-3
    depth = depth_max - depth_min
    Ymin = depth_max + depth * 0.05
    Ymax = depth_min - depth * 0.05
    # for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE', 'top'):
    for slice in ('N-S', 'W-E', 'top'):
        X = []
        Y = []
        s = []
        CN = []
        MT = []
        color = []
        width = []
        highlight = []
        g = {}
        max_c = 0
        for gp in self.grid:
            if gp['err'] or gp['sum_c'] <= 0:
                continue
            if slice == 'N-S':
                x = -gp['x']
            elif slice == 'W-E':
                x = gp['y']
            elif slice == 'NW-SE':
                x = (gp['y'] - gp['x']) * 1 / np.sqrt(2)
            elif slice == 'SW-NE':
                x = (gp['y'] + gp['x']) * 1 / np.sqrt(2)
            x *= 1e-3
            y = gp['z'] * 1e-3
            if slice == 'top':
                x = gp['x'] * 1e-3
                y = gp['y'] * 1e-3  # NS is x coordinate, so switch it with y to be vertical
            if not x in g:
                g[x] = {}
            if not y in g[x]:
                g[x][y] = {'c': 0, 'max_c': 0, 'highlight': False}
            g[x][y]['c'] += gp['sum_c']
            if g[x][y]['c'] > max_c:
                max_c = g[x][y]['c']
            if gp['sum_c'] > g[x][y]['max_c']:
                g[x][y]['max_c'] = gp['sum_c']
                g[x][y]['a'] = gp['a']
                # g[x][y]['CN'] = gp['CN']
                # g[x][y]['s'] = gp['shift']
                if self.decompose:
                    g[x][y]['dc'] = gp['dc_perc']
            if self.centroid['id'] == gp['id']:
                g[x][y]['highlight'] = True
        for x in g:
            for y in g[x]:
                X.append(x)
                Y.append(y)
                # s.append(g[x][y]['s'])
                # CN.append(g[x][y]['CN'])
                MT.append(a2mt(g[x][y]['a'], system='USE'))
                if self.decompose:
                    dc = float(g[x][y]['dc']) * 0.01
                    color.append((dc, 0, 1 - dc))
                else:
                    color.append('black')
                highlight.append(g[x][y]['highlight'])
                width.append(self.step_x * 1e-3 * np.sqrt(g[x][y]['c'] / max_c))
        if outfile:
            k = outfile.rfind(".")
            filename = outfile[:k] + '_' + slice + outfile[k:]
        else:
            filename = None
        xlabel = {'N-S': 'north - south', 'W-E': 'west - east', 'NW-SE': 'north-west - south-east',
                  'SW-NE': 'south-west - north-east', 'top': 'south - north'}[slice] + ' [km]'
        if slice == 'top':
            ymin = -r
            ymax = r
            ylabel = 'west - east [km]'
            title = 'PDF sum: top view'
        else:
            ymin = Ymin
            ymax = Ymax
            ylabel = 'depth [km]'
            title = 'PDF sum: side view'
        # self.plot_map_backend(X, Y, s, CN, MT, color, width, highlight, -r, r, ymin, ymax, xlabel, ylabel,
        # title, True, filename)
        self.plot_map_backend(X, Y, None, None, MT, color, width, highlight, -r, r, ymin, ymax, xlabel, ylabel,
                              title, True, filename)


def plot_map_backend(self, x, y, s, CN, MT, color, width, highlight, xmin, xmax, ymin, ymax, xlabel='', ylabel='',
                     title='', beachball_size_c=False, outfile=None):
    """
    The plotting back-end for functions :func:`plot_maps`, :func:`plot_slices` and :func:`plot_maps_sum`.
        There is no need for calling it directly.
    """
    plt.rcParams.update({'font.size': 16})
    xdiff = xmax - xmin
    ydiff = ymax - ymin
    if xdiff > abs(1.3 * ydiff):
        plt.figure(figsize=(16, abs(ydiff / xdiff) * 14 + 3))
    else:
        plt.figure(figsize=(abs(xdiff / ydiff) * 11 + 2, 14))
    ax = plt.gca()
    # if xmin != ymin or xmax != ymax:
    plt.axis('equal')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax, int(np.sign(ydiff)))
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    Xmin = min(x)
    Xmax = max(x)
    Ymin = min(y)
    Ymax = max(y)
    width_max = max(width)

    for i in range(len(x)):
        if highlight[i]:
            c = plt.Circle((x[i], y[i]), self.step_x / 1e3 * 0.5, color='r')
            c.set_edgecolor('r')
            c.set_linewidth(10)
            c.set_facecolor('none')  # "none" not None
            c.set_alpha(0.7)
            ax.add_artist(c)
        if width[i] > self.step_x * 1e-3 * 0.04:
            try:
                b = beach(MT[i], xy=(x[i], y[i]), width=(width[i], width[i] * np.sign(ydiff)), linewidth=0.5,
                          facecolor=color[i], zorder=10)
            except:
                # print('Plotting this moment tensor in a grid point crashed: ', mt2, 'using mopad')
                try:
                    b = beach2(MT[i], xy=(x[i], y[i]), width=(width[i], width[i] * np.sign(ydiff)), linewidth=0.5,
                               facecolor=color[i], zorder=10)  # width: at side views, mirror along horizontal axis
                    # to avoid effect of reversed y-axis
                except:
                    print('Plotting this moment tensor in a grid point crashed: ', MT[i])
                else:
                    ax.add_collection(b)
            else:
                ax.add_collection(b)
        elif width[i] > self.step_x * 1e-3 * 0.001:
            b = plt.Circle((x[i], y[i]), width[i] / 2, facecolor=color[i], edgecolor='k', zorder=10, linewidth=0.5)
            ax.add_artist(b)

    if CN and s:
        # Set up a regular grid of interpolation points
        xi = np.linspace(Xmin, Xmax, 400)
        yi = np.linspace(Ymin, Ymax, 400)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        rbf = interpolate.Rbf(x, y, s, function='linear')
        z1 = rbf(xi, yi)
        rbf = interpolate.Rbf(x, y, CN, function='linear')
        z2 = rbf(xi, yi)

        shift = plt.imshow(z1, cmap=plt.get_cmap('PRGn'),
                           vmin=self.shift_min, vmax=self.shift_max, origin='lower',
                           extent=[Xmin, Xmax, Ymin, Ymax])
        levels = np.arange(1., 21., 1.)
        CN = plt.contour(z2, levels, cmap=plt.get_cmap('gray'), origin='lower', linewidths=1,
                         extent=[Xmin, Xmax, Ymin, Ymax], zorder=4)
        plt.clabel(CN, inline=1, fmt='%1.0f', fontsize=10)  # levels[1::2]  oznacit kazdou druhou caru
        CB1 = plt.colorbar(shift, shrink=0.5, extend='both', label='shift [s]')
        # CB2 = plt.colorbar(CN, orientation='horizontal', shrink=0.4,
        # label='condition number', ticks=[levels[0], levels[-1]])
        l, b, w, h = plt.gca().get_position().bounds
        ll, bb, ww, hh = CB1.ax.get_position().bounds
        # CB1.ax.set_position([ll-0.2*w, bb+0.2*h, ww, hh])
        CB1.ax.set_position([ll, bb + 0.2 * h, ww, hh])
        # ll,bb,ww,hh = CB2.ax.get_position().bounds
        # CB2.ax.set_position([l+0.58*w, bb+0.07*h, ww, hh])

    # legend beachball's color = DC%
    if self.decompose:
        x = y = xmin * 2
        plt.plot([x], [y], marker='o', markersize=15, color=(1, 0, 0), label='DC 100 %')
        plt.plot([x], [y], marker='o', markersize=15, color=(.5, 0, .5), label='DC 50 %')
        plt.plot([x], [y], marker='o', markersize=15, color=(0, 0, 1), label='DC 0 %')
        mpl.rcParams['legend.handlelength'] = 0
        if CN and s:
            plt.legend(loc='upper left', numpoints=1, bbox_to_anchor=(1, -0.05), fancybox=True)
        else:
            plt.legend(loc='upper right', numpoints=1, bbox_to_anchor=(0.95, -0.05), fancybox=True)

    # legend beachball's area
    if beachball_size_c:  # beachball's area = PDF
        r_max = self.step_x / 1e3 / 2
        r_half = r_max / 1.4142
        text_max = 'maximal PDF'
        text_half = 'half-of-maximum PDF'
        text_area = 'Beachball area ~ PDF'
    else:  # beachball's radius = VR
        VRmax = self.centroid['VR']
        r_max = self.step_x / 1e3 / 2 * VRmax
        r_half = r_max / 2
        text_max = 'VR {0:2.0f} % (maximal)'.format(VRmax * 100)
        text_half = 'VR {0:2.0f} %'.format(VRmax * 100 / 2)
        text_area = 'Beachball radius ~ VR'
    x_symb = [xmin + r_max, xmin][bool(CN and s)]  # min(xmin, -0.8*ydiff)
    x_text = xmin + r_max * 1.8
    y_line = ymin - ydiff * 0.11
    VRlegend = plt.Circle((x_symb, y_line), r_max, facecolor=(1, 0, 0), edgecolor='k', clip_on=False)
    ax.add_artist(VRlegend)
    VRlegendtext = plt.text(x_text, y_line, text_max, verticalalignment='center')
    ax.add_artist(VRlegendtext)
    y_line = ymin - ydiff * 0.20
    VRlegend2 = plt.Circle((x_symb, y_line), r_half, facecolor=(1, 0, 0), edgecolor='k', clip_on=False)
    ax.add_artist(VRlegend2)
    VRlegendtext2 = plt.text(x_text, y_line, text_half, verticalalignment='center')
    ax.add_artist(VRlegendtext2)
    y_line = ymin - ydiff * 0.26
    VRlegendtext3 = plt.text(x_text, y_line, text_area, verticalalignment='center')
    ax.add_artist(VRlegendtext3)

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()


def plot_3D(self, outfile='output/animation.mp4'):
    """
    Creates an animation with the grid of solutios. The grid points are labeled according to their VR.

    :param outfile: path to file for saving animation
    :type outfile: string
    """
    n = len(self.grid)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    VR = np.zeros(n)
    c = np.zeros((n, 3))
    for i in range(len(self.grid)):
        gp = self.grid[i]
        if gp['err']:
            continue
        x[i] = gp['x'] / 1e3
        y[i] = gp['y'] / 1e3  # NS is x coordinate, so switch it with y to be vertical
        z[i] = gp['z'] / 1e3
        vr = max(gp['VR'], 0)
        VR[i] = np.pi * (15 * vr) ** 2
        c[i] = np.array([vr, 0, 1 - vr])
        # if self.decompose:
        # dc = float(gp['dc_perc'])/100
        # c[i,:] = np.array([dc, 0, 1-dc])
        # else:
        # c[i,:] = np.array([0, 0, 0])
    # Create a figure and a 3D Axes
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('west - east [km]')
    ax.set_ylabel('south - north [km]')
    ax.set_zlabel('depth [km]')

    # Create an init function and the animate functions.
    # Both are explained in the tutorial. Since we are changing
    # the the elevation and azimuth and no objects are really
    # changed on the plot we don't have to return anything from
    # the init and animate function. (return value is explained
    # in the tutorial).
    def init():
        ax.scatter(x, y, z, marker='o', s=VR, c=c, alpha=1.)

    def animate(i):
        ax.view_init(elev=10., azim=i)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)  # Animate
    anim.save(outfile, writer=self.movie_writer, fps=30)  # Save
    # anim.save(outfile, writer=self.movie_writer, fps=30, extra_args=['-vcodec', 'libx264'])

# def plot_seismo(self, outfile='output/seismo.png', comp_order='ZNE', no_filter=False, cholesky=False,
# add_file=None, obs_style='k', obs_width=3, synt_style='r', synt_width=2, add_file_style='k:', add_file_width=2,
# add_file2=None, add_file2_style='b-', add_file2_width=2, plot_stations=None, plot_components=None, sharey=False):


def sts_window(self, tarr, tlen, npts):
    if self.taper_perc:
        t_slope = tlen * self.taper_perc
    else:
        t_slope = tlen * 0.125
    tstart = int((tarr - 2 - t_slope + self.centroid['shift']) * self.store_samprate)
    tend = tstart + int((tlen + 2 + 2 * t_slope) * self.store_samprate)
    # if tstart < 0:
    #     tend = tend + tstart
    #     tstart = 0
    if tend > npts:
        tend = npts
    return tstart, tend


def plot_seismo(self, outfile='output/seismo.png', comp_order='ZNE', cholesky=False, obs_style='dimgray', obs_width=2.5,
                synt_style='r', synt_width=1.7, or_style='--', or_width=1.2, add_file=None, add_file_style='k:',
                add_file_width=1.5, add_file2=None,
                add_file2_style='b-', add_file2_width=2, taper=False, plot_stations=None, plot_components=None,
                sharey='row'):
    """
    Plots the fit between observed and simulated seismogram.

    :param outfile: path to file for plot output; if ``None`` plots to the screen
    :type outfile: string, optional
    :param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
    :type comp_order: string, optional
    :param cholesky: plots standardized seismogram instead of original ones
    :type cholesky: bool, optional
    :param obs_style: line style for observed data
    :param obs_width: line width for observed data
    :param synt_style: line style for simulated data
    :param synt_width: line width for simulated data
    :param or_style: line style for original data without noise
    :param or_width: line width for original data without noise
    :param add_file: path to a reference file generated by function :func:`save_seismo`
    :type add_file: string or None, optional
    :param add_file_style: line style for reference data
    :param add_file_width: line width for reference data
    :param add_file2: path to second reference file
    :type add_file2: string or None, optional
    :param add_file2_style: line style for reference data
    :param add_file2_width: line width for reference data
    :param plot_stations: list of stations to plot; if ``None`` plots all stations
    :type plot_stations: list or None, optional
    :param plot_components: list of components to plot; if ``None`` plots all components
    :type plot_components: list or None, optional
    :param sharey: if ``True`` the y-axes for all stations have the same limits, otherwise the limits are chosen
        automatically for every station
    :type sharey: bool, optional
    """
    if cholesky and not len(self.LT) and not len(self.LT3):
        raise ValueError('Covariance matrix not set. Run "covariance_matrix()" first.')
    if self.data_orig:
        data_or = self.data_orig_shifts[self.centroid['shift_idx']]
    data = self.data_shifts[self.centroid['shift_idx']]
    npts = self.npts_slice
    samprate = self.samprate
    elemse = precalc_greens(self.event, self.centroid, self.targets, self.stations, self.comps_order,
                            self.GF_store_dir)
    if taper:
        taper_window = np.empty((npts, self.nr * 3))
        # tlen_max = self.stations[-1]['dist'] / 1000 * 0.2 + 25
        # tlen_max = (tlen_max + 4 + tlen_max * 0.05)

    for r in range(self.nr):
        tarr2 = UTCDateTime(0) + self.t_min + self.stations[r]['arr_time']
        tarr3 = self.t_min + self.stations[r]['arr_time']
        tlen = calc_taper_window(self.stations[r]['dist'])

        if taper:
            taper_wind = taper_wdw(tarr=tarr3, tlen=tlen, percentage=self.taper_perc, npts=npts,
                                   samp_rate=self.store_samprate, preserve_end_taper=True)
            for comp in range(3):
                taper_window[:, 3 * r + comp] = taper_wind

        for e in range(6):
            # q_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'], tarr2, tlen, self.taper_perc)
            q_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'])
            elemse[r][e].trim(UTCDateTime(0) + self.elemse_start_origin)
    if taper:
        plot_stations, comps, f, ax, ax_l, ea = self.plot_seismo_taper_backend_1(
            plot_stations, plot_components, comp_order, sharey=sharey, xlabel='', ylabel=(self.unit, None)[cholesky],
            title_prefix=('', 'standardized ')[cholesky])  # and self.LT3 != []])
    # else:
    #     plot_stations, comps, f, ax, ax_l, ea = self.plot_seismo_taper_backend_1(
    #         plot_stations, plot_components, comp_order, sharey=sharey, xlabel='', ylabel=(self.unit, None)[cholesky],
    #         title_prefix=('', 'standardized ')[cholesky and self.LT3 != []])
    else:
        plot_stations, comps, f, ax, ax_l, ea = self.plot_seismo_backend_1(
            plot_stations, plot_components, comp_order, sharey=sharey, ylabel=(self.unit, None)[cholesky],
            title_prefix=('', 'standardized ')[cholesky])  # and self.LT3 != []])

    t = np.arange(0, (npts - 0.5) / samprate, 1. / samprate)
    if add_file:
        add = np.load(add_file)
    if add_file2:
        add2 = np.load(add_file2)
    d_max = 0
    for sta in plot_stations:
        r = plot_stations.index(sta)
        tarr3 = self.t_min + self.stations[sta]['arr_time']
        tlen = calc_taper_window(self.stations[sta]['dist'])
        if taper:
            tstart, tend = self.sts_window(tarr=tarr3, tlen=tlen, npts=npts)
        else:
            tstart = 0
            tend = npts
        winlen = tend - tstart

        ax_l[r, 0].text(tstart, 1, " {0:s}\n {1:s}".format(data[sta][0].stats.network,
                                                           data[sta][0].stats.station),
                        ha='left', va='top', fontsize=18)
        ax_l[r, 0].text(tend, 1, u"{0:1.0f} Km\n{1:1.0f}°".format(self.stations[sta]['dist'] / 1000,
                                                                  self.stations[sta]['az']),
                        ha='right', va='top', fontsize=18)
        ax[r, 0].set_xlim([tstart, tend])
        # if no_filter:
        # SAMPRATE = self.data_unfiltered[sta][0].stats.sampling_rate
        # NPTS = int(npts/samprate * SAMPRATE),
        # SHIFT = int(round(self.centroid['shift']*SAMPRATE))
        # T = np.arange(0, (NPTS-0.5) / SAMPRATE, 1. / SAMPRATE)
        SYNT = {}
        for comp in range(3):
            SYNT[comp] = np.zeros(npts)
            for e in range(6):
                SYNT[comp] += elemse[sta][e][comp].data[0:npts] * self.centroid['a'][e, 0]
        comps_used = 0
        for comp in comps:
            synt = SYNT[comp]
            # if no_filter:
            # D = np.empty(NPTS)
            # for i in range(NPTS):
            # if i+SHIFT >= 0:
            # D[i] = self.data_unfiltered[sta][comp].data[i+SHIFT]
            # else:
            d = data[sta][comp][0:len(t)]
            if self.data_orig:
                d_or = data_or[sta][comp][0:len(t)]
            if cholesky and self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                if self.LT3:
                    # print(r, comp) # DEBUG
                    d = np.zeros(npts)
                    synt = np.zeros(npts)
                    if self.data_orig:
                        d_or = np.zeros(npts)
                    x1 = -npts
                    for COMP in range(3):
                        if not self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[COMP]]:
                            continue
                        x1 += npts
                        x2 = x1 + npts
                        y1 = comps_used * npts
                        y2 = y1 + npts
                        # print(self.LT3[sta][y1:y2, x1:x2].shape, data[sta][COMP].data[0:npts].shape) # DEBUG
                        d += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
                        synt += np.dot(self.LT3[sta][y1:y2, x1:x2], SYNT[COMP])
                        if self.data_orig:
                            d_or += np.dot(self.LT3[sta][y1:y2, x1:x2], data_or[sta][COMP].data[0:npts])
                else:
                    d = np.dot(self.LT[sta][comp], d)
                    synt = np.dot(self.LT[sta][comp], synt)
                    if self.data_orig:
                        d_or = np.dot(self.LT[sta][comp], d_or)
                comps_used += 1
            c = comps.index(comp)
            # if no_filter:
            # ax[r,c].plot(T,D, color='k', linewidth=obs_width)
            if self.data_orig:
                max_amp = max(max(abs(d)), max(abs(synt)), max(abs(d_or))) * 1.4
            else:
                max_amp = max(max(abs(d)), max(abs(synt))) * 1.4
            if taper:
                d = d * taper_window[:, 3 * sta + comp]
                synt = synt * taper_window[:, 3 * sta + comp]
                if self.data_orig:
                    d_or = d_or * taper_window[:, 3 * sta + comp]
                ax_l[r, c].set_ylim([-1.2, 1.2])
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]] or not cholesky:
                    l_td, = ax_l[r, c].plot(t, taper_window[:, 3 * sta + comp], '-b', linewidth=0.8)
                    ax_l[r, c].plot(t, taper_window[:, 3 * sta + comp] * -1, '-b', linewidth=0.8)
                    # l_td, = ax[r, c].plot(t, taper_window[:, 3 * sta + comp] * max_amp, '-b', linewidth=0.8)
                    # ax[r, c].plot(t, taper_window[:, 3 * sta + comp] * max_amp * -1, '-b', linewidth=0.8)
                    l_d, = ax[r, c].plot(t, d, obs_style, linewidth=obs_width)
                    # d_max = max(max(d), -min(d), d_max)
                    # d_max = max(max(synt), -min(synt), d_max)
                    ax[r, c].set_xlim([tstart, tend])
                    # ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                    # ylim = max(abs(ymin * 1.4), abs(ymax * 1.4))
                    # ax[r, c].set_ylim([-ylim, ylim])
                    if sharey == 'none' or sharey is False:
                        ax[r, c].set_ylim([-max_amp, max_amp])

                    h1, m1, s1 = format_timedelta(td(seconds=tarr3))
                    h2, m2, s2 = format_timedelta(td(seconds=winlen))
                    if h1:
                        ax_l[r, c].text(tstart, -1, ' Arr {:d}:{:02d}:{:02d} h'.format(h1, m1, s1),
                                        ha='left', va='bottom', fontsize=18)
                    elif m1:
                        ax_l[r, c].text(tstart, -1, ' Arr {:02d}:{:02d} m'.format(m1, s1),
                                        ha='left', va='bottom', fontsize=18)
                    else:
                        ax_l[r, c].text(tstart, -1, ' Arr {:02d} s'.format(s1),
                                        ha='left', va='bottom', fontsize=18)
                    if h2:
                        ax_l[r, c].text(tend, -1, u'Δ {:d}:{:02d}:{:02d} h '.format(h2, m2, s2),
                                        ha='right', va='bottom', fontsize=18)
                    elif m2:
                        ax_l[r, c].text(tend, -1, u'Δ {:02d}:{:02d} m '.format(m2, s2),
                                        ha='right', va='bottom', fontsize=18)
                    else:
                        ax_l[r, c].text(tend, -1, u'Δ {:02d} s '.format(s2),
                                        ha='right', va='bottom', fontsize=18)
                    ax[r, c].yaxis.offsetText.set_fontsize(15)
                else:
                    ax[r, c].plot([0], [0], 'w', linewidth=0)
                    ax[r, c].get_yaxis().set_visible(False)
                    ax_l[r, c].plot((tstart, tend), (0, 0), color='black', linewidth=0.8)

                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    l_s, = ax[r, c].plot(t, synt, synt_style, linewidth=synt_width)
                    if self.data_orig:
                        l_do, = ax[r, c].plot(t, d_or, or_style, color='blue', linewidth=or_width)
                    #    d_max = max(max(synt), -min(synt), d_max)
                else:
                    if not cholesky:
                        ax[r, c].plot(t, synt, color='darkgray', linewidth=2)
                        if self.data_orig:
                            ax[r, c].plot(t, d_or, or_style, color='blue', linewidth=or_width)
            else:
                ax_l[r, c].set_ylim([-1.2, 1.2])
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]] or not cholesky:
                    l_d, = ax[r, c].plot(t, d, obs_style, linewidth=obs_width)
                    # d_max = max(max(d), -min(d), d_max)
                    # d_max = max(max(synt), -min(synt), d_max)
                    ax[r, c].set_xlim([tstart, tend])
                    # ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                    # ylim = max(abs(ymin * 1.4), abs(ymax * 1.4))
                    # ax[r, c].set_ylim([-ylim, ylim])
                    if sharey == 'none' or sharey is False:
                        ax[r, c].set_ylim([-max_amp, max_amp])
                    # h1, m1, s1 = format_timedelta(td(seconds=tarr3))
                    # h2, m2, s2 = format_timedelta(td(seconds=winlen))
                    # if h1:
                    #     ax_l[r, c].text(tstart, -1, ' Arr {:d}:{:02d}:{:02d} h'.format(h1, m1, s1),
                    #                     ha='left', va='bottom', fontsize=15)
                    # elif m1:
                    #     ax_l[r, c].text(tstart, -1, ' Arr {:02d}:{:02d} m'.format(m1, s1),
                    #                     ha='left', va='bottom', fontsize=15)
                    # else:
                    #     ax_l[r, c].text(tstart, -1, ' Arr {:02d} s'.format(s1),
                    #                     ha='left', va='bottom', fontsize=15)
                    # if h2:
                    #     ax_l[r, c].text(tend, -1, u'Δ {:d}:{:02d}:{:02d} h '.format(h2, m2, s2),
                    #                     ha='right', va='bottom', fontsize=15)
                    # elif m2:
                    #     ax_l[r, c].text(tend, -1, u'Δ {:02d}:{:02d} m '.format(m2, s2),
                    #                     ha='right', va='bottom', fontsize=15)
                    # else:
                    #     ax_l[r, c].text(tend, -1, u'Δ {:02d} s '.format(s2),
                    #                     ha='right', va='bottom', fontsize=15)
                    ax[r, c].yaxis.offsetText.set_fontsize(15)
                else:
                    ax[r, c].plot([0], [0], 'w', linewidth=0)
                    ax[r, c].get_yaxis().set_visible(False)
                    ax_l[r, c].plot((tstart, tend), (0, 0), color='black', linewidth=0.8)

                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    l_s, = ax[r, c].plot(t, synt, synt_style, linewidth=synt_width)
                    if self.data_orig:
                        l_do, = ax[r, c].plot(t, d_or, or_style, color='blue', linewidth=or_width)
                #    d_max = max(max(synt), -min(synt), d_max)
                else:
                    if not cholesky:
                        ax[r, c].plot(t, synt, color='darkgray', linewidth=2)
                        if self.data_orig:
                            ax[r, c].plot(t, d_or, or_style, color='blue', linewidth=or_width)
            # else:
            #     if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[
            #         comp]] or not cholesky:  # do not plot seismogram if the component is not used and
            #         # Cholesky decomposition is plotted
            #         l_d, = ax[r, c].plot(t, d, obs_style, linewidth=obs_width)
            #         if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
            #             d_max = max(max(d), -min(d), d_max)
            #     else:
            #         ax[r, c].plot([0], [0], 'w', linewidth=0)
            #     if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
            #         l_s, = ax[r, c].plot(t, synt, synt_style, linewidth=synt_width)
            #         d_max = max(max(synt), -min(synt), d_max)
            #     else:
            #         if not cholesky:
            #             ax[r, c].plot(t, synt, color='darkgray', linewidth=2)
            if add_file:
                ax[r, c].plot(t, add[:, 3 * sta + comp], add_file_style, linewidth=add_file_width)
            if add_file2:
                ax[r, c].plot(t, add2[:, 3 * sta + comp], add_file2_style, linewidth=add_file2_width)
    # ax[-1, 0].set_ylim([-d_max, d_max])
    if taper:
        if self.data_orig:
            ea.append(
                f.legend((l_d, l_s, l_do, l_td), ('observation', 'synthetic', 'true data', 'taper window'), loc='lower center',
                         bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=4, numpoints=1, fontsize=22,
                         fancybox=True, handlelength=3))  # , borderaxespad=0.1
        else:
            ea.append(
                f.legend((l_d, l_s, l_td), ('observation', 'synthetic', 'taper window'), loc='lower center',
                         bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=3, numpoints=1, fontsize=22,
                         fancybox=True, handlelength=3))  # , borderaxespad=0.1
    else:
        if self.data_orig:
            ea.append(f.legend((l_d, l_s, l_do), ('observation', 'synthetic', 'true data'), loc='lower center',
                               bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=3, numpoints=1,
                               fontsize=22, fancybox=True, handlelength=3))  # , borderaxespad=0.1
        else:
            ea.append(f.legend((l_d, l_s), ('observation', 'synthetic'), loc='lower center',
                               bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=2, numpoints=1,
                               fontsize=22, fancybox=True, handlelength=3))  # , borderaxespad=0.1
    ea.append(f.text(0.1, 1.06 - 0.004 * len(plot_stations), 'x', color='white', ha='center', va='center'))
    if taper:
        self.plot_seismo_taper_backend_2(outfile, plot_stations, comps, ax, extra_artists=ea, sharey=sharey)
    else:
        self.plot_seismo_backend_2(outfile, plot_stations, comps, ax, extra_artists=ea, sharey=sharey)

# revert change seismo font size and covmat num of sts for improve paper image
def plot_covariance_function(self, outfile='output/covariance.png', comp_order='ZNE', crosscovariance=False,
                             style='k', width=2, plot_stations=None, plot_components=None):
    """
    Plots the covariance functions on whose basis is the data covariance matrix generated

    :param outfile: path to file for plot output; if ``None`` plots to the screen
    :type outfile: string, optional
    :param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
    :type comp_order: string, optional
    :param crosscovariance: if ``True`` plots also the crosscovariance between components
    :param crosscovariance: bool, optional
    :param style: line style
    :param width: line width
    :param plot_stations: list of stations to plot; if ``None`` plots all stations
    :type plot_stations: list or None, optional
    :param plot_components: list of components to plot; if ``None`` plots all components
    :type plot_components: list or None, optional
    """
    if not len(self.Cf):
        raise ValueError('Covariance functions not calculated or not saved. '
                         'Run "covariance_matrix(save_covariance_function=True)" first.')
    data = self.data_shifts[self.centroid['shift_idx']]

    plot_stations, comps, f, ax, ax_l, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order,
                                                                       crosscomp=crosscovariance, yticks=True,
                                                                       ylabel=None, plot_cf=True, sharey='all')

    dt = 1. / self.samprate
    t = np.arange(-np.floor(self.Cf_len / 2) * dt, (np.floor(self.Cf_len / 2) + 0.5) * dt, dt)
    COMPS = (1, 3)[crosscovariance]
    for sta in plot_stations:
        r = plot_stations.index(sta)
        for comp in comps:
            c = comps.index(comp)
            for C in range(COMPS):  # if crosscomp==False: C = 0
                d = self.Cf[sta][(comp, C)[crosscovariance], comp]
                # if len(t) != len(d): # DEBUG
                # t = np.arange(-np.floor(len(d)/2) * dt, (np.floor(len(d)/2)+0.5) * dt, dt) # DEBUG
                # print(len(d), len(t)) # DEBUG
                if type(d) == np.ndarray and self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    color = style
                    if len(t) != len(d):
                        t = np.arange(-np.floor(len(d) / 2) * dt, (np.floor(len(d) / 2) + 0.5) * dt, dt)
                    ax[COMPS * r + C, c].plot(t, d, color=style, linewidth=width)
                else:
                    ax[COMPS * r + C, c].plot([0], [0], 'w', linewidth=0)
        if crosscovariance:
            ax[3 * r, 0].set_ylabel(' \n Z ')
            ax[3 * r + 1, 0].set_ylabel(data[sta][0].stats.station + '\n N ')
            ax[3 * r + 2, 0].set_ylabel(' \n E ')
    self.plot_seismo_backend_2(outfile, plot_stations, comps, ax, yticks=True, extra_artists=ea, plot_cf=True,
                               sharey='all')


def plot_noise(self, outfile='output/noise.png', comp_order='ZNE', obs_style='k', obs_width=2, plot_stations=None,
               plot_components=None):
    """
    Plots the noise records from which the covariance matrix is calculated together with the inverted data

    :param outfile: path to file for plot output; if ``None`` plots to the screen
    :type outfile: string, optional
    :param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
    :type comp_order: string, optional
    :param obs_style: line style
    :param obs_width: line width
    :param plot_stations: list of stations to plot; if ``None`` plots all stations
    :type plot_stations: list or None, optional
    :param plot_components: list of components to plot; if ``None`` plots all components
    :type plot_components: list or None, optional
    """
    samprate = self.samprate

    plot_stations, comps, f, ax, ax_l, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order,
                                                                 ylabel=self.unit)

    t = np.arange(0, (self.npts_slice - 0.5) / samprate, 1. / samprate)
    d_max = 0
    for sta in plot_stations:
        r = plot_stations.index(sta)
        for comp in comps:
            d = self.data_shifts[self.centroid['shift_idx']][sta][comp][0:len(t)]
            c = comps.index(comp)
            if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                color = obs_style
                d_max = max(max(d), -min(d), d_max)
            else:
                color = 'gray'
            ax[r, c].plot(t, d, color, linewidth=obs_width)
            if len(self.noise[sta]) > comp:
                NPTS = len(self.noise[sta][comp].data)
                T = np.arange(-NPTS * 1. / samprate, -0.5 / samprate, 1. / samprate)
                ax[r, c].plot(T, self.noise[sta][comp], color, linewidth=obs_width)
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    d_max = max(max(self.noise[sta][comp]), -min(self.noise[sta][comp]), d_max)
    ax[-1, 0].set_ylim([-d_max, d_max])
    ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
    for r in range(len(plot_stations)):
        for i in range(len(comps)):
            l4 = ax[r, i].add_patch(mpatches.Rectangle((-NPTS / samprate, -ymax), NPTS / samprate, 2 * ymax,
                                                       color=(1.0, 0.6, 0.4)))  # (x,y), width, height
            l5 = ax[r, i].add_patch(
                mpatches.Rectangle((0, -ymax), self.npts_slice / samprate, 2 * ymax, color=(0.7, 0.7, 0.7)))
    ea.append(f.legend((l4, l5), ('$C_D$', 'inverted'), 'lower center',
                       bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=2, fontsize='small',
                       fancybox=True, handlelength=3, handleheight=1.2))  # , borderaxespad=0.1
    ea.append(f.text(0.1, 1.06 - 0.004 * len(plot_stations), 'x', color='white', ha='center', va='center'))
    self.plot_seismo_backend_2(outfile, plot_stations, comps, ax, extra_artists=ea)


def plot_spectra(self, outfile='output/spectra.png', comp_order='ZNE', plot_stations=None, plot_components=None):
    """
    Plots spectra of inverted data, standardized data, and before-event noise together

    :param outfile: path to file for plot output; if ``None`` plots to the screen
    :type outfile: string, optional
    :param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
    :type comp_order: string, optional
    :param plot_stations: list of stations to plot; if ``None`` plots all stations
    :type plot_stations: list or None, optional
    :param plot_components: list of components to plot; if ``None`` plots all components
    :type plot_components: list or None, optional
    """
    if not len(self.LT) and not len(self.LT3):
        raise ValueError('Covariance matrix not set. Run "covariance_matrix()" first.')
    data = self.data_shifts[self.centroid['shift_idx']]
    npts = self.npts_slice
    samprate = self.samprate

    plot_stations, comps, fig, ax, ax_l, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order,
                                                                   yticks=False, xlabel='frequency [Hz]',
                                                                   ylabel='amplitude spectrum')

    # plt.yscale('log')
    ax3 = np.empty_like(ax)
    fmin = np.zeros_like(ax, dtype=float)
    fmax = np.zeros_like(fmin)
    for i in range(len(plot_stations)):
        for j in range(len(comps)):
            # ax[i,j].set_yscale('log')
            ax3[i, j] = ax[i, j].twinx()
            # ax3[i,j].set_yscale('log')
    ax3[0, 0].get_shared_y_axes().join(*ax3.flatten().tolist())

    dt = 1. / samprate
    DT = 0.5 * dt
    f = np.arange(0, samprate * 0.5 * (1 - 0.5 / npts), samprate / npts)
    D_filt_max = 0
    for sta in plot_stations:
        r = plot_stations.index(sta)
        SYNT = {}
        comps_used = 0
        for comp in comps:
            d = data[sta][comp][0:npts]
            d_filt = d.copy()
            c = comps.index(comp)
            if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                if self.LT3:
                    d_filt = np.zeros(npts)
                    x1 = -npts
                    for COMP in comps:
                        if not self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[COMP]]:
                            continue
                        x1 += npts
                        x2 = x1 + npts
                        y1 = comps_used * npts
                        y2 = y1 + npts
                        d_filt += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
                else:
                    d_filt = np.dot(self.LT[sta][comp], d)
                comps_used += 1
            fmin[r, c] = self.stations[sta]['fmin']
            fmax[r, c] = self.stations[sta]['fmax']
            ax[r, c].tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
            ax[r, c].yaxis.offsetText.set_visible(False)
            ax3[r, c].get_yaxis().set_visible(False)
            noise = self.noise[sta][comp]
            NPTS = len(noise)
            NOISE = np.sqrt(np.square(np.real(np.fft.fft(noise)) * DT) * npts * dt / (NPTS * DT))
            f2 = np.arange(0, samprate * 1. * (1 - 0.5 / NPTS), samprate * 2 / NPTS)
            D = np.absolute(np.real(np.fft.fft(d)) * dt)
            l_d, = ax[r, c].plot(f, D[0:len(f)], 'k', linewidth=2, zorder=2)
            l_noise, = ax[r, c].plot(f2, NOISE[0:len(f2)], 'gray', linewidth=4, zorder=1)
            if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                D_filt = np.absolute(np.real(np.fft.fft(d_filt)) * dt)
                D_filt_max = max(D_filt_max, max(D_filt))
                l_filt, = ax3[r, c].plot(f, D_filt[0:len(f)], 'r', linewidth=1, zorder=3)
            # else:
            #     ax[r,c].plot([0],[0], 'w', linewidth=0)
    # y3min, y3max = ax3[-1,0].get_yaxis().get_view_interval()
    ax3[-1, 0].set_ylim([0, D_filt_max])
    # print (D_filt_max, y3max, y3min)
    align_yaxis(ax[0, 0], ax3[0, 0])
    ax[0, 0].set_xlim(0, self.fmax * 1.5)
    # ax[0,0].set_xscale('log')
    # f.legend((l4, l5), ('$C_D$', 'inverted'), 'upper center', ncol=2, fontsize='small', fancybox=True)
    ea.append(
        fig.legend((l_d, l_filt, l_noise), ('data', 'standardized data (by $C_D$)', 'noise'), loc='lower center',
                   bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=3, numpoints=1, fontsize='small',
                   fancybox=True, handlelength=3))  # , borderaxespad=0.1
    ea.append(fig.text(0.1, 1.06 - 0.004 * len(plot_stations), 'x', color='white', ha='center', va='center'))
    ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
    for r in range(len(plot_stations)):
        for c in range(len(comps)):
            if fmax[r, c]:
                ax[r, c].add_artist(Line2D((fmin[r, c], fmin[r, c]), (0, ymax), color='g', linewidth=1))
                ax[r, c].add_artist(Line2D((fmax[r, c], fmax[r, c]), (0, ymax), color='g', linewidth=1))
    self.plot_seismo_backend_2(outfile, plot_stations, comps, ax, yticks=False, extra_artists=ea)


def plot_seismo_backend_1(self, plot_stations, plot_components, comp_order, crosscomp=False, sharey='row',
                          yticks=True, title_prefix='', xlabel='time [s]', ylabel='', plot_cf=False):
    """
    The first part of back-end for functions :func:`plot_seismo`, :func:`plot_covariance_function`,
    :func:`plot_noise`, :func:`plot_spectra`. There is no need for calling it directly.
    """
    data = self.data_shifts[self.centroid['shift_idx']]

    plt.rcParams.update({'font.size': 23})

    if not plot_stations:
        plot_stations = range(self.nr)
    if plot_components:
        comps = plot_components
    elif comp_order == 'NEZ':
        comps = [1, 2, 0]
    else:
        comps = [0, 1, 2]

    COMPS = (1, 3)[crosscomp]
    f, ax = plt.subplots(len(plot_stations) * COMPS, len(comps), sharex='all', sharey=sharey,
                         figsize=(len(comps) * 6, len(plot_stations) * 2 * COMPS))
    if len(plot_stations) == 1 and len(comps) > 1:  # one row only
        ax = np.reshape(ax, (1, len(comps)))
    elif len(plot_stations) > 1 and len(comps) == 1:  # one column only
        ax = np.reshape(ax, (len(plot_stations), 1))
    elif len(plot_stations) == 1 and len(comps) == 1:  # one cell only
        ax = np.array([[ax]])

    for c in range(len(comps)):
        ax[0, c].set_title(title_prefix + data[0][comps[c]].stats.channel[2], fontsize=22)

    ax_l = np.empty_like(ax)
    plt.rcParams.update({'font.size': 23})
    for sta in plot_stations:
        r = plot_stations.index(sta)
        # ax[r, 0].set_ylabel(data[sta][0].stats.station + u"\n{0:1.0f} km, {1:1.0f}°".
        # format(self.stations[sta]['dist'] / 1000, self.stations[sta]['az']), fontsize=16)
        # SYNT = {}
        # comps_used = 0
        for comp in comps:
            c = comps.index(comp)
            for C in range(COMPS):  # if crosscomp==False: C = 0
                ax_l[COMPS * r + C, c] = ax[COMPS * r + C, c].twinx()
                ax_l[COMPS * r + C, c].set_frame_on(False)
                ax_l[COMPS * r + C, c].get_yaxis().set_visible(False)

                ax[COMPS * r + C, c].set_frame_on(False)
                ax[COMPS * r + C, c].locator_params(axis='x', nbins=7)
                ax[COMPS * r + C, c].tick_params(labelsize=19)
                # if c == 0:
                if yticks:
                    ax[r, c].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                    ax[r, c].get_yaxis().tick_left()
                    # ax[r, c].yaxis.get_offset_text().set_fontsize(16)
                    if c > 0 and sharey == 'row' or c > 0 and sharey == 'all':
                        ax[r, c].get_yaxis().set_visible(False)
                    if plot_cf:
                        ax[r, 0].get_yaxis().set_visible(False)
                else:
                    ax[COMPS * r + C, c].tick_params(axis='y', which='both', left='off', right='off',
                                                     labelleft='off')
                    ax[COMPS * r + C, c].yaxis.offsetText.set_visible(False)
                # else:
                #     ax[COMPS * r + C, c].get_yaxis().set_visible(False)
                if r == len(plot_stations) - 1 and C == COMPS - 1:
                    ax[COMPS * r + C, c].set_frame_on(True)
                    ax[COMPS * r + C, c].get_xaxis().tick_bottom()
                    ax[COMPS * r + C, c].spines['right'].set_visible(False)
                    ax[COMPS * r + C, c].spines['left'].set_visible(False)
                    ax[COMPS * r + C, c].spines['top'].set_visible(False)
                    # ax[COMPS * r + C, c].xaxis.set_ticks_position('bottom')
                else:
                    ax[COMPS * r + C, c].get_xaxis().set_visible(False)
    extra_artists = []
    if xlabel:
        extra_artists.append(f.text(0.5, 0.04 + 0.002 * len(plot_stations), xlabel, ha='center', va='center'))
    if ylabel:
        extra_artists.append(
            f.text(0.04 * (len(comps) - 1), 0.5, ylabel, ha='center', va='center', rotation='vertical'))
    return plot_stations, comps, f, ax, ax_l, extra_artists


def plot_seismo_taper_backend_1(self, plot_stations, plot_components, comp_order, crosscomp=False, sharey='row',
                                yticks=True, title_prefix='', xlabel='time [s]', ylabel=''):
    """
    The first part of back-end for functions :func:`plot_seismo`, :func:`plot_covariance_function`,
    :func:`plot_noise`, :func:`plot_spectra`. There is no need for calling it directly.
    """
    data = self.data_shifts[self.centroid['shift_idx']]
    # ylabel = self.unit

    plt.rcParams.update({'font.size': 18})

    if not plot_stations:
        plot_stations = range(self.nr)
    if plot_components:
        comps = plot_components
    elif comp_order == 'NEZ':
        comps = [1, 2, 0]
    else:
        comps = [0, 1, 2]

    COMPS = (1, 3)[crosscomp]
    f, ax = plt.subplots(len(plot_stations) * COMPS, len(comps), sharex='none', sharey=sharey,
                         figsize=(len(comps) * 6, len(plot_stations) * 2 * COMPS))
    if len(plot_stations) == 1 and len(comps) > 1:  # one row only
        ax = np.reshape(ax, (1, len(comps)))
    elif len(plot_stations) > 1 and len(comps) == 1:  # one column only
        ax = np.reshape(ax, (len(plot_stations), 1))
    elif len(plot_stations) == 1 and len(comps) == 1:  # one cell only
        ax = np.array([[ax]])

    for c in range(len(comps)):
        ax[0, c].set_title(title_prefix + data[0][comps[c]].stats.channel[2], fontsize=18)

    ax_l = np.empty_like(ax)
    plt.rcParams.update({'font.size': 18})
    for sta in plot_stations:
        r = plot_stations.index(sta)
        # ax[r,0].set_ylabel(data[sta][0].stats.station + u"\n{0:1.0f} km, {1:1.0f}°".
        # format(self.stations[sta]['dist']/1000, self.stations[sta]['az']), fontsize=16)
        # SYNT = {}
        # comps_used = 0
        for comp in comps:
            c = comps.index(comp)
            for C in range(COMPS):  # if crosscomp==False: C = 0
                ax_l[COMPS * r + C, c] = ax[COMPS * r + C, c].twinx()
                ax_l[COMPS * r + C, c].set_frame_on(False)

                ax[COMPS * r + C, c].set_frame_on(False)
                # ax[COMPS*r+C,c].locator_params(axis='x',nbins=7)
                ax[COMPS * r + C, c].tick_params(labelsize=15)
                # if c == 0:
                if yticks:
                    ax[r, c].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                    ax[r, c].get_yaxis().tick_left()
                    if c > 0 and sharey == 'row' or c > 0 and sharey == 'all':
                        ax[r, c].get_yaxis().set_visible(False)
                else:
                    ax[COMPS * r + C, c].tick_params(axis='y', which='both', left='off', right='off',
                                                     labelleft='off')
                    ax[COMPS * r + C, c].yaxis.offsetText.set_visible(False)
                ax_l[COMPS * r + C, c].get_yaxis().set_visible(False)
                # else:
                #     ax[COMPS*r+C,c].get_yaxis().set_visible(False)
                # if r == len(plot_stations)-1 and C==COMPS-1:
                #     ax[COMPS*r+C,c].get_xaxis().tick_bottom()
                # else:
                #     ax[COMPS*r+C,c].get_xaxis().set_visible(False)
                ax[COMPS * r + C, c].get_xaxis().set_visible(False)
    extra_artists = []
    if xlabel:
        extra_artists.append(f.text(0.5, 0.04 + 0.002 * len(plot_stations), xlabel, ha='center', va='center'))
    if ylabel:
        extra_artists.append(
            f.text(0.04 * (len(comps) - 1), 0.5, ylabel, ha='center', va='center', rotation='vertical'))
    return plot_stations, comps, f, ax, ax_l, extra_artists


def plot_seismo_backend_2(self, outfile, plot_stations, comps, ax, yticks=True, extra_artists=None, plot_cf=False,
                          sharey='none'):
    """
    The second part of back-end for functions :func:`plot_seismo`, :func:`plot_covariance_function`,
        :func:`plot_noise`, :func:`plot_spectra`. There is no need for calling it directly.
    """
    xmin, xmax = ax[0, 0].get_xaxis().get_view_interval()
    # ymin, ymax = ax[-1, 0].get_yaxis().get_view_interval()
    if yticks:
        if sharey == 'none' or sharey is False:
            for r in range(len(plot_stations)):
                for c in range(len(comps)):
                    # xmin, xmax = ax[r, c].get_xaxis().get_view_interval()
                    ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                    ymax = np.round(ymax, int(-np.floor(np.log10(ymax))))  # round high axis limit to first valid digit
                    ax[r, c].add_artist(Line2D((xmin, xmin), (0, ymax), color='black', linewidth=1))
                    ax[r, c].yaxis.set_ticks((0., ymax))
                    # ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                    ylim = max(abs(ymin), abs(ymax))
                    if plot_cf:
                        ax[r, c].set_ylim([-1, 1])
                    else:
                        ax[r, c].set_ylim([-ylim, ylim])
                    # if r == len(plot_stations) - 1:
                    #     ax[r, c].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
        else:
            for r in range(len(plot_stations)):
                ymin = 100
                ymax = -100
                for c in range(len(comps)):
                # xmin, xmax = ax[r, c].get_xaxis().get_view_interval()
                    ymin1, ymax1 = ax[r, c].get_yaxis().get_view_interval()
                    if ymin1 < ymin:
                        ymin = ymin1
                    if ymax1 > ymax:
                        ymax = ymax1
                ymax = np.round(ymax, int(-np.floor(np.log10(ymax))))  # round high axis limit to first valid digit
                # ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                ylim = max(abs(ymin), abs(ymax))
                if plot_cf:
                    ax[r, 0].set_ylim([-1, 1])
                else:
                    ax[r, 0].add_artist(Line2D((xmin, xmin), (0, ymax), color='black', linewidth=1))
                    ax[r, 0].yaxis.set_ticks((0., ymax))
                    ax[r, 0].set_ylim([-ylim, ylim])
                # if r == len(plot_stations) - 1:
                #     ax[r, c].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    # for c in range(len(comps)):
    if outfile:
        if extra_artists:
            plt.savefig(outfile, bbox_extra_artists=extra_artists, bbox_inches='tight', dpi=300)
            # plt.savefig(outfile, bbox_extra_artists=(legend,))
        else:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close('all')


def plot_seismo_taper_backend_2(self, outfile, plot_stations, comps, ax, yticks=True, extra_artists=None,
                                sharey='none'):
    """
    The second part of back-end for functions :func:`plot_seismo`, :func:`plot_covariance_function`,
        :func:`plot_noise`, :func:`plot_spectra`. There is no need for calling it directly.
    """
    if yticks:
        if sharey == 'none' or sharey is False:
            for r in range(len(plot_stations)):
                for c in range(len(comps)):
                    xmin, xmax = ax[r, c].get_xaxis().get_view_interval()
                    ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                    # ax[r, c].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
                    # ax[r, c].plot([xmin, xmax], [ymin, ymin], '|k')
                    # ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                    ymax = np.round(ymax, int(-np.floor(np.log10(ymax))))  # round high axis limit to first valid digit
                    ax[r, c].add_artist(Line2D((xmin, xmin), (0, ymax), color='black', linewidth=1))
                    ax[r, c].yaxis.set_ticks((0., ymax))
                    # ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                    ylim = max(abs(ymin), abs(ymax))
                    ax[r, c].set_ylim([-ylim, ylim])
        else:
            for r in range(len(plot_stations)):
                ymin = 100
                ymax = -100
                for c in range(len(comps)):
                    ymin1, ymax1 = ax[r, c].get_yaxis().get_view_interval()
                    if ymin1 < ymin:
                        ymin = ymin1
                    if ymax1 > ymax:
                        ymax = ymax1
                xmin, xmax = ax[r, 0].get_xaxis().get_view_interval()
                ymax = np.round(ymax, int(-np.floor(np.log10(ymax))))  # round high axis limit to first valid digit
                ax[r, 0].add_artist(Line2D((xmin, xmin), (0, ymax), color='black', linewidth=1))
                ax[r, 0].yaxis.set_ticks((0., ymax))
                # ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
                ylim = max(abs(ymin), abs(ymax))
                ax[r, 0].set_ylim([-ylim, ylim])

    # for r in range(len(plot_stations)):
    #     for c in range(len(comps)):
    #         ax[r,c].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    if outfile:
        if extra_artists:
            plt.savefig(outfile, bbox_extra_artists=extra_artists, bbox_inches='tight', dpi=300)
            # plt.savefig(outfile, bbox_extra_artists=(legend,))
        else:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close('all')


def plot_stations(self, outfile='output/stations.png', network=True, location=False, channelcode=False, fontsize=0):
    """
    Plot a map of stations used in the inversion.

    :param outfile: path to file for plot output; if ``None`` plots to the screen
    :type outfile: string, optional
    :param network: include network code into station label
    :type network: bool, optional
    :param location: include location code into station label
    :type location: bool, optional
    :param channelcode: include channel code into station label
    :type channelcode: bool, optional
    :param fontsize: font size for all texts in the plot; if zero, the size is chosen automatically
    :type fontsize: scalar, optional


    The stations are marked according to components used in the inversion.
    """
    if fontsize:
        plt.rcParams.update({'font.size': fontsize})
    plt.figure(figsize=(16, 12))
    plt.axis('equal')
    plt.xlabel('west - east [km]')
    plt.ylabel('south - north [km]')
    plt.title('Stations used in the inversion')
    plt.plot(self.centroid['y'] / 1e3, self.centroid['x'] / 1e3, marker='*', markersize=75, color='yellow',
             label='epicenter', linestyle='None')

    L1 = L2 = L3 = True
    for sta in self.stations:
        az = radians(sta['az'])
        dist = sta['dist'] / 1000  # from meter to kilometer
        y = cos(az) * dist  # N
        x = sin(az) * dist  # E
        label = None
        if sta['useN'] and sta['useE'] and sta['useZ']:
            color = 'red'
            if L1:
                label = 'all components used'; L1 = False
        elif not sta['useN'] and not sta['useE'] and not sta['useZ']:
            color = 'white'
            if L3:
                label = 'not used'; L3 = False
        else:
            color = 'gray'
            if L2:
                label = 'some components used'; L2 = False
        if network and sta['network']:
            l = sta['network'] + ':'
        else:
            l = ''
        l += sta['code']
        if location and sta['location']: l += ':' + sta['location']
        if channelcode: l += ' ' + sta['channelcode']
        # sta['weightN'] = sta['weightE'] = sta['weightZ']
        plt.plot([x], [y], marker='v', markersize=18, color=color, label=label, linestyle='None')
        plt.annotate(l, xy=(x, y), xycoords='data', xytext=(0, -14), textcoords='offset points',
                     horizontalalignment='center', verticalalignment='top', fontsize=14)
        # plt.legend(numpoints=1)
    plt.legend(bbox_to_anchor=(0., -0.15 - fontsize * 0.002, 1., .07), loc='lower left', ncol=4, numpoints=1,
               mode='expand', fontsize='small', fancybox=True)
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()


def plot_covariance_matrix(self, outfile=None, normalize=False, cholesky=False, fontsize=60, colorbar=False):
    """
    Plots figure of the data covariance matrix :math:`C_D`.

    :param outfile: path to file for plot output; if ``None`` plots to the screen
    :type outfile: string, optional
    :param normalize: normalize each blok (corresponding to one station) of the :math:`C_D` to the same value
    :type normalize: bool, optional
    :param cholesky: plots Cholesky decomposition of the covariance matrix :math:`L^T` instead of the :math:`C_D`
    :type cholesky: bool, optional
    :param fontsize: font size for all texts in the plot
    :type fontsize: scalar, optional
    :param colorbar: show a legend for the color map
    :type colorbar: bool, optional
    """
    plt.figure(figsize=(55, 50))
    fig, ax = plt.subplots(1, 1)
    if fontsize:
        plt.rcParams.update({'font.size': fontsize})
    Cd = np.zeros((6 * self.npts_slice, 6 * self.npts_slice))
    if not len(self.Cd):
        if self.correlation:
            raise ValueError(
                f'Correlation matrix not set or not saved. Run "covariance_matrix(save_non_inverted=True)" first.')
        else:
            raise ValueError(
                f'Covariance matrix not set or not saved. Run "covariance_matrix(save_non_inverted=True)" first.')

    i = 0
    if cholesky and self.LT3:
        matrix = self.LT3
    elif cholesky:
        matrix = [item for sublist in self.LT for item in sublist]
    else:
        matrix = self.Cd[:2]
    for C in matrix:
        if type(C) == int:
            continue
        if normalize and len(C):
            mx = max(C.max(), abs(C.min()))
            C *= 1. / mx
        l = len(C)
        Cd[i:i + l, i:i + l] = C
        i += l

    values = []
    labels = []
    i = 0
    n = self.npts_slice
    for stn in self.stations[:2]:
        if cholesky and self.LT3:
            j = stn['useZ'] + stn['useN'] + stn['useE']
            if j:
                values.append(i * n + j * n / 2)
                labels.append(stn['code'])
                i += j
        else:
            if stn['useZ']:
                values.append(i * n + n / 2)
                labels.append(stn['code'] + ' ' + 'Z')
                i += 1
            if stn['useN']:
                values.append(i * n + n / 2)
                labels.append(stn['code'] + ' ' + 'N')
                i += 1
            if stn['useE']:
                values.append(i * n + n / 2)
                labels.append(stn['code'] + ' ' + 'E')
                i += 1
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    mx = max(Cd.max(), abs(Cd.min()))
    cax = plt.matshow(Cd, fignum=1, cmap=plt.get_cmap('seismic'), vmin=-mx, vmax=mx)
    # cb = plt.colorbar(shift, shrink=0.6, extend='both', label='shift [s]')

    if colorbar:
        if self.correlation:
            cbar = plt.colorbar(cax, shrink=0.6, label='Correlation [$\mathrm{m}^2\,\mathrm{s}^{-2}$]')
        else:
            cbar = plt.colorbar(cax, shrink=0.6, label='Covariance [$\mathrm{m}^2\,\mathrm{s}^{-2}$]')
        # cbar = plt.colorbar(cax, ticks=[-mx, 0, mx])

    plt.xticks(values, labels, rotation='vertical')
    plt.yticks(values, labels)
    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close('all')
