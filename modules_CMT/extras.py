#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# mpl.use('Qt5Agg')  # for test_module
from math import gcd
from obspy import UTCDateTime, Stream, Trace
from pyrocko.obspy_compat.base import to_obspy_trace
from scipy import signal
from matplotlib.lines import Line2D


def rtp2ned(MT_RTP):
    return [MT_RTP[1], MT_RTP[2], MT_RTP[0], -MT_RTP[5], MT_RTP[3], -MT_RTP[4]]  # convert to NED system



def traces_to_stream(synthetic_traces, comp_order='ZNE'):
    comp = 0
    strm = []
    strms = []
    # streams = Stream()
    j = 0
    while j < len(synthetic_traces[0]):
        comp += 1
        if comp <= 3:
            strm.append(synthetic_traces[0][j])
            j += 1
            if j == len(synthetic_traces[0]):
                if len(strm) == 3:
                    streams = Stream()
                    for str_comp in comp_order:
                        streams += to_obspy_trace([trc for trc in strm if trc.channel == str_comp][0])
                    # q_filter(streams, fmin=fmin, fmax=fmax)
                    strms.append(streams)
                    # streams.write(os.path.join(out_dir, "{:s}.mseed".format(streams[0].stats.station)), format="MSEED")
        else:
            if len(strm) == 3:
                streams = Stream()
                for str_comp in comp_order:
                    streams += to_obspy_trace([trc for trc in strm if trc.channel == str_comp][0])
                # q_filter(streams, fmin=fmin, fmax=fmax)
                strms.append(streams)
                # streams.write(os.path.join(out_dir, "{:s}.mseed".format(streams[0].stats.station)), format="MSEED")
            comp = 0
            strm = []
    return strms


def read_local_cache(cache_file):
    print(f'\nLoading SeedLink Local Cache . . .')
    cache = open(cache_file, "rb")
    return pickle.load(cache)


def write_local_cache(cache_data, cache_file):
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)


def histogram(data, outfile=None, bins=100, range=None, xlabel='', multiply=1, reference=None, reference2=None,
              fontsize=None):
    """
    Plots a histogram of a given data.

    :param data: input values
    :type data: array
    :param outfile: filename of the output. If ``None``, plots to the screen.
    :type outfile: string or None, optional
    :param bins: number of bins of the histogram
    :type bins: integer, optional
    :param range: The lower and upper range of the bins. Lower and upper outliers are ignored. If not provided,
        range is (data.min(), data.max()).
    :type range: tuple of 2 floats, optional
    :param xlabel: x-axis label
    :type xlabel: string, optional
    :param multiply: Normalize the sum of histogram to a given value. If not set, normalize to 1.
    :type multiply: float, optional
    :param reference: plots a line at the given value as a reference
    :type reference: array_like, scalar, or None, optional
    :param reference2: plots a line at the given value as a reference
    :type reference2: array_like, scalar, or None, optional
    :param fontsize: size of the font of tics, labels, etc.
    :type fontsize: scalar, optional

    Uses `matplotlib.pyplot.hist <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist>`_
    """
    if fontsize:
        plt.rcParams.update({'font.size': fontsize})
    weights = np.ones_like(data) / float(len(data)) * multiply
    if type(bins) == tuple:
        try:
            n = 1 + 3.32 * np.log10(len(data))  # Sturgesovo pravidlo
        except:
            n = 10
        if range:
            n *= (range[1]-range[0])/(max(data)-min(data))
        bins = max(min(int(n), bins[1]), bins[0])
    plt.hist(data, weights=weights, bins=bins, range=range, color='blue', ec='black', linewidth=0.6)
    ax = plt.gca()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    if reference != None:
        try:
            iter(reference)
        except:
            reference = (reference,)
        for ref in reference:
            ax.add_artist(Line2D((ref, ref), (0, ymax), color='r', linewidth=5))
    if reference2 != None:
        try:
            iter(reference2)
        except:
            reference2 = (reference2,)
        for ref in reference2:
            ax.add_artist(Line2D((ref, ref), (0, ymax), color=(1., 0, 0), linewidth=3, linestyle='-'))
    if range:
        plt.xlim(range[0], range[1])
    if xlabel:
        plt.xlabel(xlabel)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=320)
    else:
        plt.show()
    plt.clf()
    plt.close()


def isnumber(n):
    try:
        float(n)
        return True
    except ValueError:
        return False


def to_percent(y, position):
    """
    Something with tics positioning used by :func:`histogram`
    # Ignore the passed in position. This has the effect of scaling the default tick locations.
    """
    s = "{0:2.0f}".format(100 * y)
    if mpl.rcParams['text.usetex'] is True:
        return s + r' $\%$'
    else:
        return s + ' %'


def find_exponent(m):
    for k in range(30, 10, -1):
        if isinstance(m, float):
            if m / 10. ** k >= 1:
                return k
        elif isinstance(m, np.ndarray):
            if max(abs(m)) / 10.**k >= 1:
                return k


def align_yaxis(ax1, ax2, v1=0, v2=0):
    """
    Adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


def q_filter(data, fmin=None, fmax=None, tarr=UTCDateTime(0), tlen=50, taper_percentage=None, taper=False):
    """
    Taper Function and Acausal Butterworth filter for filtering both elementary and observed seismograms
    """
    data.taper(max_percentage=0.05, type="cosine")

    if fmax and fmin:
        data.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    elif fmax:
        data.filter('lowpass', freq=fmax, corners=2, zerophase=True)
    elif fmin:
        data.filter('highpass', freq=fmin, corners=2, zerophase=True)

    data.detrend(type='demean')
    # if taper and taper_percentage:  # test diff using non-tapered data on inversion
    if taper_percentage:
        q_taper(data, tarr=tarr, tlen=tlen, percentage_start=taper_percentage)


def q_taper(data, tarr=UTCDateTime(0), tlen=50, percentage_start=0.20, percentage_end=None):
    """
    Taper Function for filtering both elementary and observed seismograms
    """

    if percentage_end is None:
        percentage_end = percentage_start

    t_slope = int(tlen * percentage_start)  # slope in arrival
    start_trim = tarr - 2 - t_slope
    end_trim = tarr + tlen + t_slope
    if type(data) == Stream:
        # plt.figure()
        for tr in data:
            # plt.plot(tr)
            s, e = tr.stats.starttime, tr.stats.endtime
            tr.trim(start_trim, end_trim, pad=True, fill_value=0.0)
            tr.taper(max_percentage=percentage_start, side='left', type="cosine")
            tr.taper(max_percentage=percentage_end, side='right', type="cosine")
            tr.trim(s, e, pad=True, fill_value=0.0)
            # plt.plot(tr)
            # plt.show()
            # plt.clf()
            # plt.close()
    elif type(data) == Trace:
        s, e = data.stats.starttime, data.stats.endtime
        data.trim(start_trim, end_trim, pad=True, fill_value=0.0)
        data.taper(max_percentage=percentage_start, side='left', type="cosine")
        data.taper(max_percentage=percentage_end, side='right', type="cosine")
        data.trim(s, e, pad=True, fill_value=0.0)
    # else:
    #     window = taper_wdw(tarr, tlen, percentage_start, len(data))
    #     data = data * window


def taper_wdw(tarr, tlen, percentage, npts, samp_rate=1, preserve_end_taper=False):
    """

    :param tarr: time arrival P
    :param tlen: len of plateu taper depends on the distance from function: calc_taper_window
    :param percentage: percentage of taper slope
    :param npts: number of points used in inversion and plot
    :param samp_rate: sampling rate of data and also the taper
    :return:
    """
    t_slope = int(tlen * percentage)
    window_wid = int((tlen + 2 + 2 * t_slope) * samp_rate)
    taper_window = signal.windows.tukey(window_wid, alpha=2*percentage)
    start_sample = int((tarr - 2 - t_slope) * samp_rate)
    if start_sample < 0:  # window starts before OT (t=0s), clip the start of taper window
        taper_window = taper_window[abs(start_sample):]
    elif start_sample > 0:  # window starts after OT, add zero pad on the start
        taper_window = np.concatenate((np.zeros(start_sample), taper_window))
        # taper_window = np.concatenate((np.zeros(start_sample + 1), taper_window))

    end_sample = int(npts - len(taper_window))
    if end_sample < 0:  # window ends after loaded data,
        if preserve_end_taper:  # shift the end of taper window
            mid_taper = int(len(taper_window)/2)
            if end_sample % 2 == 0:
                del_idx = np.arange(mid_taper-int(abs(end_sample)/2), mid_taper+int(abs(end_sample)/2))
            else:
                del_idx = np.arange(mid_taper-int(abs(end_sample)/2), mid_taper+int(abs(end_sample)/2) + 1)
            taper_window = np.delete(taper_window, del_idx)
        else:  # clip the end of taper window
            taper_window = taper_window[:-abs(end_sample)]
    elif end_sample > 0:  # window ends before last data, add zero pad on the end
        taper_window = np.concatenate((taper_window, np.zeros(end_sample)))

    return taper_window


def moving_average_scale(residual, window_width):
    if window_width % 2 == 0:
        window_width -= 1
    n = len(residual)
    r_mean = np.mean(residual)
    residual += 0.0001 * r_mean
    half_window = int(window_width/2)
    r = np.concatenate((np.ones(half_window) * r_mean, residual), axis=0)
    r = np.concatenate((r, np.ones(half_window) * r_mean), axis=0)
    dev = np.zeros(n)
    for i in range(n):
        dev[i] = np.sqrt(sum(r[i:i+2*half_window+1]**2) / window_width)
    dev += 0.0001 * np.mean(dev)
    return residual/dev, dev


def covariance_function(x1, x2=None):
    x1_mean = np.mean(x1)
    if x2 is None:  # do autocovariance
        x2 = x1.copy()
        x2_mean = x1_mean
    else:
        x2_mean = np.mean(x2)
    n1 = len(x1)
    n2 = len(x2)
    # if len(x2) != n:
    #     raise ValueError("Len of two vector is different!")
    v1 = np.concatenate((np.zeros(n2-1), x1, np.zeros(n2-1)), axis=0)
    v2s = x2 - x2_mean
    xcov = np.zeros(n1 + n2 - 1)
    for j in range(len(xcov)):
        xcov[j] = np.sum((v1[j:j+n2] - x1_mean) * v2s)
    return xcov


def correlation_function(x1, x2=None):
    if x2 is None:  # do autocorrelation
        x2 = x1.copy()
    n1 = len(x1)
    n2 = len(x2)
    # if len(x2) != n:
    #     raise ValueError("Len of two vector is different!")
    v1 = np.concatenate((np.zeros(n2-1), x1, np.zeros(n2-1)), axis=0)
    v2 = x2
    xcorr = np.zeros(n1 + n2 - 1)
    for j in range(len(xcorr)):
        xcorr[j] = np.sum(v1[j:j+n2] * v2)
    return xcorr


def next_power_of_2(n):
    """
    Return next power of 2 greater than or equal to ``n``

    :type n: integer
    """
    return 2**(n-1).bit_length()


def lcmm(b, *args):
    """
    Returns generelized least common multiple.

    :param b: numbers to compute least common multiple of them
    :param args: numbers to compute least common multiple of them
    :type b: float, which is a multiple of 0.00033333
    :type args: float, which is a multiple of 0.00033333
    :returns: the least multiple of ``a`` and ``b``
    """
    b = 3/b
    if b - round(b) < 1e6:
        b = round(b)
    for a in args:
        a = 3/a
        if a - round(a) < 1e6:
            a = round(a)
        b = gcd(a, b)
    return 3/b


def format_timedelta(td):
    minutes, seconds = divmod(td.seconds + td.days * 86400, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


def decimate(a, n=2):
    """
    Decimates given sequence.

    :param a: data
    :type a: 1-D array
    :param n: decimate factor
    :type n: integer, optional

    Before decimating, filter out frequencies over Nyquist frequency using :func:`numpy.fft.fft`
    """
    npts = len(a)
    # NPTS = npts # next_power_of_2(npts)
    NPTS = npts
    A = np.fft.fft(a, NPTS)
    idx = int(np.round(npts/n/2))
    A[idx:NPTS-idx+1] = 0+0j
    a = np.fft.ifft(A)
    if npts % (2*n) == 1 or n != 2:  # keep odd length for decimate factor 2
        return a[:npts:n].real
    else:
        return a[1:npts:n].real


def matrix_correlation(new_matrix, old_matrix):
    MD = np.zeros([len(old_matrix)])
    for i, (n, o) in enumerate(zip(new_matrix, old_matrix)):
        compare_CD = np.vstack((n[0], o[0]))
        MD[i] = np.corrcoef(compare_CD)[0][1]

    return MD.mean()
