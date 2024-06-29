#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import sys
import yaml
import warnings
import pandas as pd
import multiprocessing as mp
from datetime import timezone
from six import string_types
from datetime import datetime as dt
from modules_CMT.output import write_rmt_config
from modules_CMT.multiprocessings import find_optimum_params_mp
from modules_MGMT.confidential_materials import bmkg_fsdn_vars

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
logs:

2021-Jan:
1. read_network_coordinates add filter to exclude regional and nearfield station
2. add function load_local_streams, load_local_streams_xml, load_seiscomp3_arc, correct_xml
    to load more BMKG data and metadata format

2021-Feb:
1. change station selection by distance using read_network_coordinates filter to feature stn_select
2. Modified for using precalculate green function for max freq 0.3 Hz (still not efficient-->revert)
3. add function snr_calc() to calculate snr each trace data
4. add feature data selection by Signal to Noise Ratio value (correct_data_snrcheck/now: snr_filter())
5. add feature plot_snr() to evaluate snr threeshold
6. add count_stations() after snr and mouse check to check minimum station before processing
7. add function q_correction() to automatically eliminated error phase on next correction running
8. add function remove_station() to remove station which all components not used after snr, mouse and q_correction
    to optimize processing and cleaner plotting
9. add optimizing_solution() to improve solution quality by identifying and remove consistent low VR component/station
    in inversion

2021-Mar
1. using pyrocko precalculate green's function
2. optimize memory and prevent gap in data by automaticaly set t_before and after by sts distance
    (remove variable t_before and t_after)
3. optimize memory remove grid deepcopy on optimizing solution
4. Added station weighting and selection
5. change SNR calculation using msqr(sig)/msqr(noise)
6. move data correct inside snr_filter to more eficient processing
7. change data_correct method from using PAZ to xml inventory
8. move station distance filter to read_station
9. change my_filter to q_filter using acausal filter and taper function
10. change processing flow to initial inversion test for several velocity model and freq range

2021-Apr
1. set frequency shifting with several freq range by function set_working_frequency
2. set min sampling freq, max and min depth from gfstore config
3. find and set_optimum_params get and set best parameters by forward modeling and fitting + raw_data_shift
    (change multi sensitivity inversion)
4. fixed grid_search shifting x-y inverted in calculate centroid position and log
5. fixed plot_seismo last station y-scale error
6. set min sampling freq, max and min depth that accomodate from all gfstores

2021-May
1. add find_optimum_params to find best freq range (or with best velocity model)
2. remove unused function (attach PAZ, load stream data) -- backup in class_sens

2021-Jul
1. fixed snr_filter check for limited recorded data
2. add option to save ppdf result for later plotting

2021-Aug
1. add 'seismogram' option to invert in different units (disp/vel) (modif correct data, plot label and matrix d)
2. design new plot_seismo with taper window
3. fixed elementary seismogram taper bug
4. add option fit_cov to calculate fitting on original waveform even use covariance inversion
5. set optimum parameter add flex_threshold option to reduce VR threshold when too few stations pass the threshold
6. add option cmt/mt grid

2021-Sep
1. modif working frequency and data selection on final inversion
2. change working freq using list fmax (higher fmax for depth < 15 km)
3. grid set higher depth radius (2*R)

2022-Apr
1. get_seiscomp_event to get parameter from seiscomp3 databases

"""


class CMT:
    """
    Class for moment tensor inversion.

    # :type depth_unc: float, optional
    # :param depth_unc: vertical (depth) uncertainty of the location in meters (default 0)
    :type time_unc: float, optional
    :param time_unc: uncertainty of the origin time (default 0)
    :type deviatoric: bool, optional
    :param deviatoric: if ``False``: invert full moment tensor (6 components); if ``True`` invert deviatoric part
        of the moment tensor (5 components) (default ``False``)
    :type step_x: float, optional
    :param step_x: prefered horizontal grid spacing in meter (default 500 m)
    :type step_z: float, optional
    :param step_z: prefered vertical grid spacing in meter (default 500 m)
    :type threads: integer, optional
    :param threads: number of threads for paralelisation (default 2)
    :type circle_shape: bool, optional
    :param circle_shape: if ``True``, the shape of the grid is cylinder,
        otherwise it is rectangular box (default ``True``)
    :type decompose: bool, optional
    :param decompose: performs decomposition of the found moment tensor in each grid poind
    :type logfile: string, optional
    :param logfile: path to the logfile

    .. rubric:: _`Variables`

    The following description is useful mostly for developers. The parameters from the `Parameters`
        section are not listed, but they become also variables of the class. Two of them (``step_x`` and ``step_z``)
        are changed at some point, the others stay intact.

    ``data`` : list of :class:`~obspy.core.stream`
        Prepared data for the inversion. It's filled by function :func:`add_NEZ` or :func:`trim_filter_data`.
        The list is ordered ascending by epicentral distance of the station.
    ``data_raw`` : list of :class:`~obspy.core.stream`
        Data for the inversion. They are loaded by :func:`add_SAC`, :func:`load_files`, or :func:`load_streams_ArcLink`.
        Then they are corrected by :func:`correct_data` and trimmed by :func:`trim_filter_data`.
        The list is ordered ascending by epicentral distance of the station. After processing, the list references t
        o the same streams as ``data``.
    ``data_unfiltered`` : list of :class:`~obspy.core.stream`
        The copy of the ``data`` before it is filtered. Used for plotting results only.
    ``noise`` : list of :class:`~obspy.core.stream`
        Before-event slice of ``data_raw`` for later noise analysis. Created by :func:`trim_filter_data`.
    ``Cd_inv`` : list of :class:`~numpy.ndarray`
        Inverse of the data covariance matrix :math:`C_D^{-1}` saved block-by-block.
        Created by :func:`covariance_matrix`.
    ``Cd`` : list of :class:`~numpy.ndarray`
        Data covariance matrix :math:`C_D^{-1}` saved block-by-block. Optionally created by :func:`covariance_matrix`.
    ``LT`` : list of list of :class:`~numpy.ndarray`
        Cholesky decomposition of the data covariance matrix :math:`C_D^{-1}` saved block-by-block with the
        blocks corresponding to one component of a station. Created by :func:`covariance_matrix`.
    ``LT3`` : list of :class:`~numpy.ndarray`
        Cholesky decomposition of the data covariance matrix :math:`C_D^{-1}` saved block-by-block with the
        blocks corresponding to all component of a station. Created by :func:`covariance_matrix`.
    ``Cf`` :  list of 3x3 :class:`~numpy.ndarray` of :class:`~numpy.ndarray`
        List of arrays of the data covariance functions.
    ``Cf_len`` : integer
        Length of covariance functions.
    ``fmin`` : float
        Lower range of bandpass filter for data.
    ``fmax`` : float
        Higher range of bandpass filter for data.
    ``data_deltas`` : list of floats
        List of ``stats.delta`` from ``data_raw`` and ``data``.
    ``data_are_corrected`` : bool
        Flag whether the instrument response correction was performed.
    ``event`` : dictionary
        Information about event location and magnitude.
    ``stations`` : list of dictionaries
        Information about stations used in inversion. The list is ordered ascending by epicentral distance.
    ``stations_index`` : dictionary referencing ``station`` items.
        You can also access station information like ``self.stations_index['NETWORK_STATION_LOCATION_CHANNELCODE']
        ['dist']`` insted of `self.stations[5]['dist']`. `CHANNELCODE` are the first two letters of channel, e.g. `HH`.
    ``nr`` : integer
        Number of stations used, i.e. length of ``stations`` and ``data``.
    ``samprate`` : float
        Sampling rate used in the inversion.
    ``max_samprate`` : float
        Maximal sampling rate of the source data, which can be reached by integer decimation from all input samplings.
    ``t_min`` : float
        Starttime of the inverted time window, in seconds from the origin time.
    ``t_max`` :  float
        Endtime of the inverted time window, in seconds from the origin time.
    ``t_len`` : float
        Length of the inverted time window, in seconds (``t_max``-``t_min``).
    ``npts_elemse`` : integer
        Number of elementary seismogram data points (time series for one component).
    ``npts_slice`` : integer
        Number of data points for one component of one station used in inversion
        :math:`\mathrm{npts_slice} \le \mathrm{npts_elemse}`.
    ``tl`` : float
        Time window length used in the inversion.
    ``freq`` : integer
        Number of frequencies calculated when creating elementary seismograms.
    ``xl`` : float
        Parameter ``xl`` for `Axitra` code.
    ``npts_exp`` : integer
        :math:`\mathrm{npts_elemse} = 2^\mathrm{npts_exp}`
    ``grid`` : list of dictionaries
        Spatial grid on which is the inverse where is solved.
    ``centroid`` : Reference to ``grid`` item.
        The best grid point found by the inversion.
    ``depths`` : list of floats
        List of grid-points depths.
    ``radius`` : float
        Radius of grid cylinder / half of rectangular box horizontal edge length (depending on grid shape). Value in m.
    ``depth_min`` : float
        The lowest grid-poind depth (in meters).
    ``depth_max`` : float
        The highest grid-poind depth (in meters).
    ``shift_min``, ``shift_max``, ``shift_step`` :
        Three variables controling the time grid. The names are self-explaining. Values in second.
    ``SHIFT_min``, ``SHIFT_max``, ``SHIFT_step`` :
        The same as the previous ones, but values in samples related to ``max_samprate``.
    ``components`` : integer
        Number of components of all stations used in the inversion. Created by :func:`count_components`.
    ``data_shifts`` : list of lists of :class:`~obspy.core.stream`
        Shifted and trimmed ``data`` ready.
    ``d_shifts`` : list of :class:`~numpy.ndarray`
        The previous one in form of data vectors ready for the inversion.
    ``shifts`` : list of floats
        Shift values in seconds. List is ordered in the same way as the two previous list.
    ``mt_decomp`` : list
        Decomposition of the best CMT solution calculated by :func:`decompose` or :func:`decompose_mopad`
    ``max_VR`` : tuple (VR, n)
        Variance reduction `VR` from `n` components of a subset of the closest stations
    ``logtext`` : dictionary
        Status messages for :func:`html_log`
    ``models`` : dictionary
        Crust models used for calculating synthetic seismograms
    """

    from modules_CMT._load_data import vars_by_mag, set_event_info, read_stations, create_station_index, \
        check_a_station_present, load_fsdn, load_seedlink, load_seedlink_loop, load_seedlink_mp, waveform_time_window, \
        load_seiscomp3_archive, load_sac_archive, add_SC3arc, add_SC3arc2, load_q_data
    from modules_CMT._config import set_grid, set_time_grid, set_time_window, set_parameters, set_Greens_parameters, \
        set_initial_frequencies, set_working_frequencies, set_working_sampling, config_dict, max_time  # min_time
    from modules_CMT._pre_process import detect_mouse, detect_mouse_mp, snr_filter, calc_snr, correct_data_xml, \
        trim_filter_data, prefilter_data, decimate_shift_mp, station_weight, station_selection, obs_azimuth, \
        skip_short_records, count_components, count_stations, log_components, print_components
    from modules_CMT._covariance_matrix import covariance_matrix, covariance_matrix_prenoise, covariance_matrix_residual
    from modules_CMT._inversion import run_inversion, find_best_grid_point, VR_all_components, print_solution, \
        write_solution, print_fault_planes, unused_sts_filter, solution_quality_classification,\
        decompose_fault_planes  # save_seismo
    from modules_CMT._optimization import set_optimum_params, calc_threshold  # find_optimum_params,
    from modules_CMT._plot import plot_MT, plot_seismo, plot_stations, plot_slices, plot_maps, plot_maps_sum, plot_3D, \
        plot_covariance_matrix, plot_covariance_function, plot_MT_uncertainty_centroid, plot_uncertainty, \
        plot_noise, plot_spectra, plot_gmt, plot_snr, plot_map_backend, sts_window,\
        plot_seismo_backend_1, plot_seismo_backend_2, plot_seismo_taper_backend_1, plot_seismo_taper_backend_2

    def __init__(self, event_id, parentdir, configdir, outdir='output', logfile='$outdir/log.txt', output_mkdir=True,
                 comps_order='ZNE', GF_stores='', GF_store_dir='', seismogram='velocity', circle_shape=True,
                 vr_threshold=0, decompose=True, fit_cov=True):
        # self.sl_host = seedlink_host
        # self.sl_port = seedlink_port
        #
        self.agency = 'BMKG'
        self.configdir = configdir
        self.parentdir = parentdir
        self.outdir = outdir
        self.event_id = event_id
        if len(event_id.split('.')) > 1:
            self.evt = event_id.split('.')[1]
        else:
            self.evt = self.event_id
        if not os.path.exists(outdir) and output_mkdir:
            os.mkdir(outdir)
        self.logfile = open(logfile.replace('$outdir', self.outdir), 'w', 1)
        self.data_raw = []
        self.data_deltas = []
        self.logtext = {}
        self.models = {}
        self.event = {}
        self.stations = []
        self.stations_index = {}
        self.nr = 0
        self.list_net = ''
        self.list_sta = ''
        self.list_cha = ''
        self.GF_stores = GF_stores
        if isinstance(GF_stores, string_types):
            self.GF_store = GF_stores
        elif isinstance(GF_stores, list):
            self.GF_store = GF_stores[0]
        self.GF_store_dir = GF_store_dir
        self.store_samprate = 20.
        self.store_mindepth = 0.
        self.store_maxdepth = 1000000.
        self.comps_order = comps_order
        # self.max_t_v = max_time_velocity
        self.data_are_corrected = False
        #
        self.eval_mode = 'automatic'
        CONF = self.read_config()  # fixed outputdir for Q_RMT
        S = CONF['INV_SETTINGS']
        self.selected_station = CONF['LIST_STATIONS']
        self.agency = S['agency']
        self.dc_interest_eq = S['dc_interest_eq']
        self.centroid_inv = S['centroid_inversion']
        self.threads = S['cpu_threads']  # number of threads use in paralel computation
        self.deviatoric = S['deviatoric']
        self.maxfreq = S['freq_max']
        self.minfreq = S['freq_min']  # Hz  freq range for automatic filter_corner and weighting
        self.grid_radius = S['grid_radius']
        self.step_x = S['grid_step_x']
        self.step_z = S['grid_step_z']
        self.time_unc = S['grid_time_uncertainty']
        self.grid_z_radius = S['grid_z_radius']
        self.max_t_v = S['max_time_velocity']  # m/s endtime seismogram using linear relationship (dist/max_t_v)
        self.merge_to_seiscomp = S['merge2seiscomp']
        self.force_merge_CD = S['mergeCD']
        self.min_comp = S['min_comp']
        self.min_sta = S['min_sta']
        self.save_sens_file = S['save_sens_file']
        self.snr_min = S['snr_min']
        self.stn_min_dist = S['stn_min_dist']
        self.stn_max_dist = S['stn_max_dist']
        self.taper_perc = S['taper_perc']/100
        self.use_cov_residual = S['use_cov_residual']
        self.use_cov_noise = S['use_cov_noise']
        #
        # self.time_unc = time_unc  # s
        # self.step_x = step_x  # m
        # self.step_z = step_z  # m
        # self.grid_radius = grid_radius  # m
        # self.grid_z_radius = grid_z_radius  # m
        self.circle_shape = circle_shape
        self.idx_use = {0: 'useZ', 1: 'useN', 2: 'useE'}
        self.idx_weight = {0: 'weightZ', 1: 'weightN', 2: 'weightE'}
        # self.min_comp = min_comp
        # self.min_sta = min_sta
        self.num_sta = 0
        self.components = 0
        self.grid = []
        # self.location_unc = location_unc  # m
        # self.depth_unc = depth_unc  # m
        # self.max_points = max_points
        # self.rupture_velocity = rupture_velocity
        # self.add_rupture_length = add_rupture_length
        #
        self.seismogram = seismogram
        if seismogram == 'velocity':
            self.seisunit = 'VEL'
            self.unit = 'velocity [m/s]'
        elif seismogram == 'displacement':
            self.seisunit = 'DISP'
            self.unit = 'displacement [m]'
        else:
            self.seisunit = 'ACC'
            self.unit = 'acceleration [m/s2]'
        # self.velocity_ot_the_fastest_wave = velocity_ot_the_fastest_wave
        # self.velocity_ot_the_slowest_wave = velocity_ot_the_slowest_wave
        # self.threads = threads
        self.data = []
        self.noise = []
        self.targets = []
        self.fmax = 0.
        self.max_amp = None
        self.max_snr = None
        #
        self.d_shifts = []
        self.data_shifts = []
        self.raw_data_shifts = []
        self.shifts = []
        #
        self.Cd_inv = []
        self.Cd = []
        self.LT = []
        self.LT3 = []
        self.Cf = []
        self.Cd_inv_shifts = []
        self.Cd_shifts = []
        self.LT_shifts = []
        self.Cf_len = 0
        self.Cfd_len = 0
        self.Cft_len = 0
        #
        # self.deviatoric = deviatoric
        self.decompose = decompose
        self.mt_decomp = []
        self.max_VR = ()
        self.fit_cov = fit_cov
        self.centroid = {}
        #
        self.sens_param = []
        self.vr_th = vr_threshold
        #
        self.movie_writer = 'mencoder'  # None for default
        # self.inv = resp_inventory

        # self.min_snr = 2.

        self.log('Inversion of ' + {1: 'deviatoric part of', 0: 'full'}[self.deviatoric] + ' moment tensor (' +
                 {1: '5', 0: '6'}[self.deviatoric] + ' components)')

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        self.logfile.close()
        if os.path.exists(os.path.join(self.parentdir, 'tmp', 'pid.pid')):
            os.remove(os.path.join(self.parentdir, 'tmp', 'pid.pid'))

        if self.data_raw:
            del self.data_raw
        if self.stations:
            del self.stations
        if self.stations_index:
            del self.stations_index
        #
        if self.grid:
            del self.grid
        #
        if self.data:
            del self.data
        if self.noise:
            del self.noise
        if self.targets:
            del self.targets
        #
        if self.d_shifts:
            del self.d_shifts
        if self.data_shifts:
            del self.data_shifts
        if self.raw_data_shifts:
            del self.raw_data_shifts
        if self.shifts:
            del self.shifts
        #
        if self.Cd_inv:
            del self.Cd_inv
        if self.Cd:
            del self.Cd
        if self.LT:
            del self.LT
        if self.LT3:
            del self.LT3
        if self.Cf:
            del self.Cf
        if self.Cd_inv_shifts:
            del self.Cd_inv_shifts
        if self.Cd_shifts:
            del self.Cd_shifts
        if self.LT_shifts:
            del self.LT_shifts
        #
        if self.sens_param:
            del self.sens_param
        gc.collect()

    def log(self, s, newline=True, printcopy=False):
        """
        Write text into log file
        :param self:
        :param s: Text to write into log
        :type s: string
        :param newline: if is ``True``, add LF symbol (\\\\n) at the end
        :type newline: bool, optional
        :param printcopy: if is ``True`` prints copy of ``s`` also to stdout
        :type printcopy: bool, optional
        """
        self.logfile.write(s)
        if newline:
            self.logfile.write('\n')
        if printcopy:
            print(s)

    def read_config(self):
        my_config = {'INV_SETTINGS': None, 'LIST_STATIONS': None}

        if os.path.exists(os.path.join(self.configdir, f'{self.evt}.ini')):
            with open(os.path.join(self.configdir, f'{self.evt}.ini'), "r") as yamlfile:
                man_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
            if man_conf[0]['CONFIG_MANUAL_INVERSION']:
                try:
                    if self.evt != man_conf[0]['CONFIG_EVENT']:
                        sys.exit('Config ID not match to the current event. '
                                 'Run default config first to get current event configuration')
                    else:
                        self.eval_mode = 'manual'
                        my_config['INV_SETTINGS'] = man_conf[0]['INVERSION_SETTINGS']
                        self.log('Using manual inversion settings . . .', printcopy=True)
                except KeyError:
                    sys.exit('Config ID not match to the current event. '
                             'Run default config first to get current event configuration')
            else:
                with open(os.path.join(self.configdir, 'default.ini'), "r") as yamlfile:
                    def_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
                my_config['INV_SETTINGS'] = def_conf[0]['INVERSION_SETTINGS']

            if man_conf[0]['CONFIG_MANUAL_STATION']:
                self.log('Using manual selected station . . .', printcopy=True)
                my_config['LIST_STATIONS'] = man_conf[0]['STATIONS_SELECTION']
                self.eval_mode = 'manual'

        else:
            with open(os.path.join(self.configdir, 'default.ini'), "r") as yamlfile:
                def_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
            my_config['INV_SETTINGS'] = def_conf[0]['INVERSION_SETTINGS']
            self.eval_mode = 'automatic'

        return my_config

    def load_data(self, lat, lon, dep, mag, origin, remark, list_station_file, seedlink_client=None,
                  fsdn=False, datadir=None, inv_file=None):

        print('\n * Event "{:s}", {:s}, Mag: {:.1f}'.format(self.event_id, remark, mag))
        desc = '\n * Read Station and Waveforms Data...' \
               '\n   Started {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))
        self.log(desc, printcopy=True)

        # url, user, passw = bmkg_fsdn_vars()

        self.set_event_info(self.evt, lat, lon, dep, mag, origin, remark)
        self.min_snr, mindist, maxdist, self.use_cov_noise, t_before = self.vars_by_mag()
        try:
            self.read_stations(filename=list_station_file, min_dist=mindist, max_dist=maxdist, max_cluster_num=8)
            # self.load_seiscomp3_arc2(datadir, xml_dir=os.path.join(inp_dir, 'inventory'))
            # self.load_q_data2(datadir, xml_dir=os.path.join(inp_dir, 'inventory'))
            # self.load_seiscomp3_arc(datadir, xml_file=os.path.join(inp_dir, 'bmkg.xml'))
            self.load_q_data(datadir, xml_file=inv_file)
            #self.load_fsdn(url=url, user=user, password=passw, t_before=t_before)
        except ValueError:
            return False
        else:
            return True

    def set_manual_station(self, sector_num=8):
        self.sector_number = sector_num
        self.station_weight()
        self.set_initial_frequencies(fmin=self.minfreq, fmax=self.maxfreq)
        use_sta = []
        unuse_sta = []
        part_use_sta = []
        self.list_all_azimuth = []
        for station in self.stations:
            station['useZ'] = station['useN'] = station['useE'] = False
            self.list_all_azimuth.append(station['az'])
            for sel_sta in self.selected_station:
                if sel_sta['code'] == station['code'] and sel_sta['channelcode'] == station['channelcode']:
                    station['fmin'] = float(sel_sta['fmin'])
                    station['fmax'] = float(sel_sta['fmax'])
                    station['model'] = sel_sta['model']
                    for comp in 'ZNE':
                        if comp in sel_sta['comp']:
                            station[f'use{comp}'] = True
                    self.fmax = max(self.fmax, station['fmax'])
                    break

            if station['useZ'] and station['useN'] and station['useE']:
                use_sta.append(station)
            elif station['useE'] or station['useN'] or station['useZ']:
                part_use_sta.append(station)
            else:
                unuse_sta.append(station)

        self.used_a = pd.DataFrame(use_sta, columns=['network', 'code', 'channelcode', 'lon', 'lat'])
        self.used_n = pd.DataFrame(unuse_sta, columns=['network', 'code', 'channelcode', 'lon', 'lat'])
        self.used_p = pd.DataFrame(part_use_sta, columns=['network', 'code', 'channelcode', 'lon', 'lat'])
        self.azimuth_gap = self.obs_azimuth()

    def data_QC(self, window_width=100, plot=False):

        desc = '\n * Check SNR, Detect Disturbances, Remove Instrument Response, and Select Best Station...' \
               '\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))
        self.log(desc, printcopy=True)

        self.set_initial_frequencies(fmin=self.minfreq, fmax=self.maxfreq)
        if plot:
            self.snr_filter(min_snr=self.min_snr, window_width=window_width, figures=os.path.join(self.outdir, 'snr', ''))  #
            if self.threads > 1:
                figures = os.path.join(self.outdir, 'mouse', '')
                if not os.path.exists(figures):
                    os.mkdir(figures)
                self.detect_mouse_mp(figures=figures, fit_t1=-10, fit_t2c=window_width)
            else:
                self.detect_mouse(figures=os.path.join(self.outdir, 'mouse', ''), fit_t1=-10, fit_t2c=window_width)
        else:
            self.snr_filter(min_snr=self.min_snr, window_width=window_width)  #
            if self.threads > 1:
                self.detect_mouse_mp(fit_t1=-10, fit_t2c=window_width)
            else:
                self.detect_mouse(fit_t1=-10, fit_t2c=window_width)

        self.correct_data_xml()
        # self.correct_data()

        # check wether min number of station more than min_sta to be processed
        return self.count_stations()

    def select_best_station(self, sector_num=8, numsta_sect=1, outfile=''):
        self.station_selection(sector_num=sector_num, numsta_sect=numsta_sect, outfile=outfile)
        # self.max_gap, start_gap, end_gap = self.obs_azimuth()

    def initial_inversion_settings(self):

        desc = '\n * Set Inital Inversion Settings...' \
               '\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))
        self.log(desc, printcopy=True)

        return self.set_parameters(init=True, depth_min=5000, depth_max=40000)

    def run_initial_inversion(self, sensitivity_file='', plot=True):

        desc = '\n * Run Initial Inversion...' \
               '\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))
        self.log(desc, printcopy=True)

        self.log_components()
        self.trim_filter_data(noise_slice=self.use_cov_noise)
        self.decimate_shift_mp()

        if self.use_cov_noise:
            try:
                self.covariance_matrix(crosscovariance=False, n_toeplitz=False, r_toeplitz=True, correlation=True,
                                       save_non_inverted=True, save_covariance_function=True, normalize=False,
                                       init=True)
            except ValueError:
                return False, ''

        self.run_inversion()
        self.find_best_grid_point()
        # self.update_arrtime()
        self.decompose_fault_planes()

        init_dir = os.path.join(self.outdir, 'initial_result')
        if not os.path.exists(init_dir):
            os.mkdir(init_dir)

        stdev = None
        if self.use_cov_noise:
            try:
                stdev = self.plot_uncertainty(outfile=os.path.join(init_dir, 'uncertainty.png'), n=1000,
                                              fontsize=15, save_ppdf=False)
            except ValueError:
                pass
            q = self.solution_quality_classification(stdev, outfile=os.path.join(init_dir, 'solution_quality.png'))
        else:
            q = self.solution_quality_classification(outfile=os.path.join(init_dir, 'solution_quality.png'))

        desc = '\n   Initial result: "Quality {}"' \
               '\n     VR={:.1f}%, CN={:.1f}, DC={:.1f}%, shift_time={:.2f}s, ' \
               '\n     lat={:.2f}, lon={:.2f}, depth={:.2f}km'. \
            format(q, self.centroid['VR'] * 100, self.centroid['CN'], self.centroid['dc_perc'],
                   self.centroid['shift'], self.centroid['lat'], self.centroid['lon'], self.centroid['z'] / 1e3)

        if sensitivity_file:
            with open(sensitivity_file, 'w') as f:
                f.write(desc)

        desc += '\n   {:s}...'.format(dt.now(timezone.utc).strftime("%H:%M:%S"))
        self.log(desc, printcopy=True)

        # self.print_solution()

        self.print_fault_planes()
        if plot:
            self.plot_MT(outfile=os.path.join(init_dir, 'MT_solution.png'))
            self.plot_seismo(outfile=os.path.join(init_dir, 'seismo.png'), sharey='all')
            # self.plot_seismo(outfile=os.path.join(init_dir, 'seismo_taper.png'), taper=True, sharey='all')
            self.plot_seismo(outfile=os.path.join(init_dir, 'seismo_yr.png'), sharey='row')  # selection_scheme
            # self.plot_seismo(outfile=os.path.join(init_dir, 'seismo_taper_yr.png'), taper=True, sharey='row')
            if self.use_cov_noise:
                self.plot_covariance_function(outfile=os.path.join(init_dir, 'covariance_function.png'))
                self.plot_covariance_matrix(outfile=os.path.join(init_dir, 'covariance_matrix.png'),
                                            colorbar=True)
                # self.plot_MT_uncertainty_centroid(outfile=os.path.join(init_dir, 'MT_unc_centroid.png'), n=200)
                self.plot_seismo(outfile=os.path.join(init_dir, 'seismo_cova.png'), cholesky=True, sharey='all')
                self.plot_seismo(outfile=os.path.join(init_dir, 'seismo_cova_yr.png'), cholesky=True, sharey='row')
                # self.plot_seismo(outfile=os.path.join(init_dir, 'seismo_cova_taper.png'), cholesky=True, taper=True,
                #                  sharey='all')
                self.plot_spectra(outfile=os.path.join(init_dir, 'spectra.png'))
                self.plot_noise(outfile=os.path.join(init_dir, 'noise.png'))

        return stdev, q

    def optimum_inversion_settings(self, sensitivity_file=''):

        desc = '\n * Finding optimum station, velocity model and frequency filter...' \
               '\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))
        self.log(desc, printcopy=True)
        # test multiprocessing sens_params
        # print('\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S")))
        # self.find_optimum_params(sensitivity_file=sensitivity_file)
        # print('\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S")))
        if self.threads > 1:
            pool = mp.Pool(processes=self.threads)
            results = [pool.apply_async(find_optimum_params_mp,
                                        args=(i, self.stations, self.npts_slice, self.raw_data_shifts, self.GF_stores,
                                              self.seismogram, self.t_min, self.npts_elemse, self.samprate,
                                              self.comps_order, self.centroid, self.event, self.GF_store_dir,
                                              self.freq_shifts, self.elemse_start_origin, sensitivity_file,
                                              self.stations, self.idx_use, self.idx_weight, self.taper_perc))
                       for i in range(len(self.stations))]
            output = [p.get() for p in results]
        else:
            output = []
            for i in range(len(self.stations)):
                result = find_optimum_params_mp(i, self.stations, self.npts_slice, self.raw_data_shifts, self.GF_stores,
                                                self.seismogram, self.t_min, self.npts_elemse, self.samprate,
                                                self.comps_order, self.centroid, self.event, self.GF_store_dir,
                                                self.freq_shifts, self.elemse_start_origin, sensitivity_file,
                                                self.stations, self.idx_use, self.idx_weight, self.taper_perc)
                output.append(result)

        self.sens_param = sorted(output, key=lambda d: d['dist'])

        # print('\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S")))
        # from sensitivity_parameter find optimum model and freq for each station, use all comp with good VR
        self.numsta_sect = 1
        self.count_stations()
        if self.num_sta < 4:
            self.numsta_sect = 2
        self.set_optimum_params(outfile=os.path.join(self.outdir, 'final_sta_azimuth.png'))
        if not self.count_stations():
            return False
        # self.count_components()
        if not self.set_parameters(log=True, depth_min=5000, depth_max=40000):
            return False
        # self.max_gap, start_gap, end_gap = self.obs_azimuth()

        return True

    def run_final_inversion(self, input_dir, sensitivity_file='', plot=True, save_config=False):

        desc = '\n * Run Final Inversion...' \
               '\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))
        self.log(desc, printcopy=True)

        self.trim_filter_data(noise_slice=self.use_cov_noise)
        self.decimate_shift_mp()

        if self.use_cov_residual or self.use_cov_noise:
            try:
                self.covariance_matrix(crosscovariance=False, n_toeplitz=False, r_toeplitz=True, correlation=True,
                                       save_non_inverted=True, save_covariance_function=True, normalize=False,
                                       plot_each_cov_function=True)
                # self.covariance_matrix_SACF(save_non_inverted=True, save_covariance_function=True)
                # self.Cf_len = self.Cft_len
                # self.covariance_data(crosscovariance=True, save_non_inverted=True, save_covariance_function=True)
            except ValueError:
                return False, ''

        self.run_inversion()
        self.find_best_grid_point()
        self.decompose_fault_planes()

        stdev = None
        if self.use_cov_residual or self.use_cov_noise:
            try:
                stdev = self.plot_uncertainty(outfile=os.path.join(self.outdir, 'uncertainty.png'), n=1000,
                                              fontsize=15, save_ppdf=True)
            except ValueError:
                stdev = None
                pass
            q = self.solution_quality_classification(stdev, outfile=os.path.join(self.outdir, 'solution_quality.png'))
        else:
            q = self.solution_quality_classification(outfile=os.path.join(self.outdir, 'solution_quality.png'))

        # self.update_arrtime()
        desc = '\n   Final results: "Quality {}"' \
               '\n     VR={:.1f}%, CN={:.1f}, DC={:.1f}%, shift_time={:.2f}s' \
               '\n     lat={:.2f}, lon={:.2f}, depth={:.2f}km'.\
            format(q, self.centroid['VR'] * 100, self.centroid['CN'], self.centroid['dc_perc'], self.centroid['shift'],
                   self.centroid['lat'], self.centroid['lon'], self.centroid['z'] / 1e3)
        self.log(desc, printcopy=True)

        desc = '\n * Plotting Solution...' \
               '\n   {:s}'.format(dt.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"))
        self.log(desc, printcopy=True)

        self.VR_all_components()
        if sensitivity_file:
            self.print_components(out_file=sensitivity_file, final=True)
        self.print_solution()
        self.print_fault_planes()

        if save_config:
            write_rmt_config(self.evt, self.outdir, self.config_dict())

        if plot:
            # if len(self.grid) > len(self.depths):
            #     self.plot_maps(outfile=os.path.join(self.outdir, 'map.png'))
            if len(self.depths) > 1:
                self.plot_slices(outfile=os.path.join(self.outdir, 'slice.png'))
            if len(self.grid) > len(self.depths) > 1:
                self.plot_maps_sum(outfile=os.path.join(self.outdir, 'map_sum.png'))
            try:
                self.plot_MT(outfile=os.path.join(self.outdir, 'MT_solution.png'))
            except:
                pass
            # self.plot_MT_uncertainty_centroid(outfile=os.path.join(self.outdir, 'MT_uncertainty_centroid.png'), n=200)
            # self.plot_seismo(outfile=os.path.join(self.outdir, 'seismo.png'))
            # self.plot_seismo(outfile=os.path.join(self.outdir, 'seismo_sharey.png'), sharey=True)
            if self.use_cov_residual or self.use_cov_noise:
                self.plot_covariance_function(outfile=os.path.join(self.outdir, 'covariance_function.png'))
                self.plot_covariance_matrix(outfile=os.path.join(self.outdir, 'covariance_matrix.png'), colorbar=True)
                # self.plot_seismo(outfile=os.path.join(self.outdir, 'seismo_cova.png'), cholesky=True)
                if self.use_cov_noise:
                    self.plot_spectra(outfile=os.path.join(self.outdir, 'spectra.png'))
                    self.plot_noise(outfile=os.path.join(self.outdir, 'noise.png'))

            self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_ya_allsts.png'), sharey='all')
            # self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_ya_taper_allsts.png'), taper=True, sharey='all')
            self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_yr_allsts.png'), sharey='row')
            # self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_yr_taper_allsts.png'), taper=True, sharey='row')

            self.unused_sts_filter()
            self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_ya.png'), sharey='all')
            self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_yr.png'), sharey='row')
            # self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_taper_ya.png'), taper=True,
            #                  sharey='all')
            # self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_true_waveform_taper_yr.png'), taper=True,
            #                  sharey='row')
            if self.use_cov_residual or self.use_cov_noise:
                self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_cova_waveform_ya.png'), cholesky=True,
                                 sharey='all')
                self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_cova_waveform_yr.png'), cholesky=True,
                                 sharey='row')
                # self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_cova_waveform_taper_ya.png'), taper=True,
                #                  cholesky=True, sharey='all')
                # self.plot_seismo(outfile=os.path.join(self.outdir, 'fit_cova_waveform_taper_yr.png'), taper=True,
                #                  cholesky=True, sharey='row')
            # self.plot_gmt(inputdir=input_dir, outfile=os.path.join(self.outdir, 'solution_map.png'), quality=q,
            #               add_waveform=True, time_zone='WIT')

        return stdev, q
