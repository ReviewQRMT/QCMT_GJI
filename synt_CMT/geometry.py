"""
==========================
geometric module by eQ
==========================

using:


"""
import numpy as np
from math import radians, acos, atan2, cos, sin, asin, sqrt, degrees
# import math as mt

r = 6371  # Radius of earth in kilometers. Use 3956 for miles


def lin_dist(lon_source, lat_source, lon_obs, lat_obs):
    dist = acos(cos(radians(90 - lat_obs)) * cos(radians(90 - lat_source)) + sin(radians(90 - lat_obs)) *
                sin(radians(90 - lat_source)) * cos(radians(lon_obs - lon_source))) * r
    # print(dist)
    return dist


def haversine(lon_source, lat_source, lon_obs, lat_obs):
    # convert decimal degrees to radians
    lon_source, lat_source, lon_obs, lat_obs = map(radians, [float(lon_source), float(lat_source), float(lon_obs),
                                                             float(lat_obs)])
    # haversine formula
    dlon = lon_obs - lon_source 
    dlat = lat_obs - lat_source 
    a = sin(dlat/2)**2 + cos(lat_source) * cos(lat_obs) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    dist = c * r
    return dist


def azimuth(lon_source, lat_source, lon_obs, lat_obs):
    az = degrees(atan2((float(lon_source) - float(lon_obs)), (float(lat_source) - float(lat_obs))))
    if az < 0:
        az = az + 360
    if az > 337.5:
        arah = 'utara'
    elif az > 292.5:
        arah = 'baratlaut'
    elif az > 247.5:
        arah = 'barat'
    elif az > 202.5:
        arah = 'baratdaya'
    elif az > 157.5:
        arah = 'selatan'
    elif az > 112.5:
        arah = 'tenggara'
    elif az > 67.5:
        arah = 'timur'
    elif az > 22.5:
        arah = 'timurlaut'
    else:
        arah = 'utara'
    # print("azimuth = %.2f, arah = %s" %(az, arah))
    return az, arah


def gridarea(x1, x2, y1, y2, dx, dy):
    x = np.arange(x1, x2, dx)
    y = np.arange(y1, y2, dy)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (xx.size, 1))
    yy = np.reshape(yy, (yy.size, 1))
    m_size = np.array([xx.size, yy.size])
    return xx, yy, m_size


def find_nearest_dist(source, obs):
    idx = np.zeros(len(obs))
    # obs = np.zeros(len(value))
    jarak = np.zeros((len(source), len(obs)))
    for j in range(len(idx)):
        for i in range(len(source)):
            jarak[i, j] = haversine(source[i, 0], source[i, 1], obs[j, 0], obs[j, 1])
        idx[j] = np.argmin(jarak[:, j])
    return idx


def find_nearest_val(data, value):
    array = np.asarray(data)
    idx = (np.abs(array - value)).argmin()
    return idx


def azimuth_gap(list_azimuth):
    """
    :param list_azimuth: list of azimuth data
    :return: max_azimuth gap, azimuth start_gap, azimuth end_gap (clockwise)
    """
    list_azimuth.sort()
    lst_gap = []
    for i in range(len(list_azimuth)):
        if i == 0:
            lst_gap.append(360 - list_azimuth[-1] + list_azimuth[0])
        elif i < len(list_azimuth):
            lst_gap.append(list_azimuth[i] - list_azimuth[i-1])
    idx_gapend = lst_gap.index(max(lst_gap))

    return max(lst_gap), list_azimuth[idx_gapend-1], list_azimuth[idx_gapend]


def divide_sector_edge(start_azimuth, end_azimuth, quadrant_num):
    """
    :param start_azimuth: azimuth start_gap (clockwise)
    :param end_azimuth: azimuth end_gap (clockwise)
    :param quadrant_num: number of divided quadrant
    :return: list of sector azimuth center
    """

    if end_azimuth > start_azimuth:
        az_length = 360 - end_azimuth + start_azimuth
    else:
        az_length = start_azimuth - end_azimuth

    quadrant_edge = []
    quadrant_range = az_length / (quadrant_num-1)
    for i in range(quadrant_num + 1):
        if i == 0:
            q_edge = end_azimuth - (quadrant_range/2)
        # elif i == quadrant_num:

        else:
            q_edge = end_azimuth + quadrant_range*(i-1) + quadrant_range/2
        if q_edge > 360:
            q_edge -= 360
        quadrant_edge.append(q_edge)

    return quadrant_edge


def divide_sector_center(start_azimuth, end_azimuth, quadrant_num):
    """
    :param start_azimuth: azimuth start_gap (clockwise)
    :param end_azimuth: azimuth end_gap (clockwise)
    :param quadrant_num: number of divided quadrant
    :return: list of sector start and end range
    """

    if end_azimuth > start_azimuth:
        az_length = 360 - end_azimuth + start_azimuth
    else:
        az_length = start_azimuth - end_azimuth

    quadrant_center = []
    quadrant_range = az_length / (quadrant_num-1)
    for i in range(quadrant_num):
        q_center = end_azimuth + quadrant_range * i

        if q_center > 360:
            q_center -= 360

        quadrant_center.append(q_center)

    return quadrant_center
