# -*- coding: utf-8 -*-
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import numpy as np
import pandas as pd


# RDP algorithm as long as the rdp package is not iterative.
# See https://github.com/fhirschmann/rdp/issues/5
def _DouglasPeucker(points, startIndex, lastIndex, epsilon):
    stk = []
    stk.append([startIndex, lastIndex])
    globalStartIndex = startIndex
    bitArray = np.ones(lastIndex - startIndex + 1, dtype=bool)

    while len(stk) > 0:
        startIndex = stk[-1][0]
        lastIndex = stk[-1][1]
        stk.pop()

        dmax = 0.
        index = startIndex

        for i in range(index + 1, lastIndex):
            if bitArray[i - globalStartIndex]:
                d = PointLineDistance(points[i], points[startIndex], points[lastIndex])
                if d > dmax:
                    index = i
                    dmax = d
        if dmax > epsilon:
            stk.append([startIndex, index])
            stk.append([index, lastIndex])
        else:
            for i in range(startIndex + 1, lastIndex):
                bitArray[i - globalStartIndex] = False
    return bitArray


def rdp(points, epsilon):
    """
    Ramer-Douglas-Peucker algorithm
    """
    bitArray = _DouglasPeucker(points, 0, len(points) - 1, epsilon)
    resList = []
    for i in range(len(points)):
        if bitArray[i]:
            resList.append(points[i])
    return np.array(resList)


def PointLineDistance(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point, start)
    n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
    d = math.sqrt((end[0] - start[0]) * (end[0] - start[0]) + (end[1] - start[1]) * (end[1] - start[1]))
    return n / d


def haversine(coord1, coord2):
    """
    Haversine distance in meters for two (lat, lon) coordinates
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    radius = 6371000  # mean earth radius in meters (GRS 80-Ellipsoid)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


def centroid(lonT, latT):
    # https://stackoverflow.com/questions/6671183/calculate-the-center-point-of-multiple-latitude-longitude-coordinate-pairs

    # lonT = [13.341664, 13.374373, 13.376003, 13.352830]
    # latT = [52.519198, 52.523039, 52.504053, 52.508498]

    xList = []
    yList = []
    zList = []

    dataT = pd.DataFrame({'lat': latT, 'lon': lonT})
    for index, row in dataT.iterrows():
        lat = row['lat'] * math.pi / 180
        lon = row['lon'] * math.pi / 180
        X = math.cos(lat) * math.cos(lon)
        Y = math.cos(lat) * math.sin(lon)
        Z = math.sin(lat)
        xList.append(X)
        yList.append(Y)
        zList.append(Z)

    dataXYZ = pd.DataFrame({'x': xList, 'y': yList, 'z': zList})

    x = 0
    y = 0
    z = 0
    n = 0

    for index, row in dataXYZ.iterrows():
        x += row['x']
        y += row['y']
        z += row['z']
        n += 1

    aX = x / n
    aY = y / n
    aZ = z / n

    Lon = math.atan2(y, x) * 180 / math.pi
    Hyp = math.sqrt(x * x + y * y)
    Lat = math.atan2(z, Hyp) * 180 / math.pi

    return Lon, Lat


def bearingCalculator(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    dLon = (lon2 - lon1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))
    return (360 - (brng + 360) % 360)
    # return math.asin(784.90/704.64)    # len (point1, point2) / len(point2, (lat1, lon2))
