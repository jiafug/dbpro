import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import pandas as pd


def haversine(coord1, coord2):
    """Calculate the Haversine distance in meters.

    The function determines the distance between two points on a sphere given
    their longitudes and latitudes.

    Parameters
    ----------
    coord1 : tuple of (float, float)
        Tuple (longitude, latitude)
    coord2 : tuple of (float, float)
        Tuple (longitude, latitude)

    Returns
    -------
    d : float
        The Haversine distance in meters.

    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    # mean earth radius in meters (GRS 80-Ellipsoid)
    radius = 6371000

    # haversine formula
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = radius * c
    return d


def centroid(lonT, latT):
    """Calculate the centroid of a cluster of points.
    
    The input has to be two lists, one containing longitude values
    and the other latitude values in their right order.

    Parameters
    ----------
    lonT : list of float
        List of longitude values.
    latT : list of float
        List of latitude values.

    Returns
    -------
    (Lon, Lat) : tuple of (float, float)
        Tuple of one longitude and latitude value (the centroid).

    """
    # exceptional case
    if len(lonT) == 0 or len(latT) == 0:
        return 0, 0

    x_list = []
    y_list = []
    z_list = []

    dataT = pd.DataFrame({'lat': latT, 'lon': lonT})
    for index, row in dataT.iterrows():
        lat = row['lat'] * math.pi / 180
        lon = row['lon'] * math.pi / 180
        X = math.cos(lat) * math.cos(lon)
        Y = math.cos(lat) * math.sin(lon)
        Z = math.sin(lat)
        x_list.append(X)
        y_list.append(Y)
        z_list.append(Z)

    dataXYZ = pd.DataFrame({'x': x_list, 'y': y_list, 'z': z_list})

    x = 0
    y = 0
    z = 0
    n = 0

    for index, row in dataXYZ.iterrows():
        x += row['x']
        y += row['y']
        z += row['z']
        n += 1

    Lon = math.atan2(y, x) * 180 / math.pi
    Hyp = math.sqrt(x * x + y * y)
    Lat = math.atan2(z, Hyp) * 180 / math.pi

    return Lon, Lat


def bearingCalculator(point1, point2):
    """Calculate the bearing of two points.

    Bearing is the horizontal angle between the direction of one point and another
    point.

    Parameters
    ----------
    point1 : tuple of (float, float)
        Tuple (latitude, longitude)
    point2 : tuple of (float, float)
        Tuple (latitude, longitude)

    Returns
    -------
    bearing : float
        Bearing as degree.

    """
    lat1, lon1 = point1
    lat2, lon2 = point2

    dLon = (lon2 - lon1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2) * math.cos(dLon)

    brng = math.degrees(math.atan2(y, x))

    return 360 - (brng + 360) % 360
