"""main.py is responsible for hole process of trajectory abstraction.

This process includes trajectory partition, data simplification and
common segment discovery.
trajectory partition -> tdbc.py
data simplification -> this file: main.py
common segment discovery -> csd.py

"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import csv
import math
import pandas as pd
import gps_utils as gps
import tdbc
import time as ttt
from rdp import rdp
import csd

# input variables
DATA_PATH = "../test.csv"
TIME_THRESHOLD = 500
DISTANCE_THRESHOLD = 60
EPSILON = 0.00025

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
current_milli_time = lambda: int(round(ttt.time() * 1000))
trajectory_count = 0
point_count = 0
stop_count = 0
simple_count = 0

# header flag for the csv writer
header = False


def main():
    """Start the application.

    This function is the starting point.
    The data is read in line by line (Trajectory by Trajectory).

    A Trajectory is list of trajectory Points based on their time serials,
    T = P_1, P_2, ..., P_n where P_1.t < P_2.t < ... < P_n.t.
    A Point is P = {lng, lat, t}, represented by longitude, latitude, and time.

    All Points of a Trajectory are passed in a joint data frame for further processing:
    1) Trajectory partition: identify Stop Points, SP = {lng, lat, tstart , tend}
    2) data simplification: simplify a partitioned Trajectory
    3) common segment discovery: invoke csd.py for common segment discovery

    References
    ----------
    [1] Fu, Z., Tian, Z., Xu, Y. and Zhou, K. (2017). Mining Frequent Route
        Patterns Based on Personal Trajectory Abstraction. IEEE Access, 5,
        pp.11352-11363.

    """
    # logging
    global trajectory_count
    global point_count
    global stop_count
    global simple_count
    global simple_count
    global start_time

    # logging
    start_time = current_milli_time()

    with open(DATA_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        latitude_list = []
        longitude_list = []
        time_list = []
        time = 0

        final_lts = pd.DataFrame(columns=[
            'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2',
            'tend2', 'distance', 'bearing', 'route'
        ])

        # logging
        past_time = current_milli_time()
        logger.info("trajectory partition and simplification started...")
        counter = 0

        # read one csv line (Trajectory) after another
        for row in csv_reader:
            cleared = row[8].replace("],[",
                                     " -1 ").replace("[[",
                                                     "").replace("]]", "")
            missing_data = row[7]

            # ignore header
            if cleared != "POLYLINE" and missing_data == "False" and row[
                    8] != "[]":
                trajectory_count += 1
                splitted = cleared.split(" -1 ")

                # logging
                if counter != 0 and counter % 50 == 0:
                    time = (current_milli_time() - past_time) / 1000
                    logger.info(
                        "current trajectories processed: {c} \n processed last 50 lines in {t} s"
                        .format(c=counter, t=time))
                    past_time = current_milli_time()
                    break

                # extract all the Points of a Trajectory
                for entry in splitted:
                    data = entry.split(",")
                    latitude_list.append(data[1])
                    longitude_list.append(data[0])
                    time_list.append(time)
                    time += 15
                    point_count += 1
                time = 0

                if len(latitude_list) > 1:

                    # create a Trajectory data frame
                    coords = pd.DataFrame({
                        'lon': longitude_list,
                        'lat': latitude_list,
                        'time': time_list
                    })
                    coords.lon = coords.lon.astype(float)
                    coords.lat = coords.lat.astype(float)

                    # Trajectory partition (Stop Point extraction)
                    stop_points, stop_points_cluster = tdbc.stop_point_extraction(
                        coords, TIME_THRESHOLD, DISTANCE_THRESHOLD)

                    # build Route data frame
                    new = coords.merge(stop_points,
                                       on=['lon', 'lat'],
                                       how='left')
                    new = new[new.time_y.isnull()]
                    new = new.rename(columns={
                        "time_x": "tstart",
                        "time_y": "tend"
                    })
                    route = stop_points_cluster.append(new)
                    route = route.sort_values(by=['tstart']).reset_index(
                        drop=True)

                    # data simplification
                    merged = data_simplification(route)

                    # logging
                    simple_count += merged.shape[0]
                    stop_count += route.shape[0]

                    # write Route to a new joint data frame
                    final_lts = write_to_df(merged, final_lts, counter)

                    # reset list for next Trajectory
                    latitude_list = []
                    longitude_list = []
                    time_list = []

            # logging
            counter += 1

    # logging
    statistics()
    logger.debug(final_lts)

    # csd
    csd.csd_main(final_lts, logger)


def data_simplification(route):
    """Apply the Douglas-Peuker algorithm to simplify a Route.
    
    The Douglas-Peuker algorithm is an algorithm for curve smoothing.
    It is not applied to an entire Route but from Stop Point to Stop Point.

    Parameters
    ----------
    route : DataFrame
        Dataframe containing a Trajectory Route, including SP.

    Returns
    -------
    merged : DataFrame
        Dataframe of a simplified Route.

    """
    part = []
    simplified_coords = pd.DataFrame(columns=['lon', 'lat', 'tstart', 'tend'])
    counter = 0
    for index, point in route.iterrows():
        counter += 1
        current_point = [point['lon'], point['lat']]
        part.append(current_point)
        # from SP to SP
        if (not math.isnan(point['tend'])
                and len(part) > 1) or (route.shape[0] == counter):
            l_data = rdp(part, EPSILON)
            for i in l_data:
                l_lon = i[0]
                l_lat = i[1]
                l_frame = pd.DataFrame({"lon": [l_lon], "lat": [l_lat]})
                simplified_coords = simplified_coords.append(l_frame)
            part = [current_point]

    simplified_coords = simplified_coords.drop_duplicates(
        subset=None, keep='first', inplace=False).reset_index(drop=True)

    merged = route.merge(simplified_coords, on=['lon', 'lat'])
    merged = merged.drop(['tstart_y', 'tend_y'], axis=1)
    merged = merged.rename(columns={'tstart_x': 'tstart', 'tend_x': 'tend'})

    return merged


def write_to_df(merged, final_lts, counter):
    """Create from Route a Line Temporal Sequence (LTS) and save it in a new data frame.

    In a LTS a Route is described as a sequence of lines:
    LTS = R = L_1, L_2, ..., L_n, where L_i = {P_j, P_k,l,θ}, P_j and P_k are the nodes of R
    after simplification (Douglas-Peuker algorithm),l means the length of L_i (Haversine distance),
    and θ means the angle between L_i and the horizontal axis (bearing).

    Each Route is saved in a data frame in the LTS format for further processing.

    Parameters
    ----------
    merged : DataFrame
        Dataframe containing simplified Routes of one Trajectory.
    final_lts : Dataframe
        Dataframe containing all LTS.
    counter : int
        A counter to count the number of Trajectories.

    Returns
    -------
    final_lts : DataFrame
        DataFrame containing all LTS.

    """
    global header
    p_point = pd.Series([])
    p_point_e = False
    lts = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing'
    ])
    for index, point in merged.iterrows():
        if not p_point_e:
            pass
        else:
            brng = gps.bearingCalculator((p_point['lon'], p_point['lat']),
                                         (point['lon'], point['lat']))
            dis = gps.haversine((p_point['lon'], p_point['lat']),
                                (point['lon'], point['lat']))
            lts_frame = pd.DataFrame({
                'lon1': [p_point['lon']],
                'lat1': [p_point['lat']],
                'tstart1': [p_point['tstart']],
                'tend1': [p_point['tend']],
                'lon2': [point['lon']],
                'lat2': [point['lat']],
                'tstart2': [point['tstart']],
                'tend2': [point['tend']],
                'distance': [dis],
                'bearing': [brng],
                'route': [counter]
            })
            lts = lts.append(lts_frame)
            # print(p_point, point)
            # print(brng, dis)
        p_point = point
        p_point_e = True

    lts = lts.reset_index(drop=True)
    '''
    if not header:
        header = True
        lts.to_csv('test_lts.csv', header=True, sep=';',
                   mode='w')  # header = 'False', index = 'True')
    else:
        lts.to_csv('test_lts.csv', header=False, sep=';',
                   mode='a')  # header = 'False', index = 'True')
    '''
    final_lts = final_lts.append(lts)
    return final_lts


def statistics():
    """Print logging information."""
    time = (current_milli_time() - start_time) / 1000
    compression = simple_count / point_count
    logger.info("""trajectory partition and simplification was completed
 expenditure of time in s: {t}
 total trajectories processed: {tc}
 total points processed: {pc}
 total points after stop point extraction: {spc}
 total points after data simplification: {sp}
 points compression: {c}""".format(tc=trajectory_count,
                                   pc=point_count,
                                   spc=stop_count,
                                   sp=simple_count,
                                   c=compression,
                                   t=time))


if __name__ == "__main__":
    main()
