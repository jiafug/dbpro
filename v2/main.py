import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import csv
import math
# import mplleaflet
import matplotlib.pyplot as plt
import pandas as pd
import gps_utils as gps
import tdbc
import time

current_milli_time = lambda: int(round(time.time() * 1000))
start_time = 0
trajectory_count = 0
point_count = 0
stop_count = 0
simple_count = 0
header = False


def main():
    global trajectory_count
    global point_count
    global stop_count
    global simple_count
    global simple_count
    global start_time

    start_time = current_milli_time()

    with open('C:/Users/Schatzinator/Documents/TU Berlin/DBPRO/Dataset/test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        latitude_list = []
        longitude_list = []
        time_list = []
        # breaks only for development purposes
        time = 0

        # for development only
        # test ist eine Liste mit allen Trajectories, jeder Traejctory ist ein (Pandas-) Dataframe mit den Spalten lon, lat, time
        test = []

        for row in csv_reader:
            cleared = row[8].replace("],[", " -1 ").replace("[[", "").replace("]]", "")
            if cleared != "POLYLINE":
                trajectory_count += 1
                splitted = cleared.split(" -1 ")
                # extractes all the points
                for entry in splitted:
                    data = entry.split(",")
                    latitude_list.append(data[1])
                    longitude_list.append(data[0])
                    time_list.append(time)
                    time += 15
                    point_count += 1
                time = 0
                # creats a trajectory dataframe
                coords = pd.DataFrame({'lon': longitude_list, 'lat': latitude_list, 'time': time_list})
                coords.lon = coords.lon.astype(float)
                coords.lat = coords.lat.astype(float)

                # here to insert code

                time_threshold = 500
                distance_threshold = 60
                stop_points, stop_points_cluster = tdbc.stop_point_extraction(coords, time_threshold,
                                                                              distance_threshold)

                new = coords.merge(stop_points, on=['lon', 'lat'], how='left')
                new = new[new.time_y.isnull()]
                new = new.rename(columns={"time_x": "tstart", "time_y": "tend"})
                route = stop_points_cluster.append(new)
                route = route.sort_values(by=['tstart']).reset_index(drop=True)

                stop_count += route.shape[0]

                merged = data_simplification(route)

                simple_count += merged.shape[0]

                write_to_csv(merged)

                # here to insert code ends

                # test.append(coords)
                # reset everything 
                latitude_list = []
                longitude_list = []
                time_list = []
                # breaks only for development purposes
                '''
                if counter == 6:
                    break
                '''
    statistics()


def data_simplification(route):
    part = []
    simplified_coords = pd.DataFrame(columns=['lon', 'lat', 'tstart', 'tend'])
    counter = 0
    for index, point in route.iterrows():
        counter += 1
        current_point = [point['lon'], point['lat']]
        part.append(current_point)
        if (not math.isnan(point['tend']) and len(part) > 1) or (route.shape[0] == counter):
            l_data = gps.rdp(part, 0.00025)
            for i in l_data:
                l_lon = i[0]
                l_lat = i[1]
                l_frame = pd.DataFrame({"lon": [l_lon], "lat": [l_lat]})
                simplified_coords = simplified_coords.append(l_frame)
            part = []
            part.append(current_point)
    # print(simplified_coords)
    simplified_coords = simplified_coords.drop_duplicates(subset=None, keep='first', inplace=False).reset_index(
        drop=True)
    # print(simplified_coords)
    merged = route.merge(simplified_coords, on=['lon', 'lat'])
    merged = merged.drop(['tstart_y', 'tend_y'], axis=1)
    merged = merged.rename(columns={'tstart_x': 'tstart', 'tend_x': 'tend'})
    # print(merged)
    # print("{} gps points simplified to {} points".format(route.shape[0], simplified_coords.shape[0]))
    return merged


def write_to_csv(merged):
    global header
    p_point = pd.Series([])
    p_point_e = False
    lts = pd.DataFrame(
        columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing'])
    for index, point in merged.iterrows():
        if p_point_e == False:
            pass
        else:
            # # print(point)
            brng = gps.bearingCalculator((p_point['lon'], p_point['lat']), (point['lon'], point['lat']))
            dis = gps.haversine((p_point['lon'], p_point['lat']), (point['lon'], point['lat']))
            lts_frame = pd.DataFrame(
                {'lon1': [p_point['lon']], 'lat1': [p_point['lat']], 'tstart1': [p_point['tstart']],
                 'tend1': [p_point['tend']],
                 'lon2': [point['lon']], 'lat2': [point['lat']], 'tstart2': [point['tstart']], 'tend2': [point['tend']],
                 'distance': [dis], 'bearing': [brng]})
            lts = lts.append(lts_frame)
            # print(p_point, point)
            # print(brng, dis)
        p_point = point
        p_point_e = True

    lts = lts.reset_index(drop=True)
    # print(lts)
    if header == False:
        header = True
        lts.to_csv('test_lts.csv', header=True, sep=';', mode='w')  # header = 'False', index = 'True')
    else:
        lts.to_csv('test_lts.csv', header=False, sep=';', mode='a')  # header = 'False', index = 'True')


def statistics():
    time = (current_milli_time() - start_time) / 1000
    compression = simple_count / point_count
    print("---------- statistics of data simplification ----------")
    print("total trajectories processed: {}".format(trajectory_count))
    print("total points processed {}".format(point_count))
    print("total points after stop point extraction: {}".format(stop_count))
    print("total points after data simplification: {}".format(simple_count))
    print("points compression rate: {}".format(compression))
    print("time consumption in s: {}".format(time))
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
