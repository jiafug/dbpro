import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import gps_utils as gps
import pandas as pd


def centroid(lonT, latT):
    # https://stackoverflow.com/questions/6671183/calculate-the-center-point-of-multiple-latitude-longitude-coordinate-pairs

    # lonT = [13.341664, 13.374373, 13.376003, 13.352830]
    # latT = [52.519198, 52.523039, 52.504053, 52.508498]

    if (len(lonT) == 0 or len(latT) == 0):
        return 0, 0

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


def stop_point_extraction(trajectory, time_threshold, distance_threshold):
    # leere Cluster, c_cluster ist Cluster aus dem Paper, p_cluster ist Previous C
    c_cluster = pd.DataFrame(columns=['lon', 'lat', 'time'])
    p_cluster = pd.DataFrame(columns=['lon', 'lat', 'time'])
    stop_points = pd.DataFrame(columns=['lon', 'lat', 'time'])
    stop_points_cluster = pd.DataFrame(columns=['lon', 'lat', 'tstart', 'tend'])
    # p_point ist Previous stop point in SP aus dem Paper, relevant für Zeile 9 und 11 aus dem Paper
    p_point = None
    c_point = None
    # Boolean Flag, nur relevant für die check Funktion
    is_type2 = False
    # braucht man nicht wirklich
    counter = 0

    def start_end(cluster):
        min_time = min(cluster['time'].tolist())
        max_time = max(cluster['time'].tolist())
        return min_time, max_time

    def add_stop_point(cluster):
        # print("-------------------BEGIN: add_stop_point(cluster)---------------------")
        # nonlocal, damit auf die Variablen der übergeordneten Funktion aufgerufen und bearbeiten werden kann
        nonlocal p_cluster
        nonlocal stop_points
        nonlocal stop_points_cluster
        nonlocal c_cluster
        # Variablen Initialisierung
        p_stop_point = None
        # Nur relevant für die Initailisierung des Algorithmuses: wenn kein Stopppunkt vorhanden,
        # füge das aktuelle Cluster oder Punkt der Liste der Stopppunkte hinzu
        # print("-> cluster:")
        # print(cluster)
        min_time, max_time = start_end(cluster)
        if len(stop_points.tail(1)['lon'].values) == 0:
            stop_points = stop_points.append(cluster, ignore_index=True)
            spc_frame = pd.DataFrame(
                {"lon": [cluster['lon'][0]], "lat": [cluster['lat'][0]], "tstart": [cluster['time'][0]]})
            stop_points_cluster = stop_points_cluster.append(spc_frame, ignore_index=True)
            p_cluster = cluster
            # print("--> initial SP")
            # print("-> stop_points:")
            # print(stop_points)
            # print("-> stop_points_cluster:")
            # print(stop_points_cluster)
        else:
            p_stop_point = stop_points_cluster.tail(1)['lon'].values[0], stop_points_cluster.tail(1)['lat'].values[0]
            cluster_coord = centroid(cluster['lon'].tolist(), cluster['lat'].tolist())
            # gps.haversine(coord1, coord2) berechnet die Distanc in Metern zwischen zwei Koordinaten
            # coord1 und coord2 müssen als Dataframe mit den Spalten lon und lat vorliegen
            # print("-> p_stop_point")
            # print(p_stop_point)
            # print("-> cluster")
            # print(cluster_coord)
            # print("-> distance between cluster and p_stop_point")
            # print(gps.haversine(cluster_coord, p_stop_point))
            if gps.haversine(cluster_coord, p_stop_point) < distance_threshold:
                # hängt alle Punkte des Clusters oder einzelnen Punkt an den p_cluster (vorheriger Cluster)
                p_stop_point_lon, p_stop_point_lat = p_stop_point
                test_point = centroid([p_stop_point_lon], [p_stop_point_lon])
                test_point_frame = pd.DataFrame(
                    {"lon": [p_stop_point_lon], "lat": [p_stop_point_lat], "tstart": [min_time], "tend": [max_time]})
                stop_points = pd.concat([stop_points, cluster]).drop_duplicates().reset_index(drop=True)
                stop_points_cluster = stop_points_cluster.drop(stop_points_cluster.tail(1).index, inplace=True)
                stop_points_cluster = pd.concat([stop_points_cluster, test_point_frame]).drop_duplicates().reset_index(
                    drop=True)
                # print("--> Added c_cluster to p_cluster!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print("-> c_cluster:")
                # print(c_cluster)
                # print("-> p_cluster:")
                # print(p_cluster)
                c_cluster = c_cluster.iloc[0:0]
            else:
                centeroid_lon, centeroid_lat = cluster_coord
                if centeroid_lon != 0:
                    data = {'lon': [centeroid_lon], 'lat': [centeroid_lat], 'tstart': [min_time], 'tend': [max_time]}
                    sp_frame = pd.DataFrame(data, columns=['lon', 'lat', 'tstart', 'tend'])
                    stop_points = stop_points.append(cluster, ignore_index=True)
                    stop_points_cluster = stop_points_cluster.append(sp_frame, ignore_index=True)
                    p_cluster = cluster
                    # Dataframe.iloc[0:0] löscht den Inhalt eines Dataframes, hier den Inhalt von c_cluster
                    c_cluster = c_cluster.iloc[0:0]
                    # print("---> p_cluster = cluster / drop c_cluster")
            # print("-> stop_points:")
            # print(stop_points)
            # print("-> stop_points_cluster:")
            # print(stop_points_cluster)
        # print("-------------------END: add_stop_point(cluster)---------------------")

    def check():
        # print("-------------------BEGIN: check()---------------------")
        nonlocal c_cluster
        nonlocal p_cluster
        nonlocal is_type2
        # centroid(coods, coords) berechnet den Mittelpunkt, Argumente, sind jeweils eine Liste von lon und lat
        c_cluster_coord = centroid(c_cluster['lon'].tolist(), c_cluster['lat'].tolist())
        p_cluster_coord = centroid(p_cluster['lon'].tolist(), p_cluster['lat'].tolist())
        # print("---> BEFORE:")
        # print("-> c_cluster:")
        # print(c_cluster)
        # print("-> p_cluster:")
        # print(p_cluster)
        # print("-> time_interval:")
        # print(time_interval())
        # print("-> distance c_cluster_coord, p_cluster_coord:")
        # print(gps.haversine(c_cluster_coord, p_cluster_coord))
        if (time_interval() < time_threshold and gps.haversine(c_cluster_coord, p_cluster_coord) < distance_threshold):
            # p_cluster wird erweitert mit c_cluster, Duplikate werden gelöscht
            p_cluster = p_cluster.append(c_cluster).drop_duplicates().reset_index(drop=True)
            # print("-> p_cluster")
            # print(p_cluster)
            # print("-> is_type2")
            # print(is_type2)
            if is_type2 == True:
                # print("----> add_stop_point(p_cluster)")
                add_stop_point(p_cluster)
                is_type2 == False
                # c_cluster = c_cluster.iloc[0:0]
            else:
                # print("----> p_cluster = c_cluster / drop c_cluster")
                p_cluster = c_cluster
                # löschen des Inhalts von c_cluster
                # c_cluster = c_cluster.iloc[0:0]
        else:
            # pass
            add_stop_point(c_cluster)
        # print("---> AFTER:")
        # print("-> c_cluster:")
        # print(c_cluster)
        # print("-> p_cluster:")
        # print(p_cluster)
        # print("-------------------END: check()---------------------")

    def time_interval():
        nonlocal c_cluster
        nonlocal p_cluster
        # wenn p_cluster leer, dann return 0 als Zeitinterval
        if p_cluster.shape[0] == 0:
            return 0
        # berechnet den Zeitinterval von zwei Clustern, indem der Maximalwert (Zeit) der beiden Cluster - Minimalwert
        min_time = min([c_cluster.iloc[[0]]['time'].tolist()[0], p_cluster.iloc[[0]]['time'].tolist()[0]])
        max_time = max([c_cluster.iloc[[-1]]['time'].tolist()[0], p_cluster.iloc[[-1]]['time'].tolist()[0]])
        return abs(max_time - min_time)

    def duration():
        nonlocal c_cluster
        min_time = min(c_cluster['time'].tolist())
        max_time = max(c_cluster['time'].tolist())
        return max_time - min_time

    p_coord = None
    skip = False
    # adds the first and last trajectory points as type 1 stop points
    # if counter == 0 or counter == trajectory.shape[0]-1:

    first_sp_lon = trajectory.iloc[0].values[0]
    first_sp_lat = trajectory.iloc[0].values[1]
    first_sp_frame = pd.DataFrame({"lon": [first_sp_lon], "lat": [first_sp_lat], "time": [0]})
    # print(first_sp_frame)
    add_stop_point(first_sp_frame)
    for index, point in trajectory.iterrows():
        c_point = point
        point_coord = point['lon'], point['lat']
        point_frame = pd.DataFrame({"lon": [point['lon']], "lat": [point['lat']], "time": [point['time']]})
        # print("_____________________________________________________________________________")
        # print(point_frame)
        # print("-> p_coord:")
        # print(p_coord)
        # print("-> point_coord:")
        # print(point_coord)
        # if p_coord != None:
        # print("-> distance between p_coord and point_coord:")
        # print(gps.haversine(p_coord, point_coord))
        # Lösungsvorschlag, wenn der initiale Cluster leer ist, dann füge aktuellen Punkt hinzu
        cluster_coord = centroid(c_cluster['lon'].tolist(), c_cluster['lat'].tolist())
        # # print(gps.haversine(cluster_coord, point_coord))
        # # print(cluster_coord, point_coord)
        # Punkte des Typ 2 sind der 1,2,3 Fall
        if c_cluster.shape[0] == 0:
            # print("IF -> Cluster empty")
            if p_coord != None and gps.haversine(p_coord, point_coord) < distance_threshold:
                # print("----->c_cluster empty: put current point in c_cluster")
                p_p_lon, p_p_lat = p_coord
                p_point_frame = pd.DataFrame({"lon": [p_p_lon], "lat": [p_p_lat], "time": [point['time'] - 15]})
                # if (point['time'] - 15) != 0 :
                c_cluster = c_cluster.append(p_point_frame)
                c_cluster = c_cluster.append(point_frame)
                cluster_coord = centroid(c_cluster['lon'].tolist(), c_cluster['lat'].tolist())
            else:
                skip = True
                # print("----->PASS -> p_coord None or distance between p_coord, point_coord < distance_threshold")

        else:
            skip = False
            # print("ELSE -> Cluster not empty and point may be added to cluster")
            if gps.haversine(cluster_coord, point_coord) < distance_threshold:
                # print(">>>case1: put in c_cluster")
                c_cluster = c_cluster.append(point_frame)
                cluster_coord = centroid(c_cluster['lon'].tolist(), c_cluster['lat'].tolist())
                is_type2 = True
        if (skip != True):
            # print("Other Cases:")
            # print("-> distance cluster_coord and point_coord")
            # print(gps.haversine(cluster_coord, point_coord))
            # print("-> duration()")
            # print(duration())
            if c_cluster.shape[0] != 0 and gps.haversine(cluster_coord,
                                                         point_coord) > distance_threshold and duration() > time_threshold:
                # print(">>>case2")
                add_stop_point(c_cluster)
                is_type2 = True
            if c_cluster.shape[0] != 0 and gps.haversine(cluster_coord,
                                                         point_coord) > distance_threshold and duration() < time_threshold:
                # print(">>>case3")
                check()
                is_type2 = True
            if p_coord != None and gps.haversine(p_coord, point_coord) < distance_threshold and 15 > time_threshold:
                # print(">>>case4")
                is_type2 = False

            if p_coord != None and gps.haversine(p_coord, point_coord) > distance_threshold and 15 > time_threshold:
                # print(">>>case5")
                is_type2 = False
                pass
        # print("<<<----------LOOP RESULTS BEGINN---------->>>")
        # print("-> c_cluster")
        # print(c_cluster)
        # print("-> p_cluster")
        # print(p_cluster)
        # print("-> END: stop_points")
        # print(stop_points)
        # print("-> END: stop_points_cluster")
        # print(stop_points_cluster)
        # print("<<<----------LOOP RESULTS END---------->>>")

        counter += 1
        # p_coord (Vorherige Punkt) wird gesetzt
        p_coord = point['lon'], point['lat']
    if c_cluster.shape[0] == 0:
        p_frame = {'lon': c_point['lon'], 'lat': c_point['lat'], 'time': c_point['time']}
        spc_frame = {"lon": c_point['lon'], "lat": c_point['lat'], "tstart": c_point['time']}
        # stop_points_cluster = stop_points_cluster.append(p_frame, ignore_index=True)
        stop_points = stop_points.append(p_frame, ignore_index=True)
        stop_points_cluster = stop_points_cluster.append(spc_frame, ignore_index=True)
    else:
        cluster_coord = centroid(c_cluster['lon'].tolist(), c_cluster['lat'].tolist())
        centeroid_lon, centeroid_lat = cluster_coord
        min_time, max_time = start_end(c_cluster)
        data = {'lon': [centeroid_lon], 'lat': [centeroid_lat], 'tstart': [min_time], 'tend': [max_time]}
        sp_frame = pd.DataFrame(data, columns=['lon', 'lat', 'tstart', 'tend'])
        stop_points_cluster = stop_points_cluster.append(sp_frame, ignore_index=True)
        stop_points = stop_points.append(c_cluster, ignore_index=True)
    # print("ENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDENDEND")
    # print(stop_points_cluster)

    return stop_points, stop_points_cluster
