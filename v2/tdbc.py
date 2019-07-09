import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import gps_utils as gps
import pandas as pd


def stop_point_extraction(trajectory, time_threshold, distance_threshold):
    """
    Calculate stop points of one trajectory.

    The Time Distance Based Clustering algorithm is used to extract stop points.
    The TDBC forms the GPS Point clusters.

    Parameters
    ----------
    trajectory : DataFrame
        DataFrame of one trajectory
    time_threshold : int
        Time Threshold (δθ).
    distance_threshold : int
        Distance Threshold (δd).

    Returns
    -------
    stop_points, stop_points_cluster : tuple of (DataFrame, DataFrame)
        stop_points is a DataFrame containing all Points that have been merged into one Stop Point.
        stop_points_cluster contains cluster of Stop Points as a single Point.
    """
    # empty cluster, c_cluster := current cluster, p_cluster := previous cluster
    c_cluster = pd.DataFrame(columns=['lon', 'lat', 'time'])
    p_cluster = pd.DataFrame(columns=['lon', 'lat', 'time'])
    stop_points = pd.DataFrame(columns=['lon', 'lat', 'time'])
    stop_points_cluster = pd.DataFrame(
        columns=['lon', 'lat', 'tstart', 'tend'])
    c_point = None
    # Boolean Flag only relevant to the check() function
    is_type2 = False

    def start_end(cluster):
        """Calculate the start and end time of one cluster.

        Parameters
        ----------
        cluster : DataFrame
            DataFrame containing a Cluster C.

        Returns
        -------
        min_t, max_t : tuple of (int, int)
            min_t is the start time
            max_t is the end time
        """
        min_t = min(cluster['time'].tolist())
        max_t = max(cluster['time'].tolist())
        return min_t, max_t

    def add_stop_point(cluster):
        """
        Add a cluster of points to SP if the condition is meet.

        Parameters
        ----------
        cluster : DataFrame
            DataFrame containing a Cluster C.
        """
        nonlocal p_cluster
        nonlocal stop_points
        nonlocal stop_points_cluster
        nonlocal c_cluster

        min_time, max_time = start_end(cluster)
        # if there is no SP, add the cluster/point to SP as initial SP
        if len(stop_points.tail(1)['lon'].values) == 0:
            stop_points = stop_points.append(cluster, ignore_index=True)
            sp_cluster_frame = pd.DataFrame({
                "lon": [cluster['lon'][0]],
                "lat": [cluster['lat'][0]],
                "tstart": [cluster['time'][0]]
            })
            stop_points_cluster = stop_points_cluster.append(sp_cluster_frame,
                                                             ignore_index=True)
            p_cluster = cluster
        else:
            # get previous Stop Point in SP
            p_stop_point = stop_points_cluster.tail(
                1)['lon'].values[0], stop_points_cluster.tail(
                    1)['lat'].values[0]
            # calculate the centroid of cluster C
            cluster_center = gps.centroid(cluster['lon'].tolist(),
                                          cluster['lat'].tolist())
            # if (distance(Cluster, Previous stop point in SP) < δd) then
            if gps.haversine(cluster_center,
                             p_stop_point) < distance_threshold:
                # extract long and lat from previous Stop Point in SP
                p_stop_point_lon, p_stop_point_lat = p_stop_point
                test_point_frame = pd.DataFrame({
                    "lon": [p_stop_point_lon],
                    "lat": [p_stop_point_lat],
                    "tstart": [min_time],
                    "tend": [max_time]
                })
                # Previous = merge(Cluster, Previous)
                stop_points = pd.concat(
                    [stop_points,
                     cluster]).drop_duplicates().reset_index(drop=True)
                stop_points_cluster = stop_points_cluster.drop(
                    stop_points_cluster.tail(1).index, inplace=True)
                stop_points_cluster = pd.concat([
                    stop_points_cluster, test_point_frame
                ]).drop_duplicates().reset_index(drop=True)
                # delete current cluster C
                c_cluster = c_cluster.iloc[0:0]
            # if not (distance(Cluster, Previous stop point in SP) < δd) then
            else:
                center_lon, center_lat = cluster_center
                if center_lon != 0:
                    df = {
                        'lon': [center_lon],
                        'lat': [center_lat],
                        'tstart': [min_time],
                        'tend': [max_time]
                    }
                    sp_df = pd.DataFrame(
                        df, columns=['lon', 'lat', 'tstart', 'tend'])
                    # put Cluster in SP
                    stop_points = stop_points.append(cluster,
                                                     ignore_index=True)
                    stop_points_cluster = stop_points_cluster.append(
                        sp_df, ignore_index=True)
                    # Previous = Cluster
                    p_cluster = cluster
                    # delete current cluster C
                    c_cluster = c_cluster.iloc[0:0]

    def check():
        """
        Check whether the previous cluster and the current cluster can be merged.
        
        This strategy is used so two nearby SP can be merged into one SP.
        """
        nonlocal c_cluster
        nonlocal p_cluster
        nonlocal is_type2
        # calculate the centroid of c_cluster and p_cluster
        c_cluster_coord = gps.centroid(c_cluster['lon'].tolist(),
                                       c_cluster['lat'].tolist())
        p_cluster_coord = gps.centroid(p_cluster['lon'].tolist(),
                                       p_cluster['lat'].tolist())
        # if (time interval(Cluster, Previous) < δt and distance(Cluster, Previous) < δd) then
        if (time_interval() < time_threshold and gps.haversine(
                c_cluster_coord, p_cluster_coord) < distance_threshold):
            # Previous = merge(Cluster, Previous);
            p_cluster = p_cluster.append(
                c_cluster).drop_duplicates().reset_index(drop=True)
            # if Previous is one of type II then SP.add(Previous)
            if is_type2:
                add_stop_point(p_cluster)
                is_type2 = False
            # else Previous = Cluster
            else:
                p_cluster = c_cluster
        else:
            add_stop_point(c_cluster)

    def time_interval():
        """
        Calculate the time interval of p_cluster and c_cluster.

        Notes
        ----------
        The time interval is e.g.:
        p_cluster: time = [30, 45, 60]
        c_cluster: time = [75, 90, 105]
        time interval = 105 - 30 = 75
        """
        nonlocal c_cluster
        nonlocal p_cluster
        # if p_cluster return 0
        if p_cluster.shape[0] == 0:
            return 0

        min_t = min([
            c_cluster.iloc[[0]]['time'].tolist()[0],
            p_cluster.iloc[[0]]['time'].tolist()[0]
        ])
        max_t = max([
            c_cluster.iloc[[-1]]['time'].tolist()[0],
            p_cluster.iloc[[-1]]['time'].tolist()[0]
        ])

        return abs(max_t - min_t)

    def duration():
        """
        Calculate the duration of the current cluster.

        Notes
        ----------
        The time duration is e.g.:
        c_cluster: time = [75, 90, 105]
        duration = 105 - 75 = 30
        """
        nonlocal c_cluster

        min_t = min(c_cluster['time'].tolist())
        max_t = max(c_cluster['time'].tolist())

        return max_t - min_t

    # TDBC start here: initialize variables
    p_coord = None
    skip = False
    # if the first is type I then SP.add(the point)
    first_sp_lon = trajectory.iloc[0].values[0]
    first_sp_lat = trajectory.iloc[0].values[1]
    first_sp_frame = pd.DataFrame({
        "lon": [first_sp_lon],
        "lat": [first_sp_lat],
        "time": [0]
    })
    add_stop_point(first_sp_frame)

    # for each point Pi in T do
    for index, point in trajectory.iterrows():
        c_point = point
        point_coord = point['lon'], point['lat']
        point_frame = pd.DataFrame({
            "lon": [point['lon']],
            "lat": [point['lat']],
            "time": [point['time']]
        })

        cluster_coord = gps.centroid(c_cluster['lon'].tolist(),
                                     c_cluster['lat'].tolist())
        # add point to c_cluster if distance criteria is meet / initial cluster
        if c_cluster.shape[0] == 0:
            if p_coord is not None and gps.haversine(
                    p_coord, point_coord) < distance_threshold:
                p_p_lon, p_p_lat = p_coord
                p_point_frame = pd.DataFrame({
                    "lon": [p_p_lon],
                    "lat": [p_p_lat],
                    "time": [point['time'] - 15]
                })
                c_cluster = c_cluster.append(p_point_frame)
                c_cluster = c_cluster.append(point_frame)
                cluster_coord = gps.centroid(c_cluster['lon'].tolist(),
                                             c_cluster['lat'].tolist())
            else:
                # skip is used so there is previous point / cluster
                skip = True

        else:
            skip = False
            if gps.haversine(cluster_coord, point_coord) < distance_threshold:
                c_cluster = c_cluster.append(point_frame)
                cluster_coord = gps.centroid(c_cluster['lon'].tolist(),
                                             c_cluster['lat'].tolist())
                is_type2 = True
        if not skip:
            # type II Stop Point
            if c_cluster.shape[0] != 0 and gps.haversine(
                    cluster_coord, point_coord
            ) > distance_threshold and duration() > time_threshold:
                add_stop_point(c_cluster)
                is_type2 = True
            if c_cluster.shape[0] != 0 and gps.haversine(
                    cluster_coord, point_coord
            ) > distance_threshold and duration() < time_threshold:
                check()
                is_type2 = True
            # type III Stop Point, currently not used
            if p_coord is not None and gps.haversine(
                    p_coord,
                    point_coord) < distance_threshold and 15 > time_threshold:
                is_type2 = False
            if p_coord is not None and gps.haversine(
                    p_coord,
                    point_coord) > distance_threshold and 15 > time_threshold:
                is_type2 = False
                pass

        # p_coord is set
        p_coord = point['lon'], point['lat']

    # add the last point or cluster to SP as a type I Stop Point
    if c_cluster.shape[0] == 0:
        p_frame = {
            'lon': c_point['lon'],
            'lat': c_point['lat'],
            'time': c_point['time']
        }
        spc_frame = {
            "lon": c_point['lon'],
            "lat": c_point['lat'],
            "tstart": c_point['time']
        }
        stop_points = stop_points.append(p_frame, ignore_index=True)
        stop_points_cluster = stop_points_cluster.append(spc_frame,
                                                         ignore_index=True)
    else:
        cluster_coord = gps.centroid(c_cluster['lon'].tolist(),
                                     c_cluster['lat'].tolist())
        centroid_lon, centroid_lat = cluster_coord
        min_time, max_time = start_end(c_cluster)
        data = {
            'lon': [centroid_lon],
            'lat': [centroid_lat],
            'tstart': [min_time],
            'tend': [max_time]
        }
        sp_frame = pd.DataFrame(data, columns=['lon', 'lat', 'tstart', 'tend'])
        stop_points_cluster = stop_points_cluster.append(sp_frame,
                                                         ignore_index=True)
        stop_points = stop_points.append(c_cluster, ignore_index=True)

    return stop_points, stop_points_cluster
