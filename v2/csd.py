import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import csv
import pandas as pd
import time as ttt
import csd_utils
import numpy as np
from rtree import index
from sklearn.linear_model import LinearRegression
import gps_utils as gps
import gc

# input variables
MIN_LENGTH = 75
EPSILON = 0.0005
MIN_LNS = 1
FILLER = 10000

# logging
logger = None
current_milli_time = lambda: int(round(ttt.time() * 1000))


def csd_main(final_lts, main_logger):

    # logging
    global logger
    logger = main_logger

    # numpy settings
    np.seterr(invalid='ignore', divide='ignore')

    # logging
    logger.info("common segment discovery started...")
    start_time = current_milli_time()

    # import line segments
    line_segments = final_lts[[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route'
    ]]

    line_segments = line_segments.reset_index(drop=True)

    # initialize a r-tree as a index
    rtree = r_tree(line_segments)

    # find all clusters
    ls_clusters = line_segment_clustering(line_segments, rtree)

    # ...
    more_segments(ls_clusters, line_segments)

    # logging statistics
    time = (current_milli_time() - start_time) / 1000
    logger.info('''common segment discovery was completed:
 expenditure of time in s: {t}
 total line segments processed: {ls}
 total common segment clusters: {c} 
 total line segments that were marked as noise: {n}'''.format(
        t=time,
        ls=len(line_segments.index),
        c=len(ls_clusters),
        n=(line_segments.classified == "0").sum()))

    # write LTS results to new csv
    write_to_csv(line_segments)


def csd_import(final_lts):
    """
    Import LTS and saves them in a DataFrame.

    Import line segments (LTS) form the file: 'test_lts.csv' and saves them in a 
    DataFrame if the distance criteria of minnum 25m is meet.
    If the length of a line segment (LTS) is shorter than 25m it will be ignored.
    """
    line_segments = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route'
    ])

    logger.info("lts.csv import started...")
    start_time = current_milli_time()

    for row in final_lts.iterrows():
        '''
        # test break
        if route_id == 20:
            break
        '''
        route_id: int = 0
        if row[0] == 0:
            route_id += 1
        # short line segment filter
        if row[1]['distance'] >= MIN_LENGTH:
            df = {
                'lon1': row[1]['lon1'],
                'lat1': row[1]['lat1'],
                'tstart1': row[1]['tstart1'],
                'tend1': row[1]['tend1'],
                'lon2': row[1]['lon2'],
                'lat2': row[1]['lat2'],
                'tstart2': row[1]['tstart2'],
                'tend2': row[1]['tend2'],
                'distance': row[1]['distance'],
                'bearing': row[1]['bearing'],
                'route': route_id
            }
            line_segments = line_segments.append(df, ignore_index=True)
    time = (current_milli_time() - start_time) / 1000
    logger.info("lts.csv was imported in {} s".format(time))

    # garbage collection
    gc.collect()

    return line_segments


def r_tree(line_segments):
    """
    Generate a R-Tree as a index structure for spatial searching.

    Parameters
    ----------
    line_segments : DataFrame
        DataFrame with all line segments (LTS).

    Returns
    -------
    idx : rtree.Index
        R-Tree as a a index structure.
    
    """
    # logging
    logger.info("rtree index build started...")
    start_time = current_milli_time()

    idx = index.Index()

    for entry in line_segments.iterrows():
        ls_id = entry[0]
        lon1, lat1 = entry[1]['lon1'], entry[1]['lat1']
        lon2, lat2 = entry[1]['lon2'], entry[1]['lat2']
        min_x = min(lon1, lon2) 
        min_y = min(lat1, lat2) 
        max_x = max(lon1, lon2) 
        max_y = max(lat1, lat2) 
        idx.insert(ls_id, (min_x, min_y, max_x, max_y))

    # logging
    time = (current_milli_time() - start_time) / 1000
    logger.info("rtree index was built in {} s".format(time))

    return idx


def line_segment_clustering(line_segments, rtree):
    """
    Cluster all line segments (LTS).

    Step one of TRACLUS line segment clustering algorithm.

    Note
    ----
    The following classified attributes a line segment (LTS) may have:
    -1 := unclassified
    0 := noise
    >0 := line segment cluster id

    Parameters
    ----------
    line_segments : DataFrame
        DataFrame with all line segments (LTS).
        
    rtree : rtree.Index
        R-Tree as a a index structure.
        
    Returns
    -------
    clusters : list of DataFrame
        List of DataFrame for each line segment cluster which contains all related lines.
    
    """
    # logging
    past_time = current_milli_time()
    logger.info("TRACLUS: line segment clustering started...")
    counter = 0

    # Set clusterId to be 0; /* an initial id */
    cluster_id = 1
    line_segments['classified'] = -1
    neighbors = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])
    queue = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])
    clusters = []

    for entry in line_segments.iterrows():

        # logging
        if counter != 0 and counter == 1 or counter == 5 or counter == 20 or counter == 100 or counter % 500 == 0:
            time = (current_milli_time() - past_time) / 1000
            logger.info(
                "current line segments processed: {c} \n processed last 500 lines in {t} s"
                .format(c=counter, t=time))
            past_time = current_milli_time()

        entry = (entry[0], line_segments.loc[[entry[0]]].iloc[0])
        is_classified = entry[1]['classified']
        if is_classified == -1:
            # Compute Nε(L)
            neighbors = neighbors.append(entry[1])
            neighbors = neighbors.append(
                neighborhood(line_segments, entry, False, rtree))
            neighbor_count = neighbors.shape[0]

            # debugging
            logger.debug("neighbors: {n} / {c}".format(n=neighbor_count,
                                                       c=counter))

            if neighbor_count >= MIN_LNS:
                # Assign clusterId to ∀X ∈ Nε(L)
                neighbors['classified'] = cluster_id
                # line_segments.set_value(entry[0], 'classified' , cluster_id)
                line_segments.at[entry[0], 'classified'] = cluster_id
                # Insert Nε(L) − {L} into the queue Q
                queue = queue.append(neighbors.tail(neighbor_count - 1))
                # Step 2
                neighbors = neighbors.append(
                    expand_cluster(line_segments, queue, cluster_id, rtree))
                # logger.debug(neighbors)
                for index in neighbors.index.values.tolist():
                    line_segments.set_value(index, 'classified', cluster_id)
                neighbors.drop_duplicates(keep='first', inplace=True)
                clusters.append(neighbors)
                # Increase clusterId by 1; a new id
                cluster_id += 1
            else:
                # Mark L as noise
                line_segments.at[entry[0], 'classified'] = 0

        neighbors = neighbors.iloc[0:0]
        queue = queue.iloc[0:0]

        # logging
        counter += 1

    return clusters


def neighborhood(line_segments, line, extended, rtree):
    """
    Compute the neighborhood (Nε(L)) of a line segment (LTS).

    Parameters
    ----------
    line_segments : DataFrame
        DataFrame with all line segments (LTS).
        
    line : tuple of (int, pandas.Series)
        Tuple with int as line segments (LTS) index id and pandas.Series with line data.
        
    extended : bool
        True if it is an initial neighborhood search 
        False if it is an expanded neighborhood search.
        
    rtree : rtree.Index
        R-Tree as a a index structure.

    Returns
    -------
    neighbors : DataFrame
        DataFrame with all neighbor line segments (LTS).
    
    """
    neighbors = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])
    n_candidates = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])

    # r tree index
    lon1, lat1 = line[1]['lon1'], line[1]['lat1']
    lon2, lat2 = line[1]['lon2'], line[1]['lat2']
    min_x = min(lon1, lon2)
    max_x = max(lon1, lon2) 
    min_y = min(lat1, lat2) 
    max_y = max(lat1, lat2) 

    n_candidates_ids = list(rtree.intersection((min_x, min_y, max_x, max_y)))
    n_candidates = line_segments.iloc[n_candidates_ids, :]
    logger.debug("n_candidates: {}".format(len(n_candidates)))
    for entry in n_candidates.iterrows():
        is_classified = entry[1]['classified']

        if (not extended and is_classified <= 0) or extended:
            if not line[1].equals(entry[1]):
                line_longer, line_shorter = csd_utils.longer_and_shorter_lines(
                    line[1], entry[1])
                line_a = {
                    'si': [line_longer[0], line_longer[1]],
                    'ei': [line_longer[4], line_longer[5]]
                }
                line_b = {
                    'sj': [line_shorter[0], line_shorter[1]],
                    'ej': [line_shorter[4], line_shorter[5]]
                }
                if csd_utils.distance_functions(line_a, line_b) < EPSILON:
                    neighbors = neighbors.append(entry[1])

    return neighbors


def expand_cluster(line_segments, queue, cluster_id, rtree):
    """
    Find and expand neighborhood for each line segment (LTS) in queue.

    Step 2: compute a density-connected set.

    Parameters
    ----------
    line_segments : DataFrame
        DataFrame with all line segments (LTS).
        
    queue : DataFrame
        DataFrame which contains the neighbors of a line segment (LTS).
        
    cluster_id : int
        Integer which represents the ID of the segment cluster.
        
    rtree : rtree.Index
        R-Tree as a a index structure.

    Returns
    -------
    clusters : list of DataFrame
        List of DataFrame; every DataFrame contains all segment lines of each segment cluster. 
    
    """
    cluster = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])
    logger.debug("segment_id: {id} / queue_size: {s}".format(id=cluster_id,
                                                             s=queue.shape[0]))
    i = 0
    limit = queue.shape[0] * 10
    while queue.shape[0] != 0 and i <= limit:
        
        i += 1

        neighbors = pd.DataFrame(columns=[
            'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2',
            'tend2', 'distance', 'bearing', 'route', 'classified'
        ])
        for entry in queue.iterrows():
            # Let M be the first line segment in Q
            m = entry
            break
        # Compute Nε(M);
        neighbors = neighbors.append(
            neighborhood(line_segments, m, True, rtree))
        neighbor_count = neighbors.shape[0] + 1
        # if (|Nε(M)| ≥ MinLns) then
        if neighbor_count >= MIN_LNS:
            for xn in neighbors.iterrows():
                # if (X is unclassified or noise) then
                if xn[1]['classified'] == -1 or xn[1]['classified'] == 0:
                    # line_segments.set_value(xn[0], 'classified', cluster_id)
                    # Assign clusterId to X
                    line_segments.at[xn[0], 'classified'] = cluster_id
                    series = pd.Series({
                        'lon1': float(xn[1][0]),
                        'lat1': float(xn[1][1]),
                        'tstart1': float(xn[1][2]),
                        'tend1': xn[1][3],
                        'lon2': float(xn[1][4]),
                        'lat2': float(xn[1][5]),
                        'tstart2': float(xn[1][6]),
                        'tend2': xn[1][7],
                        'distance': xn[1][8],
                        'bearing': xn[1][9],
                        'route': int(xn[1][10]),
                        'classified': cluster_id
                    })
                    series.name = xn[0]
                    cluster = cluster.append(series)
                # if (X is unclassified) then
                if xn[1]['classified'] == -1:
                    # Insert X into the queue Q
                    series = pd.Series({
                        'lon1': float(xn[1][0]),
                        'lat1': float(xn[1][1]),
                        'tstart1': float(xn[1][2]),
                        'tend1': xn[1][3],
                        'lon2': float(xn[1][4]),
                        'lat2': float(xn[1][5]),
                        'tstart2': float(xn[1][6]),
                        'tend2': xn[1][7],
                        'distance': xn[1][8],
                        'bearing': xn[1][9],
                        'route': int(xn[1][10]),
                        'classified': cluster_id
                    })
                    series.name = xn[0]
                    queue = queue.append(series)
        # Remove M from the queue Q
        queue = queue.iloc[1:]

    return cluster


def more_segments(clusters, line_segments):
    for entry in clusters:
        new_lg = consecutive_lines_connecting(entry)
        if new_lg.iloc[0]['distance'] == 0:
            logger.info("distance == 0 ---------------------------------------------")
        else:
            new_lg, line_segments = check_for_90(new_lg, line_segments)
            a, b, c = represent_line(new_lg)
            projection_lg, projection_points_list = line_projection(
                a, b, c, new_lg)
            final_projection_points, test_list = projection_points(
                projection_points_list, a, b, c)
            test_lg = form_sts(final_projection_points, test_list)
            new_lg = test(projection_lg, test_lg, new_lg)
            line_segments['classified'] = line_segments['classified'].apply(
                lambda x: str(x))
            update(new_lg, line_segments)
            seg_id = new_lg.iloc[0]['classified']
            write_representative_trajectory(test_lg, seg_id)


def check_for_90(new_lg, line_segments):

    for line in new_lg.iterrows():
        bias = 0.000001
        point1 = line[1]['lon1'],line[1]['lat1']
        point2 = line[1]['lon2'],line[1]['lat2']
        # logger.info("BLABLA DOAPKODAL: {}".format(round(gps.bearingCalculator(point1, point2))))
        if round(gps.bearingCalculator(point1, point2)) == 90 or round(gps.bearingCalculator(point1, point2)) == 270:
            new_list = line[1]['segments']
            point1 = point1[0] + bias, point1[1]
            for j in new_list:
                line_segments.at[j ,'lon1'] = point1[0]
                new_lg.at[line[0],'lon1'] = point1[0]
    return new_lg, line_segments


def consecutive_lines_connecting(entry):

    new_lg = pd.DataFrame(columns=[
        'lon1', 'lat1', 'lon2', 'lat2', 'distance', 'route', 'segments',
        'classified'
    ])

    # list of all routes in a segment cluster (LTS)
    distinct_routes = entry.route.unique().tolist()

    for i in distinct_routes:
        route = entry.loc[entry['route'] == i]
        route = route.sort_values(by=['tstart1'])
        first_line = route.iloc[0]
        last_line = route.iloc[-1]
        dis = gps.haversine((first_line[0], first_line[1]),
                            (last_line[4], last_line[5]))

        line_ids = route.index.values
        # line_ids = np.array2string(line_ids)
        df = pd.DataFrame(
            {
                'lon1': first_line[0],
                'lat1': first_line[1],
                'lon2': last_line[4],
                'lat2': last_line[5],
                'distance': dis,
                'route': i,
                'segments': [line_ids],
                'classified': first_line['classified']
            },
            index=[0])
        new_lg = new_lg.append(df)

        return new_lg


def represent_line(new_lg):

    lon1_list = new_lg['lon1'].values.tolist()
    lon2_list = new_lg['lon2'].values.tolist()
    lon1_list.extend(lon2_list)
    x_list = np.array(lon1_list).reshape(-1, 1)
    lat1_list = new_lg['lat1'].values.tolist()
    lat2_list = new_lg['lat2'].values.tolist()
    lat1_list.extend(lat2_list)
    y_list = np.array(lat1_list).reshape(-1, 1)

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x_list, y_list)  # perform linear regression
    Y_pred = linear_regressor.predict(x_list)  # make predictions

    a = linear_regressor.coef_[0][0]
    b = -1
    c = linear_regressor.intercept_[0]

    return a, b, c


def line_projection(a, b, c, new_lg):
    lon_l = []
    lat_l = []
    projection_lg = pd.DataFrame(
        columns=['lon1', 'lat1', 'lon2', 'lat2', 'route', 'segments'])
    for i in new_lg.iterrows():
        p1_lon = i[1][0]
        p1_lat = i[1][1]
        x1 = (b * b * p1_lon - a * b * p1_lat - a * c) / (a * a + b * b)
        y1 = (a * a * p1_lat - a * b * p1_lon - b * c) / (a * a + b * b)
        p2_lon = i[1][2]
        p2_lat = i[1][3]
        x2 = (b * b * p2_lon - a * b * p2_lat - a * c) / (a * a + b * b)
        y2 = (a * a * p2_lat - a * b * p2_lon - b * c) / (a * a + b * b)
        save_x1 = x1
        save_y1 = y1
        if x1 > x2:
            x1 = x2
            y1 = y2
            x2 = save_x1
            y2 = save_y1
        df = pd.DataFrame(
            {
                'lon1': x1,
                'lat1': y1,
                'lon2': x2,
                'lat2': y2,
                'route': i[1][5],
                'segments': [i[1][6]]
            },
            index=[0])
        lon_l.append(x1)
        lat_l.append(y1)
        lon_l.append(x2)
        lat_l.append(y2)
        projection_lg = projection_lg.append(df)

    x_lon = np.asarray(lon_l).reshape(-1, 1)
    y_lat = np.asarray(lat_l).reshape(-1, 1)

    projection_points = np.concatenate((x_lon, y_lat), axis=1).tolist()

    projection_points.sort()
    projection_lg = projection_lg.sort_values(by=['lon1'])

    return projection_lg, projection_points


def regression_line(a, b, c, x):
    y = c + a * x
    return y


def projection_points(projection_points_list, a, b, c):

    # die projektionspunkte
    final_projection_points = []
    # list mit min, max lon wert eines Punktes (speziell cluster mit centroid)
    test_list = []
    p_point = None
    merged = False
    first = True
    last_point = None
    for point in projection_points_list:
        x = round(point[0], 6)
        if p_point is not None:
            distance = gps.haversine(point, p_point)
            if distance >= 50:
                if merged == False:
                    # auch punkte ohne in einem cluster , bekommen min und max lon der vollständigkeits halber
                    final_projection_points.append(p_point)
                    test_list += 2 * [p_point[0]]
                if merged:
                    test_list.append(p_point_max)
                    first = True
                last_point = point
                merged = False
            else:
                lon = [p_point[0], point[0]]
                lat = [p_point[1], point[1]]
                point = middle_p = list(gps.centroid(lon, lat))
                p_point_max = lon[1]
                if first:
                    test_list.append(lon[0])
                    first = False
                if merged:
                    del final_projection_points[-1]
                final_projection_points.append(middle_p)
                merged = True
        p_point = point
    # um es abzuschließen
    if merged:
        test_list.append(p_point_max)
    else:
        test_list += 2 * [last_point[0]]
        final_projection_points.append(last_point)

    # debug
    if len(final_projection_points) == 1:
        results = []
        new_list = []
        for x in test_list:
            y = regression_line(a, b, c, x)
            results.append([x, y])
            new_list += 2 * [x]
        final_projection_points = results
        test_list = new_list

    return final_projection_points, test_list


def form_sts(final_projection_points, test_list):
    test_lg = pd.DataFrame(columns=[
        'lon1', 'lat1', 'lon2', 'lat2', 'min_x1', 'max_x1', 'min_x2', 'max_x2'
    ])

    lon1 = None
    counter = 0
    l_id = 0
    for i in final_projection_points:
        lon2 = i[0]
        lat2 = i[1]
        if (lon1 is not None):
            df = pd.DataFrame(
                {
                    'lon1': lon1,
                    'lat1': lat1,
                    'lon2': lon2,
                    'lat2': lat2,
                    'min_x1': test_list[counter],
                    'max_x1': test_list[counter + 1],
                    'min_x2': test_list[counter + 2],
                    'max_x2': test_list[counter + 3]
                },
                index=[l_id])
            test_lg = test_lg.append(df)
            lon1 = None
            counter += 2
            l_id += 1
        lon1 = lon2
        lat1 = lat2

    # representiv line segments
    return test_lg


def test(projection_lg, test_lg, new_lg):
    id_list = []
    p_test_lg = pd.DataFrame(columns=[
        'lon1', 'lat1', 'lon2', 'lat2', 'min_x1', 'max_x1', 'min_x2', 'max_x2'
    ])
    loop_break = False
    for entry in projection_lg.iterrows():
        e_lon1 = round(entry[1][0], 6)
        e_lat1 = round(entry[1][1], 6)
        e_lon2 = round(entry[1][2], 6)
        e_lat2 = round(entry[1][3], 6)
        # linenvergleich beginnt
        for line in test_lg.iterrows():
            # jede line in test_lg eigenschaften
            min_x1 = round(line[1][4], 6)
            max_x1 = round(line[1][5], 6)
            min_x2 = round(line[1][6], 6)
            max_x2 = round(line[1][7], 6)
            if e_lon1 >= min_x1 and e_lon1 < max_x2 and e_lon2 <= max_x2:
                id_list.append([line[0]])
                break
            elif p_test_lg.shape[0]:
                for test in p_test_lg.iterrows():
                    if e_lon1 >= round(
                            test[1]['min_x1'], 6
                    ) and e_lon1 < max_x2 and e_lon2 <= max_x2 and e_lon1 < test[
                            1]['max_x2']:
                        id_list.append([test[0], line[0]])
                        loop_break = True
                        break
                if loop_break:
                    loop_break = False
                    break
            p_test_lg = p_test_lg.append(line[1])
        p_test_lg = p_test_lg.iloc[0:0]

    if len(id_list) != new_lg.shape[0]:
        id_list.append([0])
        logger.debug(projection_lg)
        logger.debug(test_lg)
        logger.debug(id_list)
        logger.debug(new_lg)

    new_lg['sub_segment'] = id_list
    return new_lg


def update(new_lg, line_segments):
    # update the classified id with the new sub segments
    for lg in new_lg.iterrows():
        segments = lg[1]['segments']
        sub_segment = lg[1]['sub_segment']
        cluster = lg[1]['classified']
        new_cluster = cluster * FILLER
        for ls in list(segments):
            cluster_seg = []
            i = sub_segment[0]
            while i <= sub_segment[-1]:
                new_id = new_cluster + i
                cluster_seg.append(new_id)
                i += 1

            line_segments.at[ls, 'classified'] = str(cluster_seg)


def write_to_csv(line_segments):
    """
    Write line segments to a new file.

    The new file will contain line segments (LTS) with an additional route id and segment id.

    Parameters
    ----------
    line_segments : DataFrame
        DataFrame with all line segments (LTS).
        
    """

    line_segments.to_csv('test_seg.csv', header=True, sep=';',
                         mode='w')  # header = 'False', index = 'True')

    # logging
    logger.info("wrote results to 'test_seg.csv'")


def write_representative_trajectory(test_lg, seg_id):
    seg_id = int(seg_id)
    test_lg.index = range(seg_id * FILLER, seg_id * FILLER + len(test_lg))
    test_lg.to_csv('representative_trajectories.csv',
                   header=False,
                   sep=';',
                   mode='a')  # header = 'False', index = 'True')


if __name__ == "__main__":
    main()
