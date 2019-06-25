import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import csv
import pandas as pd
import time as ttt
import csd_utils
import numpy as np
from rtree import index

# input variables
DATA_PATH = "test_lts.csv"
MIN_LENGTH = 25
EPSILON = 0.0005
MIN_LNS = 3

# logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
current_milli_time = lambda: int(round(ttt.time() * 1000))


def main():
    # numpy settings
    np.seterr(invalid='ignore', divide='ignore')

    # logging
    logger.info("common segment discovery started...")
    start_time = current_milli_time()

    # import line segments
    line_segments = csd_import()

    # initialize a r-tree as a index
    rtree = r_tree(line_segments)

    # find all clusters
    ls_clusters = line_segment_clustering(line_segments, rtree)

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
        n=(line_segments.classified == 0).sum()))

    # write LTS results to new csv
    write_to_csv(line_segments)


def csd_import():
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

    with open(DATA_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        route_id = 0
        for row in csv_reader:
            if row[0] == '':
                pass
            else:
                if float(row[0]) == 0:
                    route_id += 1
                # short line segment filter
                if float(row[9]) >= MIN_LENGTH:
                    df = {
                        'lon1': float(row[1]),
                        'lat1': float(row[2]),
                        'tstart1': float(row[3]),
                        'tend1': row[4],
                        'lon2': float(row[5]),
                        'lat2': float(row[6]),
                        'tstart2': float(row[7]),
                        'tend2': row[8],
                        'distance': row[9],
                        'bearing': row[10],
                        'route': route_id
                    }
                    line_segments = line_segments.append(df, ignore_index=True)
    time = (current_milli_time() - start_time) / 1000
    logger.info("lts.csv was imported in {} s".format(time))

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
        min_x = min(lon1, lon2) - EPSILON
        min_y = min(lat1, lat2) - EPSILON
        max_x = max(lon1, lon2) + EPSILON
        max_y = max(lat1, lat2) + EPSILON
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
        if counter != 0 and counter % 500 == 0:
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
                logger.debug(neighbors)
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
    while queue.shape[0] != 0:
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
                        'route': xn[1][10],
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
                        'route': xn[1][10],
                        'classified': cluster_id
                    })
                    series.name = xn[0]
                    queue = queue.append(series)
        # Remove M from the queue Q
        queue = queue.iloc[1:]

    return cluster


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


if __name__ == "__main__":
    main()
