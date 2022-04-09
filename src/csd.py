"""csd.py is responsible for common segment discovery.

Based on the previous steps, lines will first grouped together as
line groups (LG) and then a representative line will be calculated
and each line group will be divided into 1 or more different parts.
These parts are common segments and describe as a result of csd.py,
the routes.

"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
    """Starting point to generate STS from LTS.

    A STS is a Common Segment Temporal Sequence, where R is described by
    common segments additionally to LTS.

    Parameters
    ----------
    final_lts : DataFrame
        DataFrame containing all LTS.
    main_logger : logging.Logger
        Main Logger instance for logging purposes.

    """
    # logging
    global logger
    logger = main_logger

    # numpy settings
    np.seterr(invalid='ignore', divide='ignore')

    # logging
    logger.info("common segment discovery started...")
    start_time = current_milli_time()

    # import LTS
    lts_lines = final_lts[[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route'
    ]]

    print(lts_lines.to_latex())

    # assign each L an id
    lts_lines = lts_lines.reset_index(drop=True)

    # garbage collector
    gc.collect()

    # initialize a rtree as an index structure
    rtree = r_tree(lts_lines)

    # find all clusters
    line_groups = group_lines(lts_lines, rtree)

    # generate STS for each LTS group
    calculate_segments(line_groups, lts_lines)

    # logging statistics
    time = (current_milli_time() - start_time) / 1000
    logger.info('''common segment discovery was completed:
 expenditure of time in s: {t}
 total line segments processed: {ls}
 total common segment clusters: {c} 
 total line segments that were marked as noise: {n}'''.format(
        t=time,
        ls=len(lts_lines.index),
        c=len(line_groups),
        n=(lts_lines.classified == "0").sum()))

    # write STS results to new csv
    write_to_csv(lts_lines)


def r_tree(lines):
    """Generate a R-Tree as a index structure for spatial searching.

    A rectangle bounding box is created for each LTS line and its id
    is stored in a rtree.

    Parameters
    ----------
    lines : DataFrame
        DataFrame with all liness (LTS).

    Returns
    -------
    idx : rtree.Index
        R-Tree as a a index structure.

    """
    # logging
    logger.info("rtree index build started...")
    start_time = current_milli_time()

    idx = index.Index()

    for lts in lines.iterrows():
        ls_id = lts[0]
        lon1, lat1 = lts[1]['lon1'], lts[1]['lat1']
        lon2, lat2 = lts[1]['lon2'], lts[1]['lat2']
        min_x = min(lon1, lon2)
        min_y = min(lat1, lat2)
        max_x = max(lon1, lon2)
        max_y = max(lat1, lat2)
        idx.insert(ls_id, (min_x, min_y, max_x, max_y))

    # logging
    time = (current_milli_time() - start_time) / 1000
    logger.info("rtree index was built in {} s".format(time))

    return idx


def group_lines(lines, rtree):
    """Cluster all lines which belong together in a LG.
    
    Step one of TRACLUS line segment clustering algorithm.
    Three different distance functions are used to determine whether a line
    is a neighbor of another line.
    A LG, line group is a cluster of LTS which belong together,
    LG={L, L1,. . . ,Ln}.

    Parameters
    ----------
    lines : DataFrame
        DataFrame with all lines (LTS).
    rtree : rtree.Index
        R-Tree as a a index structure.

    Returns
    -------
    line_groups : list of DataFrame
        List of DataFrame for each line group which contains all related lines.

    Notes
    -----
    The following classified attributes a line (LTS) may have:
        -1  := unclassified
        0   := noise
        >0  := line group id

    References
    ----------
    [3] Lee, Jae-Gil & Han, Jiawei & Whang, Kyu-Young. (2007). Trajectory Clustering:
        A Partition-and-Group Framework. Proceedings of the ACM SIGMOD International
        Conference on Management of Data.

    """
    # logging
    past_time = current_milli_time()
    logger.info("TRACLUS: line segment clustering started...")
    counter = 0

    # Set clusterId to be 0; /* an initial id */
    group_id = 1
    lines['classified'] = -1
    neighbors = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])
    queue = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])
    line_groups = []

    for lts in lines.iterrows():

        # logging
        if counter != 0 and counter == 1 or counter == 5 or counter == 20 or counter == 100 or counter % 500 == 0:
            time = (current_milli_time() - past_time) / 1000
            logger.info(
                "current line segments processed: {c} \n processed last 500 lines in {t} s"
                .format(c=counter, t=time))
            past_time = current_milli_time()

        lts = (lts[0], lines.loc[[lts[0]]].iloc[0])
        is_classified = lts[1]['classified']
        if is_classified == -1:
            # Compute Nε(L)
            neighbors = neighbors.append(lts[1])
            neighbors = neighbors.append(neighborhood(lines, lts, False,
                                                      rtree))
            neighbor_count = neighbors.shape[0]

            # debugging
            logger.debug("neighbors: {n} / {c}".format(n=neighbor_count,
                                                       c=counter))

            if neighbor_count >= MIN_LNS:
                # Assign clusterId to ∀X ∈ Nε(L)
                neighbors['classified'] = group_id
                # line_segments.set_value(entry[0], 'classified' , cluster_id)
                lines.at[lts[0], 'classified'] = group_id
                # Insert Nε(L) − {L} into the queue Q
                queue = queue.append(neighbors.tail(neighbor_count - 1))
                # Step 2
                neighbors = neighbors.append(
                    expand_cluster(lines, queue, group_id, rtree))
                # logger.debug(neighbors)
                for neighbor_index in neighbors.index.values.tolist():
                    lines.set_value(neighbor_index, 'classified', group_id)
                neighbors.drop_duplicates(keep='first', inplace=True)
                line_groups.append(neighbors)
                # Increase clusterId by 1; a new id
                group_id += 1
            else:
                # Mark L as noise
                lines.at[lts[0], 'classified'] = 0

        neighbors = neighbors.iloc[0:0]
        queue = queue.iloc[0:0]

        # logging
        counter += 1

    return line_groups


def neighborhood(lines, line, extended, rtree):
    """Compute the neighborhood (Nε(L)) of a line.

    The neighborhood (Nε(L)) is defined by Nε(Li)={Lj ∈D|dist(Li,Lj)≤ε}.
    In the first step, all possible neighbors are retrieved, where the rtree
    bounding boxes of original line are intersecting with other lines.
    In the second step, the distance between the possible neighbors and the
    original line is calculated to determine weather a line is a real neighbor
    and therefor can be put in the same line group.

    Parameters
    ----------
    lines : DataFrame
        DataFrame with all lines (LTS).
    line : tuple of (int, pandas.Series)
        Tuple with int as line index id and pandas.Series with line data.
    extended : bool
        True if it is an initial neighborhood search.
        False if it is an expanded neighborhood search.
    rtree : rtree.Index
        R-Tree as a an index structure.

    Returns
    -------
    neighbors : DataFrame
        DataFrame with all neighbor lines.

    """
    neighbors = pd.DataFrame(columns=[
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
    n_candidates = lines.iloc[n_candidates_ids, :]
    logger.debug("n_candidates: {}".format(len(n_candidates)))
    for entry in n_candidates.iterrows():
        is_classified = entry[1]['classified']

        if (not extended and is_classified <= 0) or extended:
            if not line[1].equals(entry[1]):
                line_longer, line_shorter = csd_utils.longer_and_shorter_lines(
                    line[1], entry[1])
                line_i = {
                    'si': [line_longer[0], line_longer[1]],
                    'ei': [line_longer[4], line_longer[5]]
                }
                line_j = {
                    'sj': [line_shorter[0], line_shorter[1]],
                    'ej': [line_shorter[4], line_shorter[5]]
                }
                if csd_utils.distance_functions(line_i, line_j) < EPSILON:
                    neighbors = neighbors.append(entry[1])

    return neighbors


def expand_cluster(lines, queue, group_id, rtree):
    """Find and expand neighborhood for each line in queue.
    
    Step two of TRACLUS line segment clustering algorithm: compute a
    density-connected set.

    Parameters
    ----------
    lines : DataFrame
        DataFrame with all lines (LTS).
    queue : DataFrame
        DataFrame which contains the neighbors of a line.
    group_id : int
        Integer which represents the ID of the line group.
    rtree : rtree.Index
        R-Tree as a a index structure.

    Returns
    -------
    line_group : DataFrame
        DataFrame which contains all new lines of a line group.

    """
    line_group = pd.DataFrame(columns=[
        'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
        'distance', 'bearing', 'route', 'classified'
    ])
    logger.debug("segment_id: {id} / queue_size: {s}".format(id=group_id,
                                                             s=queue.shape[0]))
    i = 0
    limit = queue.shape[0] * 20
    while queue.shape[0] != 0 and i <= limit:

        i += 1
        logger.debug("Total Queue Size: {}".format(i))
        if i == limit - 1:
            logger.info("expand_cluster limit reached")

        neighbors = pd.DataFrame(columns=[
            'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2',
            'tend2', 'distance', 'bearing', 'route', 'classified'
        ])
        for entry in queue.iterrows():
            # Let M be the first line segment in Q
            m = entry
            break
        # Compute Nε(M);
        neighbors = neighbors.append(neighborhood(lines, m, True, rtree))
        neighbor_count = neighbors.shape[0] + 1
        # if (|Nε(M)| ≥ MinLns) then
        if neighbor_count >= MIN_LNS:
            for xn in neighbors.iterrows():
                # if (X is unclassified or noise) then
                if xn[1]['classified'] == -1 or xn[1]['classified'] == 0:
                    # line_segments.set_value(xn[0], 'classified', cluster_id)
                    # Assign clusterId to X
                    lines.at[xn[0], 'classified'] = group_id
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
                        'classified': group_id
                    })
                    series.name = xn[0]
                    line_group = line_group.append(series)
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
                        'classified': group_id
                    })
                    series.name = xn[0]
                    queue = queue.append(series)
        # Remove M from the queue Q
        queue = queue.iloc[1:]

    return line_group


def calculate_segments(line_groups, lines):
    """Generate common segments from grouped lines.

    Parameters
    ----------
    line_groups : List of DataFrame
        List which contains all line groups as DataFrame.
    lines : DataFrame
        DataFrame with all lines (LTS).

    References
    ----------
    [1] Fu, Z., Tian, Z., Xu, Y. and Zhou, K. (2017). Mining Frequent Route
        Patterns Based on Personal Trajectory Abstraction. IEEE Access, 5,
        pp.11352-11363.

    """
    for line in line_groups:
        # connect consecutive Lines in LG
        connected_line = connect_consecutive_lines(line)
        # Lines with a lengh of 0 will be ignored
        if connected_line.iloc[0]['distance'] == 0:
            logger.info("distance == 0")
        else:
            # exceptional case for representative line generation
            connected_line, lines = check_for_90(connected_line, lines)
            # representative line generation
            a, b, c = calculate_representative_line(connected_line)
            # calculate line projection points
            projection_lg, projection_points_list = calculate_line_projection(
                a, b, c, connected_line)
            # merge line projection points if a distance criteria is meet
            final_projection_points, pp_boundaries = calculate_projection_points(
                projection_points_list, a, b, c)
            # get final representative lines / line segments
            seg_lines = form_representative_seg(final_projection_points,
                                                pp_boundaries)
            # assign final representative lines / line segments to lines
            connected_line = assign_seg_to_line(projection_lg, seg_lines,
                                                connected_line)
            # update lines DataFrame with segment id
            lines['classified'] = lines['classified'].apply(lambda x: str(x))
            update(connected_line, lines)
            seg_id = connected_line.iloc[0]['classified']
            # write the result to a csv file
            write_representative_trajectory(seg_lines, seg_id)


def check_for_90(connected_line, lines):
    """Check if lines of a line group has a bearing of 90 or 270 degree.

    If line has a an angle between L and the horizontal axis (bearing) of 90 degree or 270
    degree, a insignificant deviation is added to longitude value of the start point.
    This is necessary to calculate a representative line as f(x) function in a later step.

    Parameters
    ----------
    connected_line :  DataFrame
        New DataFrame of a line group.
    lines : DataFrame
        DataFrame with all lines (LTS).

    Returns
    -------
    connected_line : DataFrame
        DataFrame which contains all new lines of a line group.
    lines : DataFrame
        DataFrame which contains all lines of a line group.

    """
    for line in connected_line.iterrows():
        deviation = 0.000001
        point1 = line[1]['lon1'], line[1]['lat1']
        point2 = line[1]['lon2'], line[1]['lat2']

        if round(gps.bearingCalculator(point1, point2)) == 90 or round(
                gps.bearingCalculator(point1, point2)) == 270:

            logger.info("check_for_90(connected_line, lines)")

            new_list = line[1]['segments']
            point1 = point1[0] + deviation, point1[1]

            for j in new_list:
                # update lon1 value
                lines.at[j, 'lon1'] = point1[0]
                connected_line.at[line[0], 'lon1'] = point1[0]

    return connected_line, lines


def connect_consecutive_lines(line_group):
    """Check if a line group contains lines from the same Route and connect them.

    Lines that belong to the same LTS in LG, are merged into one line,
    LG={L, ..., L_m, ..., L_i}.

    Parameters
    ----------
    line_group : DataFrame
        DataFrame which contains lines of one LG.

    Returns
    -------
    connected_line : DataFrame
        DataFrame which contains lines of a line group.

    """
    connected_line = pd.DataFrame(columns=[
        'lon1', 'lat1', 'lon2', 'lat2', 'distance', 'route', 'segments',
        'classified'
    ])

    # list of all routes in a LG
    distinct_routes = line_group.route.unique().tolist()

    for i in distinct_routes:
        # get all L from the same LG
        route = line_group.loc[line_group['route'] == i]
        route = route.sort_values(by=['tstart1'])
        first_line = route.iloc[0]
        last_line = route.iloc[-1]
        # calculate the Haversine distance
        dis = gps.haversine((first_line[0], first_line[1]),
                            (last_line[4], last_line[5]))
        # get all L ids
        line_ids = route.index.values
        # create a new combined L
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
        connected_line = connected_line.append(df)

    return connected_line


def calculate_representative_line(line_group):
    """Calculate representative line of a line group

    Ordinary least squares linear regression is used to calculate
    the representative line in the form of function coefficients:
    Ax + By + C = 0

    Parameters
    ----------
    line_group : DataFrame
        DataFrame which contains LTS of one LG.

    Returns
    -------
    a : float
        slope
    b : float
        y
    c : float
        y-axis intercept

    """
    lon1_list = line_group['lon1'].values.tolist()
    lon2_list = line_group['lon2'].values.tolist()
    lon1_list.extend(lon2_list)
    x_list = np.array(lon1_list).reshape(-1, 1)
    lat1_list = line_group['lat1'].values.tolist()
    lat2_list = line_group['lat2'].values.tolist()
    lat1_list.extend(lat2_list)
    y_list = np.array(lat1_list).reshape(-1, 1)

    # create object for the class
    linear_regressor = LinearRegression()
    # perform linear regression
    linear_regressor.fit(x_list, y_list)

    a = linear_regressor.coef_[0][0]
    b = -1
    c = linear_regressor.intercept_[0]

    return a, b, c


def calculate_line_projection(a, b, c, line_group):
    """Calculate the projection points on the representative line.

    Parameters
    ----------
    a : float
        slope
    b : float
        y
    c : float
        y-axis intercept
    line_group : DataFrame
        DataFrame which contains LTS of one LG.

    Returns
    -------
    projection_df, projection_points : tuple of (DataFrame, list)
        DataFrame and list contain all ordered projection points.

    """
    lon_l = []
    lat_l = []
    projection_df = pd.DataFrame(
        columns=['lon1', 'lat1', 'lon2', 'lat2', 'route', 'segments'])

    for i in line_group.iterrows():
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
        # order them from left to right
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
        projection_df = projection_df.append(df)

    x_lon = np.asarray(lon_l).reshape(-1, 1)
    y_lat = np.asarray(lat_l).reshape(-1, 1)
    projection_points = np.concatenate((x_lon, y_lat), axis=1).tolist()

    projection_points.sort()
    projection_df = projection_df.sort_values(by=['lon1'])

    return projection_df, projection_points


def get_regression_line(a, b, c, x):
    """Calculate the y value based on function coefficients.

    Parameters
    ----------
    a : float
        slope
    b : float
        y
    c : float
        y-axis intercept
    x : float
        x variable

    Returns
    -------
    y : float
        y value

    """
    y = c + a * x
    return y


def calculate_projection_points(projection_points_list, a, b, c):
    """Calculate the the final projection points.

    Removal of redundant point.
    If he distance between two consecutive points is less than 50m, the two
    points will be removed and replaced by their middle point.

    Parameters
    ----------
    projection_points_list
    a : float
        slope
    b : float
        y
    c : float
        y-axis intercept

    Returns
    -------
    final_projection_points : DataFrame
        DataFrame with the final projection points.
    pp_boundaries : list of float
        list that contains the outer points of each projection point (only long value).

    """
    final_projection_points = []
    # list with min, max lon values
    pp_boundaries = []
    p_point = None
    merged = False
    first = True
    last_point = None

    for point in projection_points_list:
        if p_point is not None:
            distance = gps.haversine(point, p_point)
            if distance >= 50:
                if not merged:
                    # points which will not be merged get min and max lon for the sake of completeness
                    final_projection_points.append(p_point)
                    pp_boundaries += 2 * [p_point[0]]
                if merged:
                    # set max lon of last merged point
                    pp_boundaries.append(p_point_max)
                    first = True
                last_point = point
                merged = False
            else:
                lon = [p_point[0], point[0]]
                lat = [p_point[1], point[1]]
                point = middle_p = list(gps.centroid(lon, lat))
                p_point_max = lon[1]
                # set first point of a merged cluster
                if first:
                    pp_boundaries.append(lon[0])
                    first = False
                if merged:
                    # delete the last projection point
                    del final_projection_points[-1]
                final_projection_points.append(middle_p)
                merged = True
        # set previous point
        p_point = point
    # to consider the last point
    if merged:
        pp_boundaries.append(p_point_max)
    else:
        pp_boundaries += 2 * [last_point[0]]
        final_projection_points.append(last_point)

    # exception handling
    if len(final_projection_points) == 1:
        results = []
        new_list = []
        for x in pp_boundaries:
            y = get_regression_line(a, b, c, x)
            results.append([x, y])
            new_list += 2 * [x]
        final_projection_points = results
        pp_boundaries = new_list

    return final_projection_points, pp_boundaries


def form_representative_seg(final_projection_points, pp_boundaries):
    """Form representative segment lines.

    Parameters
    ----------
    final_projection_points : DataFrame
        DataFrame which contains the final projection points.
    pp_boundaries : list of float
        list which contains the boundaries of each projection point.

    Returns
    -------
    seg_lines : DataFrame
        DataFrame which represents representative line segments of a LG.

    """
    seg_lines = pd.DataFrame(columns=[
        'lon1', 'lat1', 'lon2', 'lat2', 'min_x1', 'max_x1', 'min_x2', 'max_x2'
    ])

    lon1 = None
    counter = 0
    l_id = 0
    for i in final_projection_points:
        lon2 = i[0]
        lat2 = i[1]
        if lon1 is not None:
            df = pd.DataFrame(
                {
                    'lon1': lon1,
                    'lat1': lat1,
                    'lon2': lon2,
                    'lat2': lat2,
                    'min_x1': pp_boundaries[counter],
                    'max_x1': pp_boundaries[counter + 1],
                    'min_x2': pp_boundaries[counter + 2],
                    'max_x2': pp_boundaries[counter + 3]
                },
                index=[l_id])
            seg_lines = seg_lines.append(df)
            lon1 = None
            counter += 2
            l_id += 1
        lon1 = lon2
        lat1 = lat2

    return seg_lines


def assign_seg_to_line(projection_lg, seg_lines, connected_line):
    """Assign each L of in a LG to a one or more representative line segments.

    Parameters
    ----------
    projection_lg : DataFrame
        DataFrame which contains a LG, described with projection points.
    seg_lines : DataFrame
        DataFrame which represents representative line segments of a LG.
    connected_line : DataFrame
        DataFrame which contains the original lines of a line group.

    Returns
    -------
    connected_line : DataFrame
        DataFrame which contains the original lines of a line group.

    """
    id_list = []
    p_seg_lines = pd.DataFrame(columns=[
        'lon1', 'lat1', 'lon2', 'lat2', 'min_x1', 'max_x1', 'min_x2', 'max_x2'
    ])
    loop_break = False
    for entry in projection_lg.iterrows():
        # lines in LG
        e_lon1 = round(entry[1][0], 6)
        e_lon2 = round(entry[1][2], 6)
        # line comparison
        for line in seg_lines.iterrows():
            # lines in representative line segments
            min_x1 = round(line[1][4], 6)
            max_x2 = round(line[1][7], 6)
            if min_x1 <= e_lon1 < max_x2 and e_lon2 <= max_x2:
                id_list.append([line[0]])
                break
            elif p_seg_lines.shape[0]:
                for previous in p_seg_lines.iterrows():
                    if round(
                            previous[1]['min_x1'], 6
                    ) <= e_lon1 < max_x2 and e_lon2 <= max_x2 and e_lon1 < previous[
                            1]['max_x2']:
                        id_list.append([previous[0], line[0]])
                        # if all segments are found, search can be interrupted
                        loop_break = True
                        break
                if loop_break:
                    loop_break = False
                    break
            p_seg_lines = p_seg_lines.append(line[1])
        p_seg_lines = p_seg_lines.iloc[0:0]

    # update
    connected_line['sub_segment'] = id_list

    return connected_line


def update(connected_line, lines):
    """Update the classified id with the new segment id.

    The classified id of each L is updated with their final
    segment ids.

    Parameters
    ----------
    connected_line : DataFrame
        DataFrame which contains the original lines of a line group.
    lines : DataFrame
        DataFrame with all lines (LTS).

    """
    # update the classified id with the new sub segments
    for lg in connected_line.iterrows():
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

            lines.at[ls, 'classified'] = str(cluster_seg)


def write_to_csv(lines):
    """Write lines to a new file.
    
    The new file is called 'test_seg.csv' and  will contain lines (LTS) with an additional
    route id and segment id. So each route is described as a combination of lines (LTS)
    which are also assigned to segments. Based on that information a sequential pattern
    mining can be applied.

    Parameters
    ----------
    lines : DataFrame
        DataFrame with all lines (LTS).

    """

    lines.to_csv('test_seg.csv', header=True, sep=';',
                 mode='w')  # header = 'False', index = 'True')

    # logging
    logger.info("wrote results to 'test_seg.csv'")


def write_representative_trajectory(seg_lines, seg_id):
    """Write representative line segments to a new file.

    The file is called 'representative_trajectories.csv' which is later used to visualize
    the most frequent routes.
    Every frequent route is described as a sequence of representative line segment ids and
    the coordinates of each representative line segment is stored in
    'representative_trajectories.csv' for lookup.

    Parameters
    ----------
    seg_lines : DataFrame
        DataFrame which represents representative line segments of a LG.
    seg_id : int
        int that is the id of a line segment.

    """
    seg_id = int(seg_id)
    seg_lines.index = range(seg_id * FILLER, seg_id * FILLER + len(seg_lines))
    seg_lines.to_csv('representative_trajectories.csv',
                     header=False,
                     sep=';',
                     mode='a')  # header = 'False', index = 'True')
