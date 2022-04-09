"""csd_utils.py is responsible for the distance calculations of two lines.

Before the distance can be calculated, the longer and shorter line are
determined and assigned to longer line line_i and shorter line line_j.

The distance function used in clustering line segments is composed of
three components:
(i) the perpendicular distance (d⊥),
(ii) the parallel distance (d∥), and
(iii) the angle distance (dθ).

"""
import numpy as np
import math


def longer_and_shorter_lines(line_a, line_b):
    """Compare which line segment has the greater distance.

    Parameters
    ----------
    line_a : Series
        Line segment (LTS) as a series.
    line_b : Series
        Line segment (LTS) as a series.

    Returns
    -------
    (line_a, line_b) : tuple of (Series, Series)
        Return a tuple where the first line segment (LTS) is the longer one.

    """
    if line_a['distance'] > line_b['distance']:
        return line_a, line_b
    else:
        return line_b, line_a


def projection_points(line_i, line_j):
    """Calculate the projection points from line_j to line_i.

    Parameters
    ----------
    line_i : Series
        Line segment (LTS) as a series.
    line_j : Series
        Line segment (LTS) as a series.

    Returns
    -------
    (pe, ps, cos_angle) : tuple of (float, float, float)
        Tuple with the projection points ps and pe and the angle theta.

    """
    si = line_i['si']
    ei = line_i['ei']
    sj = line_j['sj']
    ej = line_j['ej']

    # ps
    sisj = np.array([sj[0] - si[0], sj[1] - si[1]])
    siei = np.array([ei[0] - si[0], ei[1] - si[1]])
    siei_norm = np.linalg.norm(siei)
    u1 = np.dot(sisj, siei) / (siei_norm * siei_norm)
    ps = si + u1 * siei

    # pe
    siej = np.array([ej[0] - si[0], ej[1] - si[1]])
    u2 = np.dot(siej, siei) / (siei_norm * siei_norm)
    pe = si + u2 * siei

    # angle θ
    sjej = np.array([ej[0] - sj[0], ej[1] - sj[1]])
    cos_angle = np.dot(siei, sjej) / np.multiply(np.linalg.norm(siei),
                                                 np.linalg.norm(sjej))

    return pe, ps, cos_angle


def distance_functions(line_i, line_j):
    """Calculate the distance between two line segments (LTS).
    
    Perpendicular distance (d⊥), the parallel distance (d∥), and the angle distance (dθ)
    is used to determine the distance between two line segments (LTS).

    Parameters
    ----------
    line_i : Series
        Line segment (LTS) as a series.
    line_j : Series
        Line segment (LTS) as a series.

    Returns
    -------
    (perpendicular_distance + parallel_distance + angle_distance) : int
        The Distance is the sum of d⊥, d∥ and dθ.

    References
    ----------
    [3] Lee, Jae-Gil & Han, Jiawei & Whang, Kyu-Young. (2007). Trajectory Clustering: A
        Partition-and-Group Framework. Proceedings of the ACM SIGMOD International
        Conference on Management of Data.

    """
    si = line_i['si']
    ei = line_i['ei']
    sj = line_j['sj']
    ej = line_j['ej']

    # calculate the projection points first
    pe, ps, cos_angle = projection_points(line_i, line_j)

    def perpendicular_distance():
        """Calculate the perpendicular distance (d⊥).

        Returns
        -------
        dist : float
            Perpendicular distance (d⊥).

        """
        l1 = np.linalg.norm(ps - sj)
        l2 = np.linalg.norm(pe - ej)
        dist = (l1 * l1 + l2 * l2) / l1 + l2
        return dist

    def parallel_distance():
        """Calculate the parallel distance d∥.

        Returns
        -------
        dist : float
            Parallel distance d∥.

        """
        l1 = min(np.linalg.norm(si - ps), np.linalg.norm(ei - ps))
        l2 = min(np.linalg.norm(si - pe), np.linalg.norm(ei - pe))
        dist = min(l1, l2)
        return dist

    def angle_distance():
        """Calculate the angle distance (dθ).

        Returns
        -------
        dist : float
            Angle distance (dθ).

        """
        nonlocal cos_angle
        if cos_angle > 1:
            cos_angle = 1
        elif cos_angle < -1:
            cos_angle = -1
        angle = math.acos(cos_angle)
        sjej = np.array([ej[0] - sj[0], ej[1] - sj[1]])
        dist = 0
        if 0 <= np.degrees(angle) < 90:
            dist = np.linalg.norm(sjej) * math.sin(angle)
        elif 90 <= np.degrees(angle) <= 180:
            dist = np.linalg.norm(sjej)
        return dist

    perpendicular_distance = perpendicular_distance()
    parallel_distance = parallel_distance()
    angle_distance = angle_distance()

    return perpendicular_distance + parallel_distance + angle_distance
