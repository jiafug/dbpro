import numpy as np
import math


def longer_and_shorter_lines(line_a, line_b):
    """
    Compares which line segment has the greater distance.

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
        return (line_b, line_a)
    else:
        return (line_a, line_b)


def projection_points(line_a, line_b):
    """
    Calulate the projection points from line_b to line_a.

    Parameters
    ----------
    line_a : Series
        Line segment (LTS) as a series.

    line_b : Series
        Line segment (LTS) as a series.
        
    Returns
    -------
    (pe, ps, cos_angle) : tuple of (float, float, float)
        Tuple with the projection points ps and pe and the angle theta

    """
    si = line_a['si']
    ei = line_a['ei']
    sj = line_b['sj']
    ej = line_b['ej']

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

    return (pe, ps, cos_angle)


def distance_functions(line_a, line_b):
    """
    Calculate the distance between two line segments (LTS).

    Perpendicular distance (d⊥), the parallel distance (d∥), and the angle distance (dθ)
    is used to determine the distance between two line segments (LTS).

    Parameters
    ----------
    line_a : Series
        Line segment (LTS) as a series.

    line_b : Series
        Line segment (LTS) as a series.
        
    Returns
    -------
    (perpendicular_distance + parallel_distance + angle_distance) : int
        The Distance is the sum of d⊥, d∥ and dθ.

    """
    si = line_a['si']
    ei = line_a['ei']
    sj = line_b['sj']
    ej = line_b['ej']

    # calculate the projection points first
    pe, ps, cos_angle = projection_points(line_a, line_b)

    def perpendicular_distance(line_a, line_b):
        """
        Calculate the perpendicular distance (d⊥).

        Parameters
        ----------
        line_a : Series
            Line segment (LTS) as a series.

        line_b : Series
            Line segment (LTS) as a series.
            
        Returns
        -------
        dist : float
            Perpendicular distance (d⊥).

        """
        l1 = np.linalg.norm(ps - sj)
        l2 = np.linalg.norm(pe - ej)
        dist = (l1 * l1 + l2 * l2) / l1 + l2
        return dist

    def parallel_distance(line_a, line_b):
        """
        Calculate the parallel distance d∥.

        Parameters
        ----------
        line_a : Series
            Line segment (LTS) as a series.

        line_b : Series
            Line segment (LTS) as a series.
            
        Returns
        -------
        dist : float
            Parallel distance d∥.

        """
        l1 = min(np.linalg.norm(si - ps), np.linalg.norm(ei - ps))
        l2 = min(np.linalg.norm(si - pe), np.linalg.norm(ei - pe))
        dist = min(l1, l2)
        return dist

    def angle_distance(line_a, line_b):
        """
        Calculate the angle distance (dθ).

        Parameters
        ----------
        line_a : Series
            Line segment (LTS) as a series.

        line_b : Series
            Line segment (LTS) as a series.

        Returns
        -------
        dist : float 
            Angle distance (dθ).

        """
        nonlocal cos_angle
        if (cos_angle) > 1:
            cos_angle = 1
        elif (cos_angle) < -1:
            cos_angle = -1
        angle = math.acos(cos_angle)
        sjej = np.array([ej[0] - sj[0], ej[1] - sj[1]])
        dist = 0
        if np.degrees(angle) >= 0 and np.degrees(angle) < 90:
            dist = np.linalg.norm(sjej) * math.sin(angle)
        elif np.degrees(angle) >= 90 and np.degrees(angle) <= 180:
            dist = np.linalg.norm(sjej)
        return dist

    perpendicular_distance = perpendicular_distance(line_a, line_b)
    parallel_distance = parallel_distance(line_a, line_b)
    angle_distance = angle_distance(line_a, line_b)

    return (perpendicular_distance + parallel_distance + angle_distance)
