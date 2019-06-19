import numpy as np
import math

'''
Input: Zwei Linien als Pandas Series
Output: Die längere Line zuerst, dann die kürzere Line als Pandas Series
'''
def longer_and_shorter_lines(line_a, line_b):
    if line_a['distance'] > line_b['distance']:
        return (line_b, line_a)
    else:
        return (line_a, line_b)


def projection_points(line_a, line_b):
    '''
    si = [13.323899, 52.512749]
    ei = [13.332428,  52.513389]
    sj = [13.325390,  52.513223]
    ej = [13.328577,  52.514398]
    '''
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
    cos_angle = np.dot(siei, sjej) / np.multiply(np.linalg.norm(siei), np.linalg.norm(sjej))
        
    return (pe, ps, cos_angle)

def distance_functions(line_a, line_b):
    '''
    si = [13.323899, 52.512749]
    ei = [13.332428,  52.513389]
    sj = [13.325390,  52.513223]
    ej = [13.328577,  52.514398]
    '''
    si = line_a['si']
    ei = line_a['ei']
    sj = line_b['sj']
    ej = line_b['ej']
    
    '''
    if (si == sj and ei == ej) or (si == ei or sj == ej):
        return 0
    '''
    
    pe, ps, cos_angle = projection_points(line_a, line_b)
    
    def perpendicular_distance(line_a, line_b):
        l1 = np.linalg.norm(ps-sj)
        l2 = np.linalg.norm(pe-ej)
        dist = (l1*l1+l2*l2)/l1+l2
        return dist

    def parallel_distance(line_a, line_b):
        l1 = min(np.linalg.norm(si-ps), np.linalg.norm(ei-ps))
        l2 = min(np.linalg.norm(si-pe), np.linalg.norm(ei-pe))
        dist = min(l1, l2)
        return dist

    def angle_distance(line_a, line_b):
        nonlocal cos_angle
        if (cos_angle) > 1:
            cos_angle = 1
        elif (cos_angle) < -1:
            cos_angle = -1
        angle = math.acos(cos_angle)
        sjej = np.array([ej[0] - sj[0], ej[1] - sj[1]])
        dist = 0
        # print("angle: {}".format(np.degrees(angle)))
        if np.degrees(angle) >= 0 and np.degrees(angle) < 90:
            dist = np.linalg.norm(sjej) * math.sin(angle)
        elif np.degrees(angle) >= 90 and np.degrees(angle) <= 180:
            dist = np.linalg.norm(sjej)
        return dist

    perpendicular_distance = perpendicular_distance(line_a, line_b)
    parallel_distance = parallel_distance(line_a, line_b)
    angle_distance = angle_distance(line_a, line_b)
    
    # print("perpendicular_distance: {}".format(perpendicular_distance))
    # print("parallel_distance: {}".format(parallel_distance))
    # print("angle_distance: {}".format(angle_distance))
    
    '''
    Per Diffinition ist die Distanz die Addierte Distanzen der verschiedenen Funktionen,
    eine Gewichting kann noch erfolgen falls notwendig
    '''
    
    return (perpendicular_distance + parallel_distance + angle_distance)

