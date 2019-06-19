import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import csv
import math
import mplleaflet
import matplotlib.pyplot as plt
import pandas as pd
import gps_utils as gps
import time as ttt

import csd_utils

import numpy as np
global line_segments
current_milli_time = lambda: int(round(ttt.time() * 1000))
line_segments = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route'])


def csd_import():
    global line_segments
    # line_segments = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route'])
    with open('test_lts.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        route_id = 0 
        for row in csv_reader:
            if row[0] == '':
                pass
            else:
                if float(row[0]) == 0:
                    route_id += 1
                # a short line segment might induce over-clustering, thus a the partitioning criteria is added
                if float(row[9]) >= 25:
                    df = {'lon1': float(row[1]), 'lat1': float(row[2]), 'tstart1': float(row[3]), 'tend1': row[4],
                        'lon2': float(row[5]), 'lat2': float(row[6]), 'tstart2': float(row[7]), 'tend2': row[8],
                        'distance': row[9], 'bearing': row[10], 'route': route_id}
                    line_segments = line_segments.append(df, ignore_index=True)

'''
Berechnet die Nachbarschaft indem die Distanz der Input Line mit allen anderen Linien verglichen wird:
The ε-neighborhood Nε(Li) of a line segment Li ∈DisdefinedbyNε(Li)={Lj ∈D|dist(Li,Lj)≤ε
'''
# Compute Nε(L);
def neighborhood(line, cluster_id, extended): 
    global line_segments
    neighbors = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route', 'classified'])
    for entry in line_segments.iterrows():
        is_classified = entry[1]['classified']
        
        if (not extended and is_classified < 0) or extended:    
            if not line[1].equals(entry[1]):
                line_longer, line_shorter = csd_utils.longer_and_shorter_lines(line[1], entry[1])
                line_a = {'si': [line_longer[0], line_longer[1]], 'ei': [line_longer[4], line_longer[5]]}
                line_b = {'sj': [line_shorter[0], line_shorter[1]], 'ej': [line_shorter[4], line_shorter[5]]}
                '''
                Parameter ε ist aktuell 0.0005
                '''
                if csd_utils.distance_functions(line_a, line_b) < 0.0005:
                    # line_segments.set_value(entry[0],'classified',cluster_id)
                    neighbors = neighbors.append(entry[1])
    return neighbors

# /* Step 2: compute a density-connected set */
def expand_cluster(queue, cluster_id):
    global line_segments
    # print("----------------input queue start----------------")
    # print(queue)
    # print("----------------input queue ended----------------")
    cluster = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route', 'classified'])
    while queue.shape[0] != 0:
        neighbors = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route', 'classified'])
        for entry in queue.iterrows():
            # Let M be the first line segment in Q;
            m = entry
            break
        # Compute Nε(M);
        neighbors = neighbors.append(neighborhood(m, cluster_id, True))
        neighbor_count = neighbors.shape[0] + 1
        # if (|Nε(M)| ≥ MinLns) then
        if (neighbor_count >= 3):
            for xn in neighbors.iterrows():
                #if (X is unclassified or noise) then
                if xn[1]['classified'] == -1 or xn[1]['classified'] == -2:
                    # line_segments.set_value(xn[0], 'classified', cluster_id)
                    # Assign clusterId to X;
                    line_segments.at[xn[0],'classified'] = cluster_id
                    series = pd.Series({'lon1': float(xn[1][0]), 'lat1': float(xn[1][1]), 'tstart1': float(xn[1][2]), 'tend1': xn[1][3],
                          'lon2': float(xn[1][4]), 'lat2': float(xn[1][5]), 'tstart2': float(xn[1][6]), 'tend2': xn[1][7],
                          'distance': xn[1][8], 'bearing': xn[1][9], 'route': xn[1][10], 'classified': cluster_id})
                    series.name = xn[0]
                    cluster = cluster.append(series)
                # if (X is unclassified) then
                if xn[1]['classified'] == -1:
                    # Insert X into the queue Q;
                    series = pd.Series({'lon1': float(xn[1][0]), 'lat1': float(xn[1][1]), 'tstart1': float(xn[1][2]), 'tend1': xn[1][3],
                          'lon2': float(xn[1][4]), 'lat2': float(xn[1][5]), 'tstart2': float(xn[1][6]), 'tend2': xn[1][7],
                          'distance': xn[1][8], 'bearing': xn[1][9], 'route': xn[1][10], 'classified': cluster_id})
                    series.name = xn[0]
                    queue = queue.append(series)
        # Remove M from the queue Q;
        queue = queue.iloc[1:]
        
        # print("----------------output queue start----------------")
        # print(queue)
        # print("----------------output queue ended----------------")
        
    # print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
    # print(cluster)
    # print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
    return cluster

def main():
    global line_segments
    csd_import()

    '''
    benchmark beginns
    '''
    start_time = current_milli_time()

    np.seterr(invalid='ignore', divide='ignore')
    # /* Step 1 */
    # Set clusterId to be 0; /* an initial id */
    cluster_id = 0
    line_segments['classified'] = -1
    neighbors = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route', 'classified'])
    queue = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route', 'classified'])
    clusters = []

    '''
    die Spalte classified der Dataframes kann folgende Werte besitzen:
    -1 := unclassified
    -2 := noise
    >= 0 := die entsprechende Segmente (Cluster)

    Output dieser Zelle:
    1.) Modifiziertes Dataframe der line_segments (Ausgangsdaten der test_lts.csv) mit der Routennummer und classified Status
    2.) cluster: eine Liste mit Dataframes, die die Segmente (Cluster) darstellen
    '''
    counter = 0

    for entry in line_segments.iterrows():

        counter += 1

        if (counter == 10 or counter == 25 or counter % 50 == 0):
            time = (current_milli_time() - start_time) / 1000
            print("Current Lines processed: {}".format(counter))
            print("Current time running in s: {}".format(time))
        
        entry = (entry[0], line_segments.loc[[entry[0]]].iloc[0])
        is_classified = entry[1]['classified']
        if is_classified == -1:
            # Compute Nε(L);
            neighbors = neighbors.append(entry[1])
            neighbors = neighbors.append(neighborhood(entry, cluster_id, False))
            neighbor_count = neighbors.shape[0]
            '''
            Parameter MinLns ist aktuell 3
            '''
            if neighbor_count >= 3:
                # Assign clusterId to ∀X ∈ Nε(L);
                neighbors['classified'] = cluster_id
                # line_segments.set_value(entry[0], 'classified' , cluster_id)
                line_segments.at[entry[0],'classified'] = cluster_id
                # Insert Nε(L) − {L} into the queue Q;
                queue = queue.append(neighbors.tail(neighbor_count - 1))
                # /* Step 2 */
                neighbors = neighbors.append(expand_cluster(queue, cluster_id))
                # print("????????????????????????????")
                # print(neighbors)
                # print("????????????????????????????")
                for index in neighbors.index.values.tolist():
                    line_segments.set_value(index, 'classified' , cluster_id)
                neighbors.drop_duplicates(keep='first', inplace=True)
                '''
                clusters ist eine Liste mit allen Linien eines gemeinsamen Segments
                '''
                clusters.append(neighbors)
                # Increase clusterId by 1; /* a new id */
                cluster_id += 1
            else:
                # Mark L as noise;
                # line_segments.set_value(entry[0],'classified', -2)
                line_segments.at[entry[0],'classified'] = -2
                if entry[0] == 4:
                    print("What should i do? First")
                    
        neighbors = neighbors.iloc[0:0]
        queue = queue.iloc[0:0]
        
        '''
        Aktuell auf 4 Segmente (Cluster) beschränkt (aus Testzwecken / Performance)
        '''

    '''
    benchmark ends
    '''
    time = (current_milli_time() - start_time) 
    print("----------Line Segment Clustering: BEGIN----------")
    print("time consumption in ms: {}".format(time))
    print("----------Line Segment Clustering: END----------")
    write_to_csv()

def write_to_csv():
    '''
    Schreibt das modifizierte Dataframe der line_segments in eine neue CSV (test_seg.csv)
    '''



    '''
    benchmark beginns
    '''
    start_time = current_milli_time()


    line_segments.to_csv('test_seg.csv', header=True, sep=';', mode='w')  # header = 'False', index = 'True')


    '''
    benchmark ends
    '''
    time = (current_milli_time() - start_time) 
    print("----------write_to_csv(...): BEGIN----------")
    print("time consumption in ms: {}".format(time))
    print("----------write_to_csv(...): END----------")

if __name__ == "__main__":
    main()