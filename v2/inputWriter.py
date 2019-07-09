# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:46:26 2019

@author: m.bauknecht
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import csv
import math
import pandas as pd
import time as ttt

current_milli_time = lambda: int(round(ttt.time() * 1000))
start_time = current_milli_time()

# einlesen des csv
line_segments = pd.DataFrame(columns=[
    'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
    'distance', 'bearing', 'route', 'classified'
])
global result
result = ''
with open('test_seg.csv', 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=';')
    route_id = 1
    #limit = 10  for debugging only
    for row in reader:
        # skip header
        if row[1] == 'lon1':
            print('header passed')
            pass
        else:
            #print('for row')
            #print(row)
            if int(float(row[11])) != route_id:
                result += '-2\n'
                route_id += 1
            #print(row[11])
            #print(row[12])
            isFloat = 0
            buffer = ''
            cluster = list(row[12] + '+')
            for i in cluster:
                if isFloat == 1:
                    isFloat = 0
                    pass
                else:
                    #print(i)
                    try:
                        i = int(float(i))
                    except:
                        #print('no number')
                        if buffer == '':
                            pass
                        else:
                            #print(buffer)
                            result = result + buffer + ' -1 '
                            buffer = ''
                            if i == '.':
                                isFloat = 1
                    else:
                        if isinstance(i, int):
                            if i >= 0:
                                buffer += str(i)

            #df = {'lon1': float(row[1]), 'lat1': float(row[2]), 'tstart1': float(row[3]), 'tend1': row[4], 'lon2': float(row[5]), 'lat2': float(row[6]), 'tstart2': float(row[7]), 'tend2': row[8], 'distance': row[9], 'bearing': row[10], 'route': row[11], 'classified': row[12]}
            #print(str(df))
            #line_segments = line_segments.append(df, ignore_index=True)
'''
inputSP = ''
eins = ' -1 '
zwei = '-2\n'
route = 2
for entry in line_segments.iterrows():
    if( entry[1]['route'] == route):
        cluster = str(entry[1]['classified'])
        inputSP = inputSP + cluster
        inputSP = inputSP + eins
        #print('if:' + cluster + 'String: ' + inputSP)
    else:
        route += 1
        cluster = str(entry[1]['classified'])
        inputSP = inputSP + zwei + cluster + eins
        #print('else:' + cluster + 'String: ' + inputSP)
        
'''
#print(result)

result += '-2\n'
f = open("inputprefixspan.txt", "w")
f.write(result)
f.close()

#.join(['num' for num in xrange(loop_count)])
'''
benchmark ends
'''
time = (current_milli_time() - start_time)
print("----------write_to_csv(...): BEGIN----------")
print("time consumption in ms: {}".format(time))
print("----------write_to_csv(...): END----------")
