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
line_segments = pd.DataFrame(columns=['lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2', 'distance', 'bearing', 'route', 'classified'])
global result
result = ''
with open('C:\\Users\\markb\\Documents\\GitHub\\dbpro2\\v2\\test_test_seg.csv', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        route_id = 1 
        limit = 10
        i = 0
        for row in reader:
            i += 1
            while (i < limit):
                print('for row')
                if row[11] == route_id:
                    print('if')
                    print(row[11])
                    cluster = str(row[11])
                    result =  result + cluster + ' -1 '
                    print('if:' + cluster + 'String: ' + result)
                    break
                else:
                    print('else')
                    route_id += 1
                    cluster = str(row[11])
                    result +=  '-2\n' + cluster + ' -1 '
                    print('else:' + cluster + 'String: ' + result)
                    
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