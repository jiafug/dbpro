# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:55:12 2019

@author: markb
"""
import csv
import time

DATA_PATH = "C:\\Users\\markb\\Documents\\GitHub\\dbpro2\\v2\\test_seg.csv"
RESULT_PATH = "C:\\Users\\markb\\Documents\\GitHub\\dbpro2\\v2\\clusterAvg.csv"

current_milli_time = lambda: int(round(time.time() * 1000))
    

def main():
    # logging
    start_time = current_milli_time()
    
    with open(DATA_PATH) as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        #print('input_file opend')
        #distance_list = []
        #time = 0

        # 'index', 'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 
        # 'tend2', 'distance', 10:'bearing', 'route', 'classified'

        # logging
        #past_time = current_milli_time()
        print("started... " + str(start_time))
        #counter = 0

        # read one csv line after another
        with open(RESULT_PATH, 'a') as result_file:
            result_file.write('cluster, Average Distance, Line Count, lon1, lat1, lon2, lat2')     
            print('result_file wrote header')
            max = 10
            avgTotal = 0.0
            countTotal = 0
            for i in range(0, max):
                avg = 0.0
                count = 0
                cluster = ''
                lon1 =  ''
                lat1 =  ''
                lon2 =  ''
                lat2 =  ''
                for row in csv_reader:
                    # ignore header
                    print(row)
                    if row[1] != 'lon1':
                        # adapt max
                        cluster = list(row[12])
                        
                        for j in cluster:
                            print(j)
                            try:
                                j = int(j)
                            except:
                                print('no number')
                            else:
                                if isinstance(j, int):
                                    if int(j) < max:
                                        max = int(j)
                                    if int(j) == i:
                                        avg += float(row[9])
                                        count += 1  
                                        cluster = j
                                        lonlat =  (row[1] + ', ' + row[2] + ', ' + row[5] + ', ' + row[6] + ', ')
                if count > 0:
                    avg = avg / count
                avgTotal += avg
                countTotal = i
                line = (str(cluster) + ', ' + str(avg) + ', ' + str(count) + ', ' + lonlat)
                print(line)
                result_file.write(line)
            avgTotal = avgTotal / countTotal
            print('avg: ' + str(avgTotal) + '; Count: ' + str(countTotal))
            end_time = current_milli_time()
            sek = str(end_time - start_time)
            print('distanceAverage finished after ' + sek + ' sec.')
    

if __name__ == "__main__":
    main()
            