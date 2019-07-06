# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:55:12 2019

@author: markb
"""
import csv
import time

DATA_PATH = "C:\\Users\\markb\\Documents\\GitHub\\dbpro2\\v2\\test_seg.csv"
RESULT_PATH = "C:\\Users\\markb\\Documents\\GitHub\\dbpro2\\v2\\clusterAvg.csv"
sep = '; '

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
        print("started...") #+ str(start_time))
        #counter = 0

        # read one csv line after another
        # 'a' for append, 'w' would override the file every time
        with open(RESULT_PATH, 'a') as result_file:
            #result_file.write('/n')
            #result_file.write('/n')
            #result_file.write('/n')
            header = ('cluster' + sep + 'Average Distance' + sep + 'Line Count' + sep + 'lon1' + sep + 'lat1' + sep + 'lon2' + sep + 'lat2')
            result_file.write(header)     
            print('result_file wrote header')
            # max should be small and adjust in runtime
            # isnt working right now
            max = 1000000
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
                                    if int(j) > max:
                                        max = int(j)
                                    if int(j) == i:
                                        avg += float(row[9])
                                        count += 1  
                                        cluster = j
                                        lonlat =  (row[1] + sep + row[2] + sep + row[5] + sep + row[6] + sep)
                if count > 0:
                    avg = avg / count
                    countTotal += 1
                avgTotal += avg
                #countTotal = i
                line = (str(cluster) + sep + str(avg) + sep + str(count) + sep + lonlat + sep + '/n')
                print(line)
                result_file.write(line)
                print('Starting ' + str(i) + '. cluster / ' + str(max) + '.')
            avgTotal = avgTotal / countTotal
            print('avg: ' + str(avgTotal) + '; Count: ' + str(countTotal))
            end_time = current_milli_time()
            #time = (current_milli_time() - start_time) 
            sek = str(end_time - start_time)
            print("----------write_to_csv(...): BEGIN----------")
            print('time consumption in s:' + str(sek))
            print("----------write_to_csv(...): END----------")
            #print('distanceAverage finished after ' + sek + ' sec.')
    

if __name__ == "__main__":
    main()
            