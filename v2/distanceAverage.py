# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:55:12 2019

@author: markb
"""


DATA_PATH = "C:\\Users\\markb\\Documents\\DBPRO\\raw_data\\test.csv"

 # logging
start_time = current_milli_time()
    

def main():
    # logging
    global start_time
    
    with open(DATA_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        distance_list = []
        time = 0

        final_seg = pd.DataFrame(columns=[
                    'lon1', 'lat1', 'tstart1', 'tend1', 'lon2', 'lat2', 'tstart2', 'tend2',
                    'distance', 'bearing', 'route', 'classified'])

        # logging
        past_time = current_milli_time()
        logger.info("started...")
        counter = 0

        # read one csv line (trajectory) after another
        for row in csv_reader:
            
            # test break
            # if trajectory_count == 10:
            #    break

            # ignore header
            if cleared != "POLYLINE":