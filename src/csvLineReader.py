# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:08:27 2019

@author: markb
"""

import csv

with open('C:\\Users\\markb\\Documents\\DBPRO\\raw_data\\train.csv', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        counter = 0
        for row in reader:   
            counter = counter + 1
            if counter < 10: print (counter)
            if counter == 100: print (counter)
            if counter == 5000: print (counter)
            if counter == 10000: print (counter)
            if counter == 50000: print (counter)
            if (counter % 100000) == 0 : print (counter) 
            if counter > 1710000 : print (counter)