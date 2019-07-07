import matplotlib.pyplot as plt
import pandas as pd
fig =plt.figure()
with open('/Users/karenveronica/Desktop/Uni/SS19/dbpro-master/SPMFResults/inputprefixspanm.txt') as results:
    data=pd.read_csv("/Users/karenveronica/Desktop/Uni/SS19/dbpro-master/SPMFResults/representative_trajectories.csv")
    x_long=[]
    y_lat=[]
    
    for i in results:
        line=i
        line=line.replace("-1","").replace("-2","").replace(" ","")
        print(line)
       
        for i in line:
            if i != "\n":
                ID_results=int(i)
                #print('i',i)                    
                for j in data.iterrows():
                    ID=int(j[1][0])
                    if(ID_results==ID):
                        x_long.insert(ID_results,j[1][1])
                        y_lat.insert(ID_results,j[1][2])
                        plt.plot(x_long,y_lat)
                    
                   
                    
                     #   x_long.insert(ID_results,j[1][1])
                      #  y_lat.insert(ID_results,j[1][2])
                       # plt.plot(x_long,y_lat)
                        
    plt.show()  