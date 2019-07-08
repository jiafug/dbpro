import matplotlib.pyplot as plt
import pandas as pd
import mplleaflet
import math

fig =plt.figure()

with open('results.txt') as results:
    data=pd.read_csv("representative_trajectories.csv",
                     sep=";",
                     index_col=0,
                     header=None, 
                     names=['lon1', 'lat1', 'lon2', 'lat2', 'min_x1', 'max_x1', 'min_x2', 'max_x2'])
        
    for i in results:
        line=i
        line=line.replace("-1","").replace("-2","")
        # print(line)
        entries = line.split("#")
        abc = entries[1].split(": ")
        sup = abc[1].replace("\n", "")
        sup = int(sup)
        # print(int(sup))
        final = entries[0].split("  ")
        del final[-1]
        # print(final)
        
        x_list = []
        y_list = []
        
        for i in final:
            save = data.loc[int(i)]
            lon1 = save['lon1']
            lat1 = save['lat1']
            lon2 = save['lon2']
            lat2 = save['lat2']
                
            x_list += [lon1, lon2]
            y_list += [lat1, lat2]
                
            #print(x_list)
            #print(y_list)
        
        line_size = math.log(sup)
        color = ""
        if sup >= 700
            color = "red"
        if sup < 700:
            color = "orange"
        elif sup < 500:
            color = "lime"
        elif sup < 400:
            color = "green"
        elif sup < 300:
            color = "blue"
        plt.plot(x_list,y_list, linewidth=2, color=color)
        # plt.plot(x_list,y_list, 'ro',  markersize=1.5)
        # break
                                   
    plt.savefig('foo.png')

    mplleaflet.show(fig=fig)