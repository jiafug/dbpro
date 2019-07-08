import matplotlib.pyplot as plt
import pandas as pd
import mplleaflet

fig =plt.figure()

with open('/Users/jiafug/Desktop/DBPRO/Codev2/results.txt') as results:
    data=pd.read_csv("/Users/jiafug/Desktop/DBPRO/Codev2/representative_trajectories.csv",
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
        
        if len(final) >= 3:
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
            plt.plot(x_list,y_list)
            plt.plot(x_list,y_list, 'ro')
        # break
                                   
    plt.savefig('foo.png')

    mplleaflet.show(fig=fig)