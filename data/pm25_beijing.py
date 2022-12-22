import numpy as np
import os
#import seaborn as sns

def get_data(PCAN = None): #PCAN:PCAのコンポーネント数
    #pm2.5,year,month,day,hour,DEWP,TEMP,PRES,Iws,Is,Ir
    dataset_path = os.path.dirname(__file__)+"/proced_data/Beijing_PM252.csv"
    pm25 = np.array([])
    year = np.array([])
    month = np.array([])
    day = np.array([])
    hour = np.array([])
    dewp = np.array([])
    temp = np.array([])
    pres = np.array([])
    Iws = np.array([])
    Is = np.array([])
    Ir = np.array([])
    with open(dataset_path) as f:
        for line in f:
            datas = line.split(",")
            pm25 = np.append(pm25,float(datas[0]))
            year = np.append(year,int(datas[1]))
            month = np.append(month,int(datas[2]))
            day = np.append(day,int(datas[3]))
            hour = np.append(hour,int(datas[4]))
            dewp = np.append(dewp,float(datas[5]))
            temp = np.append(temp,float(datas[6]))
            pres = np.append(pres,float(datas[7]))
            Iws = np.append(Iws,float(datas[8]))
            Is = np.append(Is,float(datas[9]))
            Ir = np.append(Ir,float(datas[10][:-1]))
        f.close()
    
    return pm25,np.array([year,month,day,hour,dewp,temp,pres,Iws,Is,Ir])