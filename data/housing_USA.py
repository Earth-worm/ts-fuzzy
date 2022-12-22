import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from scipy.stats import iqr
import statistics
import matplotlib.pyplot as plt
#import seaborn as sns

#[price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,yr_built]
def get_data(PCAN = None): #PCAN:PCAのコンポーネント数
    dataset_path = os.path.dirname(__file__)+"/proced_data/USAHousing2.csv"
    price = np.array([])
    bedrooms = np.array([])
    bathrooms = np.array([])
    sqft_lot = np.array([])
    sqft_living= np.array([])
    floors= np.array([])
    sqft_above= np.array([])
    sqft_basement= np.array([])
    yr_built= np.array([])
    n = 3
    best = [ 0.0 for i in range(n) ]
    with open(dataset_path) as f:
        for line in f:
            datas = line.split(",")
            p = float(datas[0])
            for i in range(n):
                if best[i] > p:
                    continue
                else:
                    p,best[i] = best[i],p
        f.close()
    with open(dataset_path) as f:
        for line in f:
            datas = line.split(",")
            flag = False
            for b in best:
                flag = flag or (b == float(datas[0]))
            if flag:
                continue
            price = np.append(price,float(datas[0]))
            bedrooms = np.append(bedrooms,float(datas[1]))
            bathrooms = np.append(bathrooms,float(datas[2]))
            sqft_living = np.append(sqft_living,float(datas[3]))
            sqft_lot = np.append(sqft_lot,float(datas[4]))
            floors = np.append(floors,float(datas[5]))
            sqft_above = np.append(sqft_above,float(datas[6]))
            sqft_basement = np.append(sqft_basement,float(datas[7]))
            yr_built = np.append(yr_built,float(datas[8][:-1]))
        f.close()
    return price,np.array([bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,yr_built])

class split_dataset():
    class Dataset():
        class OneDataInfo():
            Avg = None
            Mdn = None
            Iqr = None
            Range = None
            def __init__(self,Avg,Mdn,Iqr,Range):
                self.Avg = Avg
                self.Mdn = Mdn
                self.Iqr = Iqr
                self.Range = Range

            def __str__(self):
                rtn = "Avg: {0}\nMdn: {1}\nIqr: {2}\nRange: {3}\n".format(self.Avg,self.Mdn,self.Iqr,self.Range)
                return rtn

        input_str = None
        output_str = None
        size = None
        output = None
        input = None
        eachDataInfo = None

        def __init__(self,output,input):
            self.output = output
            self.input = input
            self.size = len(output)
            self.input_str = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"]
            self.output_str = "price"

        def __str__(self):
            return "test!"

        def analyze(self):
            self.eachDataInfo = dict({})
            self.eachDataInfo[self.output_str] = self.analyzeOneVector(self.output)
            for str,d in zip(self.input_str,self.input):
                self.eachDataInfo[str] = self.analyzeOneVector(d)

        def analyzeOneVector(self,data):
            Avg = statistics.mean(data)
            Mdn = statistics.median(data)
            Iqr = iqr(data)
            Range = [np.min(data),np.max(data)]
            return self.OneDataInfo(Avg,Mdn,Iqr,Range)

    dataset = None
    def __init__(self):
        dataset = dict({})
        files = os.listdir(os.path.dirname(__file__)+"/proced_data/USA_housing_split")
        cities = []
        for file in files:
            cities.append(file.replace(".csv",""))
            price = np.array([])
            bedrooms = np.array([])
            bathrooms = np.array([])
            sqft_lot = np.array([])
            sqft_living= np.array([])
            floors= np.array([])
            sqft_above= np.array([])
            sqft_basement= np.array([])
            yr_built= np.array([])
            with open(os.path.dirname(__file__)+"/proced_data/USA_housing_split/"+file) as f:
                for row in f:
                    split_row = row.split(",")
                    price = np.append(price,float(split_row[0]))
                    bedrooms = np.append(bedrooms,float(split_row[1]))
                    bathrooms = np.append(bathrooms,float(split_row[2]))
                    sqft_living = np.append(sqft_living,float(split_row[3]))
                    sqft_lot = np.append(sqft_lot,float(split_row[4]))
                    floors = np.append(floors,float(split_row[5]))
                    sqft_above = np.append(sqft_above,float(split_row[6]))
                    sqft_basement = np.append(sqft_basement,float(split_row[7]))
                    yr_built = np.append(yr_built,float(split_row[8][:-1]))
                f.close()
            dataset[file.replace(".csv","")] = self.Dataset(price,np.array([bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,yr_built]))
        self.dataset = dataset
        self.cities = cities

    def getAllCity(self):
        return self.cities

    def getData(self,tarCity):
        if(tarCity not in self.cities):
            print("not exist such a city :",tarCity)
            exit()
        return self.dataset[tarCity].output,self.dataset[tarCity].input.T

    def getInfo(self,tarCity):
        if(tarCity not in self.cities):
            print("not exist such a city :",tarCity)
            exit()
        if(self.dataset[tarCity].eachDataInfo is None):
            self.dataset[tarCity].analyze()
        return self.dataset[tarCity]

def compareEachCity(city):
    dataset = split_dataset()
    for d,j in zip(["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"],range(8)):
        fig = plt.figure(figsize=(8, 6), dpi=80)
        fig.suptitle(d+"-price")
        xlim = [10000000000000,0]
        ylim = [10000000000000,0]
        for c in city:
            output,input = dataset.getData(c)
            xlim = [min(xlim[0],min(input[j])),max(xlim[1],max(input[j]))]
            ylim = [min(ylim[0],min(output)),max(ylim[1],max(output))]
        for c,i in zip(city,range(2)):
            output,input = dataset.getData(c)
            ax = fig.add_subplot(1,2,i+1)
            ax.scatter(input[j],output)
            #print(c,"\ndatasize:",info.size,"\n",info.eachDataInfo[d])
            ax.set_title(c)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        plt.savefig("imgs/city/"+city[1]+"/"+d)
        plt.clf()
        plt.close()

if __name__ == "__main__":
    dataset = get_data()
    print(dataset)



