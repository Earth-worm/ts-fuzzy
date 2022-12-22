import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
#import seaborn as sns

def get_data(PCAN = None): #PCAN:PCAのコンポーネント数
    #"price","bedrooms","bathrooms","built","sqft_lot"
    dataset_path = os.path.dirname(__file__)+"/proced_data/ChennaiHousing2.csv"
    price = np.array([])
    bedrooms = np.array([])
    bathrooms = np.array([])
    built = np.array([])
    sqft_lot = np.array([])
    with open(dataset_path) as f:
        for line in f:
            datas = line.split(",")
            price = np.append(price,float(datas[0]))
            bedrooms = np.append(bedrooms,float(datas[1]))
            bathrooms = np.append(bathrooms,float(datas[2]))
            built = np.append(built,int(datas[3]))
            sqft_lot = np.append(sqft_lot,float(datas[4][:-2]))
        f.close()
    
    return price,np.array([bedrooms,bathrooms,built,sqft_lot])
            
def abc():
    column_names = ["price","bedrooms","bathrooms","built","sqft_lot"]
    dataset_path = os.path.dirname(__file__)+"/raw_data/ChennaiHousing2.csv"
    raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values = "", comment='\t',sep=",", skipinitialspace=True,encoding_errors='ignore'
    ,dtype={"price": int,"bedrooms":float,"bathrooms":float,"built":float,"sqft_lot":float})
    #print(raw_dataset.dtypes)
    raw_dataset = raw_dataset[1:]
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    return dataset



if __name__ == "__main__":
    dataset = get_data()
    print(dataset)


    
