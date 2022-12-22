import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os

def get_data(PCAN = None): #PCAN:PCAのコンポーネント数
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
    dataset_path = os.path.dirname(__file__)+"/proced_data/auto-mpg.data"
    raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values = "?", comment='\t',sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()

    raw_tgt_data = dataset[dataset["Model Year"] == 82]
    ytgt = raw_tgt_data["MPG"]
    raw_src_data = dataset[dataset["Model Year"] != 82]
    ysrc = raw_src_data["MPG"]
    src_data = np.array([raw_src_data['Cylinders'],raw_src_data['Displacement'],raw_src_data['Weight'],raw_src_data['Horsepower']]).T
    tgt_data = np.array([raw_tgt_data['Cylinders'],raw_tgt_data['Displacement'],raw_tgt_data['Weight'],raw_tgt_data['Horsepower']]).T
    Xsrc = src_data
    Xtgt = tgt_data
    pca1 = PCA(n_components=2).fit(src_data)
    pca2 = PCA(n_components=2).fit(tgt_data)
    Xsrc2d = pca1.transform(src_data)
    Xtgt2d = pca2.transform(tgt_data)
    return Xsrc,ysrc,Xtgt,ytgt,Xsrc2d,Xtgt2d

