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
    dataset = dataset.drop('Origin',axis=1)
    return dataset

