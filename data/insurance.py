
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os

def get_data():
    # ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    dataset_path = os.path.dirname(__file__)+"/proced_data/insurance.csv"
    raw_dataset = pd.read_csv(dataset_path)
    dataset = raw_dataset.copy()
    dataset = dataset.drop(columns=['sex', 'smoker'])
    dataset = dataset.dropna()
    return dataset

if __name__ == "__main__":
    data = get_data()
    src = data[data['region'] == "southeast" ]
    tgt = data[data['region'] == "northwest" ]
    print(src,tgt)