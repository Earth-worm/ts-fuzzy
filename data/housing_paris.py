import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
#import seaborn as sns

def get_data(PCAN = None): #PCAN:PCAのコンポーネント数
    column_names = ['squareMeters','numberOfRooms','hasYard','hasPool','floors','cityCode', 'cityPartRange', 'numPrevOwners',
    'made','isNewBuilt','hasStormProtector','basement','attic','garage','hasStorageRoom','hasGuestRoom','price']
    dataset_path = os.path.dirname(__file__)+"/proced_data/ParisHousing.csv"
    raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values = "?", comment='\t',sep=",", skipinitialspace=True)
    raw_dataset = raw_dataset[1:]
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    return dataset
"""  

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
    return Xsrc,ysrc,Xtgt,ytgt,Xsrc2d,Xtgt2d"""

if __name__ == "__main__":
    dataset = get_data()
    plt.hist(dataset["squareMeters"])
    plt.ylabel("squareMeters")
    plt.xlabel("num")
    plt.show()

    
