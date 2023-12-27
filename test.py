from concurrent import futures
from multiprocessing import Pool
from utils.k_fold_cross_validation import Khold
from data import housing_USA
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import os
    
if __name__ == "__main__":
    dataset = housing_USA.split_dataset()
    src = dataset.getData("Seattle")
    tgt = dataset.getData("Bellevue")
    columns = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"]
    result = np.zeros([len(columns),len(columns)])
    for src_idx in range(len(columns)):
        for tgt_idx in range(len(columns)):
            if(src_idx == tgt_idx):
                continue
            src_col = np.append(np.delete(columns,[src_idx,tgt_idx]),columns[src_idx])
            tgt_col = np.append(np.delete(columns,[src_idx,tgt_idx]),columns[tgt_idx])
            reduce_col = np.delete(columns,[src_idx,tgt_idx])
            dir = f"{columns[src_idx]}_to_{columns[tgt_idx]}"
            print(dir)
            
            
            os.mkdir(f"result/housing/after/{dir}")
    resultDF = pd.DataFrame(result,columns = columns)
    resultDF.to_excel("result/housing/after/result1.xlsx")