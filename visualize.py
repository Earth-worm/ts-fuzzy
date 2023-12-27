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
    tgtCorr = tgt.corr(numeric_only = True)
    srcCorr = src.corr(numeric_only = True)
    

    tgtCorr.to_csv("result/housing/result.csv")
    rcParams['figure.figsize'] = 7,7
    sns.set(color_codes=True, font_scale=1.2)
    ax = sns.heatmap(
        srcCorr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    
    # 図の保存と図示
    plt.savefig('result/housing/src.jpg')
    plt.show()
    
    """
    columns = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"]
    for src_idx in range(len(columns)):
        for tgt_idx in range(len(columns)):
            if(src_idx == tgt_idx):
                continue
            #print("src:",np.append(np.delete(columns,[src_idx,tgt_idx]),columns[src_idx]))
            #print("tgt:",np.append(np.delete(columns,[src_idx,tgt_idx]),columns[tgt_idx]))
            #print("reduced",np.delete(columns,[src_idx,tgt_idx]))
            dir = f"{columns[src_idx]}_to_{columns[tgt_idx]}"
            print(dir)
            os.mkdir(f"result/housing/after/{dir}")
    """