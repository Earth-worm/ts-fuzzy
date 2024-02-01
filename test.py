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
import japanize_matplotlib
    
if __name__ == "__main__":
    dataset = housing_USA.split_dataset()
    tgt = dataset.getData("Bellevue")
    columns = ["住宅価格","寝室数","バスルーム数","リビング面積","敷地面積","フロア数","屋根裏部屋面積","地下室面積","建築年"]
    tgt.columns = columns
    
    corr = tgt.corr(numeric_only = True)
    rcParams['figure.figsize'] = 12,12
    sns.set(color_codes=True, font_scale=1.2,font="Yu Gothic")
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(0, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontdict={"fontweight":"bold"}
    )
    ax.set_yticklabels(ax.get_yticklabels(),fontdict={"fontweight":"bold"})
    # 図の保存と図示
    plt.savefig('result/housing_tgt.png')
    plt.show()
    print(plt.rcParams['font.family'])