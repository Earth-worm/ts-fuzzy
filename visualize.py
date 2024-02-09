from concurrent import futures
from multiprocessing import Pool
from utils.k_fold_cross_validation import Khold
from data import housing_USA,auto_mpg,insurance
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == "__main__":
    """
    # housing
    housing_dataset = housing_USA.split_dataset()
    housing_tgt = housing_dataset.getData("Bellevue")
    columns = ["住宅価格","寝室数","バスルーム数","リビング面積","敷地面積","フロア数","屋根裏部屋面積","地下室面積","建築年"]
    housing_tgt.columns = columns
    #housing_src = housing_dataset.getData("Seattle")
    housing_tgt_corr = housing_tgt.corr(numeric_only = True)
    #housing_src_corr = housing_src.corr(numeric_only = True)
    
    rcParams['figure.figsize'] = 12,12
    sns.set(color_codes=True, font_scale=1.2,font="Yu Gothic")
    ax = sns.heatmap(
        housing_tgt_corr, 
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
    plt.savefig('result/housing/tgt.jpg')
    #plt.show()
    """
    
    mpg_dataset = auto_mpg.get_data()
    columns = ['燃費','気筒数','排気量','馬力','重さ','速度', 'Model Year']
    mpg_dataset.columns = columns
    mpg_tgt = mpg_dataset[mpg_dataset["Model Year"] > 80]
    mpg_tgt = mpg_tgt.drop(mpg_tgt.index[-1])
    mpg_tgt = mpg_tgt.drop(columns = ["Model Year"],axis = 1)
    #mpg_src = mpg_dataset[mpg_dataset["Model Year"] <= 80]
    mpg_tgt_corr = mpg_tgt.corr(numeric_only = True)
    #mpg_src_corr = mpg_src.corr(numeric_only = True)
    
    rcParams['figure.figsize'] = 12,12
    sns.set(color_codes=True, font_scale=1.2,font="Yu Gothic")
    ax = sns.heatmap(
        mpg_tgt_corr, 
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
    plt.savefig('result/mpg/tgt.jpg')
    """
    
    insurance_dataset = insurance.get_data()
    insurance_dataset.columns = ['年齢', 'BMI', '子供の数', 'region','医療コスト']
    #insurance_src = insurance_dataset[insurance_dataset['region'] == "southwest" ]
    insurance_tgt = insurance_dataset[insurance_dataset['region'] == "northwest" ]
    insurance_tgt = insurance_tgt.drop(columns=['region'])
    insurance_tgt_corr = insurance_tgt.corr(numeric_only = True)
    #mpg_src_corr = mpg_src.corr(numeric_only = True)
    
    rcParams['figure.figsize'] = 12,12
    sns.set(color_codes=True, font_scale=1.2,font="Yu Gothic")
    ax = sns.heatmap(
        insurance_tgt_corr, 
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
    plt.savefig('result/insurance/tgt.jpg')
    """