import pandas as pd
import numpy as np
from utils.disp import scatter_data
from minepy import MINE
from sklearn.decomposition import PCA

def pearson_coef(d1,d2):
    s1 = pd.Series(d1)
    s2 = pd.Series(d2)
    res = s1.corr(s2)
    return res

def mine_coef(d1,d2):
    mine = MINE()
    mine.compute_score(d1,d2)
    return mine.mic()

def correlation(d1,d2,fig=None):
    if(len(d1)!=len(d2)):
        print("入力データのサイズが異なります。\ninput1:",len(d1),"\ninput2:",len(d2))
        exit()
    if fig is not None:
        scatter_data(d1,d2,fig)
    pear = pearson_coef(d1,d2)
    mine = mine_coef(d1,d2)
    return pear,mine

def to2D(data):
    pca = PCA(n_components=2).fit(data)
    return pca.transform(data)
