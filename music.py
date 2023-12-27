from data import housing_USA
from utils.disp import scatter_data_3d,scatter_2data_3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.k_fold_cross_validation import Khold
from utils.data_analize import correlation,to2D
from utils.ts_fuzzy.ts_fuzzy import TSFuzzy
from utils.ts_fuzzy.actfunc import ActiveFunc
from utils.optimize_algorithms.particle_swarm_optimization import PSO
import sklearn.metrics as metrics
from multiprocessing import Pool
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

M = 2.1
C = 5
Node = 5
ParMax = 5
Size = 150 #300
TMax = 500 #1200
W = 0.9
CP = 0.8
CG = 0.6

def learn(input):
    x_train = input[0]
    y_train = input[1]
    x_test = input[2]
    y_test = input[3]
    x_src = input[4]
    y_src = input[5]
    pso = PSO(size=Size,
            tMax=TMax,
            w=W,
            cp=CP,
            cg=CG,
            parMax=ParMax,
            X_src=x_src,
            y_src=y_src,
            X_tgt=x_train,
            y_tgt=y_train,
            X_test=x_test,
            y_test=y_test,
            M=M,
            C=C,
            Node=Node,
            ActFunc=ActiveFunc.sigmoid)
    gBestValue,y_pred,trans,gBestPos = pso.learn()
    return [metrics.mean_squared_error(y_pred,y_test),metrics.r2_score(y_pred,y_test)]
    
def NoTL(x_tgt,y_tgt):
    khold = Khold(x_tgt.T,y_tgt,10)
    arg_list = []
    for x_test,y_test,x_train,y_train in khold:
        learn([x_train,y_train,x_test,y_test,None,None])
        arg_list.append([x_train,y_train,x_test,y_test,None,None])
    """
    with Pool(processes=6) as p:
        result = p.map(func=learn,iterable=arg_list)
    ## info x_train.size < x_test.size
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv("result/music/notl.csv")
    """

def ReduceFeatureTL(x_tgt,y_tgt,x_src,y_src):
    khold = Khold(x_tgt.T,tgt_y,10)
    result = []
    arg_list = []
    for x_test,y_test,x_train,y_train in khold:
        arg_list.append([x_train,y_train,x_test,y_test,x_src,y_src])
    with Pool(processes=6) as p:
        result = p.map(func=learn,iterable=arg_list)
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv("result/housing/reduce.csv")
    
def MappingFeatureTL(x_tgt,y_tgt,x_src,y_src):
    khold = Khold(x_tgt.T,tgt_y,10)
    arg_list = []
    for x_test,y_test,x_train,y_train in khold:
        arg_list.append([x_train,y_train,x_test,y_test,x_src,y_src])
    with Pool(processes=6) as p:
        result = p.map(func=learn,iterable=arg_list)
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv("result/housing/mapping.csv")

def AllFeatureTL(x_tgt,y_tgt,x_src,y_src):
    khold = Khold(x_tgt.T,tgt_y,10)
    arg_list = []
    for x_test,y_test,x_train,y_train in khold:
        arg_list.append([x_train,y_train,x_test,y_test,x_src,y_src])
    with Pool(processes=6) as p:
        result = p.map(func=learn,iterable=arg_list)
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv("result/music/all.csv")
    print(result)

def get_data(filename):
    missing_values = ["n/a", "na", "--", " ", "N/A", "NA"]
    return pd.read_csv(filename,na_values = missing_values,sep=",")

if __name__ == "__main__":
    originalParty = get_data("data/proced_data/spotify/genre/party.csv")
    party = originalParty.copy()

    originalPunk = get_data("data/proced_data/spotify/genre/punk.csv")
    punk = originalPunk.copy()
    
    partyCorr = party.corr(numeric_only = True)
    rcParams['figure.figsize'] = 7,7
    sns.set(color_codes=True, font_scale=1.2)
    ax = sns.heatmap(
        partyCorr, 
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
    plt.savefig('result/music_src.png')
    plt.show()
    
    punkCorr = punk.corr(numeric_only = True)
    rcParams['figure.figsize'] = 7,7
    sns.set(color_codes=True, font_scale=1.2)
    ax = sns.heatmap(
        punkCorr, 
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
    plt.savefig('result/music_tgt.png')
    plt.show()
    
    src_y = np.array(party['popularity'])
    src_x = np.array([
        party['year'],
        party['danceability'],
        party['key'],
        party['loudness'],
        party['mode'],
        party['speechiness'],
        party['acousticness'],
        party['instrumentalness'],
        party['liveness'],
        party['valence'],
        party['tempo'],
        party['duration_ms'],
        party['time_signature'],
    ])
    
    tgt_y = np.array(punk['popularity'])
    tgt_x = np.array([
        punk['year'],
        punk['danceability'],
        punk['key'],
        punk['loudness'],
        punk['mode'],
        punk['speechiness'],
        punk['acousticness'],
        punk['instrumentalness'],
        punk['liveness'],
        punk['valence'],
        punk['tempo'],
        punk['duration_ms'],
        punk['time_signature'],
    ])
    
    # mapping danceability -> tempo
    mapping_src_x = np.array([
        party['year'],
        party['danceability'],
        party['key'],
        party['loudness'],
        party['mode'],
        party['speechiness'],
        party['acousticness'],
        party['instrumentalness'],
        party['liveness'],
        party['valence'],
        party['duration_ms'],
        party['time_signature'],
    ])
    
    mapping_tgt_x = np.array([
        punk['year'],
        punk['tempo'],
        punk['key'],
        punk['loudness'],
        punk['mode'],
        punk['speechiness'],
        punk['acousticness'],
        punk['instrumentalness'],
        punk['liveness'],
        punk['valence'],
        punk['duration_ms'],
        punk['time_signature'],
    ])
    
    reduced_src_x = np.array([
        party['year'],
        party['key'],
        party['loudness'],
        party['mode'],
        party['speechiness'],
        party['acousticness'],
        party['instrumentalness'],
        party['liveness'],
        party['valence'],
        party['duration_ms'],
        party['time_signature'],
    ])
    
    reduced_tgt_x = np.array([
        punk['year'],
        punk['key'],
        punk['loudness'],
        punk['mode'],
        punk['speechiness'],
        punk['acousticness'],
        punk['instrumentalness'],
        punk['liveness'],
        punk['valence'],
        punk['duration_ms'],
        punk['time_signature'],
    ])
    
    """
    #notl 
    print("no tl")
    print(mapping_tgt_x.shape,tgt_y.shape)
    NoTL(mapping_tgt_x,tgt_y)
    
    #redice
    print("reduce")
    ReduceFeatureTL(reduced_tgt_x,tgt_y,reduced_src_x,src_y)
    
    #mapping
    print("mapping")
    MappingFeatureTL(mapping_tgt_x,tgt_y,mapping_src_x,src_y)
    
    #all
    print("all")
    AllFeatureTL(tgt_x,tgt_y,src_x,src_y)
    """