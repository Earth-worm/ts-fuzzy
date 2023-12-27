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
import os

M = 2.1
C = 5
Node = 5
ParMax = 5
Size = 300
TMax = 1000
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

def noTLLearn(input):
    x_src = input[0]
    y_src = input[1]
    x_test = input[2]
    y_test = input[3]
    ts = TSFuzzy(x_src,y_src,None,None,M,C,Node,parMax,ActFunc)
    y_pred = ts.predict(x_test)
    return [metrics.mean_squared_error(y_pred,y_test),metrics.r2_score(y_pred,y_test)]

def NoTL(x_tgt,y_tgt,dir):
    khold = Khold(x_tgt.T,y_tgt,10)
    i = 0
    result = []
    arg_list = []
    for x_train,y_train,x_test,y_test in khold:
        arg_list.append([x_train,y_train,x_test,y_test])
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv(f"{dir}/notl.csv")
    return np.mean(df["mean squared error"])

def ReduceFeatureTL(x_tgt,y_tgt,x_src,y_src,dir):
    khold = Khold(x_tgt.T,y_tgt,10)
    i = 0
    result = []
    arg_list = []
    for x_train,y_train,x_test,y_test in khold:
        arg_list.append([x_train,y_train,x_test,y_test,x_src,y_src])
    with Pool(processes=6) as p:
        result = p.map(func=learn,iterable=arg_list)
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv(f"{dir}/reduce.csv")
    return np.mean(df["mean squared error"])
    
def MappingFeatureTL(x_tgt,y_tgt,x_src,y_src,dir):
    khold = Khold(x_tgt.T,y_tgt,10)
    i = 0
    result = []
    arg_list = []
    for x_train,y_train,x_test,y_test in khold:
        arg_list.append([x_train,y_train,x_test,y_test,x_src,y_src])
    with Pool(processes=6) as p:
        result = p.map(func=learn,iterable=arg_list)
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv(f"{dir}/mapping.csv")
    return np.mean(df["mean squared error"])

def AllFeatureTL(x_tgt,y_tgt,x_src,y_src,dir):
    khold = Khold(x_tgt.T,y_tgt,10)
    result = []
    arg_list = []
    for x_train,y_train,x_test,y_test in khold:
        arg_list.append([x_train,y_train,x_test,y_test,x_src,y_src])
    with Pool(processes=6) as p:
        result = p.map(func=learn,iterable=arg_list)
    df = pd.DataFrame(result,columns=["mean squared error","R"])
    df.to_csv(f"{dir}/all.csv")
    return np.mean(df["mean squared error"])

if __name__ == "__main__":
    dataset = housing_USA.split_dataset()
    src = dataset.getData("Seattle")
    tgt = dataset.getData("Bellevue")
    columns = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"]
    no_tl_result = np.zeros([len(columns),len(columns)])
    mapping_result = np.zeros([len(columns),len(columns)])
    reduce_result = np.zeros([len(columns),len(columns)])
    for src_idx in range(len(columns)):
        for tgt_idx in range(len(columns)):
            if(src_idx == tgt_idx):
                continue
            src_col = np.append(np.delete(columns,[src_idx,tgt_idx]),columns[src_idx])
            tgt_col = np.append(np.delete(columns,[src_idx,tgt_idx]),columns[tgt_idx])
            reduce_col = np.delete(columns,[src_idx,tgt_idx])
            y_src = src["price"]
            y_tgt = tgt["price"]
            
            x_src = []
            x_tgt = []
            x_src_reduce = []
            x_tgt_reduce = []
            for col in src_col:
                x_src.append(src[col])
            for col in tgt_col:
                x_tgt.append(tgt[col])
            for col in reduce_col:
                x_src_reduce.append(src[col])
                x_tgt_reduce.append(tgt[col])
            x_src = np.array(x_src)
            x_tgt = np.array(x_tgt)
            x_src_reduce = np.array(x_src_reduce)
            x_tgt_reduce = np.array(x_tgt_reduce)
            
            dir = f"result/housing/after/{columns[src_idx]}_to_{columns[tgt_idx]}"
            print((src_idx)*len(columns) + (tgt_idx + 1),"/",len(columns)*len(columns),"   ",dir)
            os.mkdir(dir)
            
            print("no tl")
            no_tl_result[src_idx][tgt_idx] = NoTL(x_tgt,y_tgt,dir)
            
            print("reduce")
            reduce_result[src_idx][tgt_idx] = ReduceFeatureTL(x_tgt_reduce,y_tgt,x_src_reduce,y_src,dir)
            
            print("mapping")
            mapping_result[src_idx][tgt_idx]  = MappingFeatureTL(x_tgt,y_tgt,x_src,y_src,dir)
            
    resultDF = pd.DataFrame(no_tl_result,columns = columns)
    resultDF.to_excel("result/housing/after/no_tl_result.xlsx")
    
    resultDF = pd.DataFrame(reduce_result,columns = columns)
    resultDF.to_excel("result/housing/after/reduce_result.xlsx")
    
    resultDF = pd.DataFrame(mapping_result,columns = columns)
    resultDF.to_excel("result/housing/after/mapping_result.xlsx")