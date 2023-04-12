from data import housing_USA
from utils.disp import scatter_data,scatter_2data_3d
import matplotlib.pyplot as plt
import numpy as np
from utils.optimize_algorithms.particle_swarm_optimization import PSO
from utils.ts_fuzzy.ts_fuzzy import TSFuzzy
from utils.ts_fuzzy.actfunc import ActiveFunc
from utils.data_analize import to2D
from utils.k_fold_cross_validation import Khold
from multiprocessing import Pool
from sklearn import metrics

def eval(y_obs,y_pred):
    r2 = metrics.r2_score(y_obs,y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_obs,y_pred))
    mae = metrics.mean_absolute_error(y_obs,y_pred)
    print("r2:",r2,"\nrmse",rmse,"\nmae",mae)

def func(x_tgt,y_tgt,x_test,y_test,x_src,y_src,i):
    tsM = 2.6
    tsC = 3

    M = 1.9 #1.9
    C = 6 #3
    Node = 5 #3
    ParMax = 5 #50
    Size = 150 #100
    T = 300 #100
    CG = 0.7 #0.7
    CP = 0.7 #0.7
    W = 1 #1

    model = PSO(size=Size,tMax=T,w=W,cp=CP,cg=CG,parMax=ParMax,X_src=x_src,y_src=y_src,X_tgt=x_tgt,y_tgt=y_tgt,X_test=x_test,y_test=y_test,M=M,C=C,Node=Node)
    score,y_pred_transfer,trans,_ = model.learn()
    print("iter",i)
    print("transfer")
    eval(y_test,y_pred_transfer)
    XTest2d = to2D(x_test)
    scatter_2data_3d(XTest2d,y_pred_transfer,y_test,"imgs/transfer_"+str(i)+"transfer",[20,30])

    tsfuzzy = TSFuzzy(x_src,y_src,x_tgt,y_tgt,tsM,tsC,Node,ParMax,ActiveFunc.sigmoid)
    y_pred_tsfuzzy = tsfuzzy.predict(x_test)
    print("tsfuzzy")
    eval(y_test,y_pred_tsfuzzy)
    XTest2d = to2D(x_test)
    scatter_2data_3d(XTest2d,y_pred_tsfuzzy,y_test,"imgs/tsfuzzy_"+str(i),[20,30])

if __name__ == "__main__":
    dataset = housing_USA.split_dataset()
    y_src,x_src = dataset.getData("Seattle")
    y_tgt,x_tgt = dataset.getData("Auburn")
    train_test_data = Khold(x_tgt,y_tgt,9)
    input = [i for i in train_test_data]
    with Pool() as p:
        pools = [p.apply_async(func,(input[i][0],input[i][1],input[i][2],input[i][3],x_src,y_src,i)) for i in range(9)]
        [f.get() for f in pools]