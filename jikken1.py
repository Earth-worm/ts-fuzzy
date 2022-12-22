from data import housing_USA
from utils.disp import scatter_data,scatter_2data_3d
import matplotlib.pyplot as plt
import numpy as np
from utils.optimize_algorithms.particle_swarm_optimization import PSO
from utils.data_analize import to2D
from utils.k_fold_cross_validation import Khold
from multiprocessing import Pool


def func(x_tgt,y_tgt,x_test,y_test,x_src,y_src,i):
    M = 1.9 #1.9
    C = 5#3
    Node = 3 #3
    ParMax = 100 #50
    Size = 300 #100
    T = 500 #100
    CG = 0.5 #0.7
    CP = 0.5 #0.7
    W = 1 #1
    print("iter=",i)
    model =PSO(size=Size,tMax=T,w=W,cp=CP,cg=CG,parMax=ParMax,X_src=x_src,y_src=y_src,X_tgt=x_tgt,y_tgt=y_tgt,X_test = x_test,y_test=y_test,M=M,C=C,Node=Node)#ここから、データスプリットの入力の型がおかしいかも
    score,y_pred,trans,_ = model.learn()
    """
    f = open("test.txt","w")
    for t in trans:
        f.write("{}\n".format(t))
    f.close()
    """
    print(score)
    XTest2d = to2D(x_test)
    print(trans)
    scatter_2data_3d(XTest2d,y_pred,y_test,"imgs/testttt"+str(i),[20,30])

if __name__ == "__main__":
    dataset = housing_USA.split_dataset()
    y_src,x_src = dataset.getData("Seattle")
    y_tgt,x_tgt = dataset.getData("Auburn")
    train_test_data = Khold(x_tgt,y_tgt,9)
    input = [i for i in train_test_data]
    with Pool() as p:
        pools = [p.apply_async(func,(input[i][0],input[i][1],input[i][2],input[i][3],x_src,y_src,i)) for i in range(9)]
        [f.get() for f in pools]