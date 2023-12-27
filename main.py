from data import housing_USA
from utils.disp import scatter_data_3d,scatter_2data_3d
import matplotlib.pyplot as plt
import numpy as np
from utils.k_fold_cross_validation import Khold
from utils.data_analize import correlation,to2D
from utils.ts_fuzzy.ts_fuzzy import TSFuzzy
from utils.ts_fuzzy.actfunc import ActiveFunc
from utils.optimize_algorithms.particle_swarm_optimization import PSO
from utils.optimize_algorithms.particle_swarm_optimization import PSO
import sklearn.metrics as metrics

def NoTL(x_tgt,y_tgt):
    global M
    global C
    global Node
    global ParMax
    khold = Khold(x_tgt.T,y_tgt,10)
    i = 0
    result = ""
    for x_train,y_train,x_test,y_test in khold:
        tsfuzzy = TSFuzzy(x_train,y_train,None,None,M,C,Node,ParMax,ActiveFunc.sigmoid)
        y_pred = tsfuzzy.predict(x_test)
        input2d = to2D(x_test)
        scatter_2data_3d(input2d,y_test,y_pred,"imgs/housing_no_tl/"+str(i))
        result += "iter:"+str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+"R:"+str(metrics.r2_score(y_pred,y_test))+"\n"
        print("iter:",str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+"R:"+str(metrics.r2_score(y_pred,y_test)))
        i+=1
    WriteTextFile("imgs/housing_no_tl/result.txt",result)

def ReduceFeatureTL(x_tgt,y_tgt,x_src,y_src):
    global M
    global C
    global Node
    global ParMax
    global Size
    global TMax
    global W
    global CP
    global CG
    khold = Khold(x_tgt.T,tgt_y,10)
    i = 0
    result = ""
    for x_train,y_train,x_test,y_test in khold:
        pso = PSO(size=Size,
                tMax=TMax,
                w=W,
                cp=CP,
                cg=CG,
                parMax=ParMax,
                X_src=src_x,
                y_src=src_y,
                X_tgt=x_train,
                y_tgt=y_train,
                X_test=x_test,
                y_test=y_test,
                M=M,
                C=C,
                Node=Node,
                ActFunc=ActiveFunc.sigmoid)
        gBestValue,y_pred,trans,gBestPos = pso.learn()
        input2d = to2D(x_test)
        scatter_2data_3d(input2d,y_test,y_pred,"imgs/housing_reduce/"+str(i))
        result += "iter:"+str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+"R:"+str(metrics.r2_score(y_pred,y_test))+"\n"
        print("iter:",str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+"R:"+str(metrics.r2_score(y_pred,y_test)))
        i+=1
    WriteTextFile("imgs/housing_reduce/result.txt",result)
    
def MappingFeatureTL(x_tgt,y_tgt,x_src,y_src):
    global M
    global C
    global Node
    global ParMax
    global Size
    global TMax
    global W
    global CP
    global CG
    khold = Khold(x_tgt.T,tgt_y,10)
    i = 0
    result = ""
    for x_train,y_train,x_test,y_test in khold:
        pso = PSO(size=Size,
                tMax=TMax,
                w=W,
                cp=CP,
                cg=CG,
                parMax=ParMax,
                X_src=src_x,
                y_src=src_y,
                X_tgt=x_train,
                y_tgt=y_train,
                X_test=x_test,
                y_test=y_test,
                M=M,
                C=C,
                Node=Node,
                ActFunc=ActiveFunc.sigmoid)
        gBestValue,y_pred,trans,gBestPos = pso.learn()
        input2d = to2D(x_test)
        scatter_2data_3d(input2d,y_test,y_pred,"imgs/housing_mapping/"+str(i))
        result += "iter:"+str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+"R:"+str(metrics.r2_score(y_pred,y_test))+"\n"
        print("iter:",str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+"R:"+str(metrics.r2_score(y_pred,y_test)))
        i+=1
    WriteTextFile("imgs/housing_mapping/result.txt",result)

def AllFeatureTL(x_tgt,y_tgt,x_src,y_src):
    global M
    global C
    global Node
    global ParMax
    global Size
    global TMax
    global W
    global CP
    global CG
    khold = Khold(x_tgt.T,tgt_y,10)
    i = 0
    result = ""
    for x_train,y_train,x_test,y_test in khold:
        pso = PSO(size=Size,
                tMax=TMax,
                w=W,
                cp=CP,
                cg=CG,
                parMax=ParMax,
                X_src=src_x,
                y_src=src_y,
                X_tgt=x_train,
                y_tgt=y_train,
                X_test=x_test,
                y_test=y_test,
                M=M,
                C=C,
                Node=Node,
                ActFunc=ActiveFunc.sigmoid)
        gBestValue,y_pred,trans,gBestPos = pso.learn()
        input2d = to2D(x_test)
        scatter_2data_3d(input2d,y_test,y_pred,"imgs/housing_all/"+str(i))
        result += "iter:"+str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+" R:"+str(metrics.r2_score(y_pred,y_test))+"\n"
        print("iter:",str(i)+" mse:"+str(metrics.mean_squared_error(y_pred,y_test))+" R:"+str(metrics.r2_score(y_pred,y_test)))
        i+=1
    WriteTextFile("imgs/housing_all/result.txt",result)

def VisualizeData():
    dataset = housing_USA.split_dataset()
    src_y,src_x = dataset.getData("Seattle")
    tgt_y,tgt_x = dataset.getData("Bellevue")
    input = np.concatenate([src_x,tgt_x])
    input2d = to2D(input)
    print("input",input2d.T[0][:len(src_y)].shape)
    src_x1 = input2d.T[0][:len(src_y)]
    src_x2 = input2d.T[1][:len(src_y)]
    tgt_x1 = input2d.T[0][len(src_y):]
    tgt_x2 = input2d.T[1][len(src_y):]
    print(tgt_x1.shape,tgt_x2.shape,tgt_y.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.show()

def WriteTextFile(path,text):
    with open(path,mode='w') as f:
        f.write(text)

if __name__ == "__main__":
    M = 2.1
    C = 7
    Node = 10
    ParMax = 5
    Size = 300
    TMax = 1200
    W = 0.9
    CP = 0.8
    CG = 0.6

    dataset = housing_USA.split_dataset()
    # inputdata: [bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,yr_built]

    #case 1 src: no bedrooms, tgt: no bathrooms, mapping bathrooms -> bedrooms
    src_y,src_x = dataset.getData("Seattle")
    tgt_y,tgt_x = dataset.getData("Bellevue")
    src_x = src_x.T
    tgt_x = tgt_x.T
    c1_src_x = np.array([src_x[1],src_x[2],src_x[3],src_x[4],src_x[5],src_x[6],src_x[7]])
    c1_tgt_x = np.array([tgt_x[0],tgt_x[2],tgt_x[3],tgt_x[4],tgt_x[5],tgt_x[6],tgt_x[7]])

    reduced_src_x = np.array([src_x[2],src_x[3],src_x[4],src_x[5],src_x[6],src_x[7]])
    reduced_tgt_x = np.array([tgt_x[2],tgt_x[3],tgt_x[4],tgt_x[5],tgt_x[6],tgt_x[7]])
    
    #notl 
    #NoTL(c1_tgt_x,tgt_y)
    
    #redice
    ReduceFeatureTL(reduced_tgt_x,tgt_y,reduced_src_x,src_y)
    
    #mapping
    MappingFeatureTL(c1_tgt_x,tgt_y,c1_src_x,src_y)
    
    #all
    #AllFeatureTL(tgt_x,tgt_y,src_x,src_y)