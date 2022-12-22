from data.auto_mpg import get_data
from utils.disp import scatter_data_3d,scatter_2data_3d
import numpy as np
from utils.optimize_algorithms.particle_swarm_optimization import PSO

Xsrc,ysrc,Xtgt,ytgt,Xsrc2d,Xtgt2d = get_data()

M = 1.9 #1.9
C = 3#3
Node = 3 #3
ParMax = 50 #50
Size = 1000 #100
T = 5000 #100
CG = 0.7 #0.7
CP = 0.7 #0.7
W = 1 #1
"""
model =PSO(size=Size,tMax=T,w=W,cp=CP,cg=CG,parMax=ParMax,X_src=Xsrc,y_src=ysrc,X_tgt=Xtgt,y_tgt=ytgt,M=M,C=C,Node=Node)
score,y_pred,trans = model.learn()
print(score)
scatter_2data_3d(Xtgt2d,y_pred,ytgt,"imgs/test1000",[20,30])
"""

model =PSO(size=Size,tMax=T,w=W,cp=CP,cg=CG,parMax=ParMax,X_src=Xsrc,y_src=ysrc,X_tgt=Xtgt,y_tgt=ytgt,M=M,C=C,Node=Node)
score,y_pred,trans = model.learn()
print(score)
scatter_2data_3d(Xtgt2d,y_pred,ytgt,"imgs/test_m1.9_c3_node3_parmax50_size1000_T5000",[20,30])