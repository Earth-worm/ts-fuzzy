
from data import housing_USA
import numpy as np
from utils.k_fold_cross_validation import Khold

x = np.array([1,2,3,4,5,6,7,8])
y = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8]])

kholdDataSplit = Khold(x,y,4)
for x_train,y_train,x_test,y_test in kholdDataSplit:
    print("x_train:",x_train)
    print("y_train:",y_train)
    print("x_test:",x_test)
    print("y_test:",y_test)