from data import housing_USA
from utils.disp import scatter_data_3d,scatter_2data_3d
import matplotlib.pyplot as plt
import numpy as np
from utils.k_fold_cross_validation import Khold
from utils.data_analize import correlation

dataset = housing_USA.split_dataset()
y_src,x_src = dataset.getData("Seattle")
y_tgt,x_tgt = dataset.getData("Auburn")
print(x_src.shape)
print(x_tgt.shape)
print(dataset.getInfo("Seattle"))
print(dataset.getInfo("Auburn"))


print("ソース")
for x in x_src.T:
    print(correlation(y_src,x))

print("ターゲット")
for x in x_tgt.T:
    print(correlation(y_tgt,x))