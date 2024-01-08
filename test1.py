import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from data import housing_USA
import matplotlib.pyplot as plt

dataset = housing_USA.split_dataset()
# inputdata: [bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,yr_built]

src = dataset.getData("Seattle")
tgt = dataset.getData("Bellevue")

tgt_size = len(tgt['price'])
src_size = len(src['price'])

columns = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"]

result = np.zeros([len(columns),len(columns)])

tgt_col = "bedrooms"
src_col = "bedrooms"
sorted_tgt = tgt.sort_values(tgt_col)
sorted_src = src.sort_values(src_col)

pair_tgt = np.array([sorted_tgt[tgt_col],sorted_tgt['price']]).T
pair_src = np.array([sorted_src[src_col],sorted_src['price']]).T

selected_src = np.array([pair_src[int(src_size * i / tgt_size)] for i in range(tgt_size)])

mtx1,mtx2,disparity = procrustes(pair_tgt, selected_src)
plt.scatter(pair_tgt.T[0],pair_tgt.T[1])
plt.scatter(selected_src.T[0],selected_src.T[1])
plt.scatter(after.T[0],after.T[1])
plt.show()

"""
for i in range(len(columns)):
    src_col = columns[i]
    for j in range(len(columns)):
        tgt_col = columns[j]
        sorted_tgt = tgt.sort_values(tgt_col)
        sorted_src = src.sort_values(src_col)

        pair_tgt = np.array([sorted_tgt[tgt_col],sorted_tgt['price']]).T
        pair_src = np.array([sorted_src[src_col],sorted_src['price']]).T

        selected_src = np.array([pair_src[int(src_size * i / tgt_size)] for i in range(tgt_size)])

        _,_,disparity = procrustes(pair_tgt, selected_src)
        result[i][j] = disparity

resultDF = pd.DataFrame(result,columns = columns)
resultDF.to_excel("result/test.xlsx")
"""