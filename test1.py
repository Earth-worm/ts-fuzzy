import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from data import housing_USA
import matplotlib.pyplot as plt

dataset = housing_USA.split_dataset()
# inputdata: [bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement,yr_built]

tgt = dataset.getData("Bellevue")
tgt_size = len(tgt['price'])
columns = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"]

result = np.zeros([len(columns),len(columns)])
for i in range(len(columns)):
    for j in range(len(columns)):
        if i == j: continue
        pair1 = np.array([tgt[columns[i]],tgt['price']]).T
        pair2 = np.array([tgt[columns[j]],tgt['price']]).T
        _,_,disparity = procrustes(pair1, pair2)
        result[i][j] = disparity

print(result)
resultDF = pd.DataFrame(result)
resultDF.to_excel("abcde.xlsx")