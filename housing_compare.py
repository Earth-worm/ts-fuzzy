import pandas as pd
import numpy as np

columns = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","yr_built"]

result = np.zeros([len(columns),len(columns)])
dir = "./result/housing/after/"

for i in range(len(columns)):
    for j in range(len(columns)):
        if i == j:
            continue
        path = dir + columns[i] + "_to_" + columns[j] + "/notl.csv"
        df = pd.read_csv(path)
        avg = df[""]
        