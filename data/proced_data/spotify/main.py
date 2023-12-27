# https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

filename = "spotify_data.csv"

def get_data(filename):
    missing_values = ["n/a", "na", "--", " ", "N/A", "NA"]
    return pd.read_csv(filename,na_values = missing_values,sep=",")
'''
originalDF = get_data(filename)
df = originalDF.copy()
'''

originalParty = get_data("genre/party.csv")
party = originalParty.copy()

originalPunk = get_data("genre/punk.csv")
punk = originalPunk.copy()

print("party:",party.info())
print("punk:",len(punk))