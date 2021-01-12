''' Combine the data from the csvs created in formatdata into one csv that 
contains all data and drops any nans ''' 



import numpy as np
import pandas as pd
import math



df = pd.read_csv("df.csv")
idf = pd.read_csv("idf.csv")
fedf = pd.read_csv("fedf.csv")
df = df.set_index(pd.DatetimeIndex(df['dt']))
df = df.drop(columns = 'dt')
fedi = fedf["FEDFUNDS"].values
ecbi = idf['Value'].values
df['ecbrate'] = ecbi
df['fedrate'] = fedi
df = df.dropna()
df.to_csv("eurousddailywithrates.csv")