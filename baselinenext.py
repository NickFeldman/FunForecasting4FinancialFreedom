''' Produces a baseline predicting only the next period based on moving average '''


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

file = ""
df = pd.read_csv(file)
df = df.dropna()
def fit_moving_average_trend(series, window=200):
    return series.rolling(window).mean()
predictions = fit_moving_average_trend(df['<CLOSE>'])
real = df['<CLOSE>']
predictions = predictions[200:]
real = real[200:]
testScore = math.sqrt(mean_squared_error(real, predictions))
print('Baseline Score: %.5f RMSE' % (testScore))