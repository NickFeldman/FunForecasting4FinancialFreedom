''' Creates a baseline that predicts fifty periods out using the moving average.
In this case it is using days as the period, however, a different file could be 
read in that could use a different period. '''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

file = "df.csv"
df = pd.read_csv(file)
df = df.dropna()
def fit_moving_average_trend(series, window=200):
    return series.rolling(window).mean()
series = fit_moving_average_trend(df['<CLOSE>'])
real = df['<CLOSE>']

real = real[199:]
real = np.array(real)
series = series[199:]
series = series[:-50]
series = np.array(series)
predictions = []
for i in range(len(series)):
    arr = np.full(50, series[i])
    predictions.append(arr)

reals = []
for i in range(len(real)-50):
    arr = real[i:i+50]
    reals.append(arr)

testScore = math.sqrt(mean_squared_error(reals, predictions))
print('Baseline Score: %.5f RMSE' % (testScore))
#