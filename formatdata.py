''' This code formats the incoming dataset csvs into csvs that can easily be 
combined into a single master csv or the exhange rate csv called simply df can 
be used by itself so final combination is left for other code. '''

import numpy as np
import pandas as pd



def clean(file):
    df = pd.read_csv(file, index_col = 0)
    df['ds'] = df['<DTYYYYMMDD>'].astype(str)
    df['ts'] = df['<TIME>'].astype(str).str.zfill(6)
    df['dts'] = df['ds'] + df['ts']
    df['dt'] = pd.to_datetime(df['dts'], format = '%Y%m%d%H%M%S')
    df = df.drop(columns=['<DTYYYYMMDD>', '<TIME>', 'ds', 'ts', 'dts'])
    df = df.set_index('dt')
    return df


idf = pd.read_csv("Downloads/interestecb.csv")
idf2 = idf.reindex(index=idf.index[::-1]).copy()
idf2['Date'] = pd.to_datetime(idf2['Date'], format = '%Y-%m-%d')
idf2 = idf2.set_index(pd.DatetimeIndex(idf2['Date']))
idf2 = idf2.drop(columns = 'Date')
idf2 = idf2[idf2.index > pd.datetime(2000,11,30)]
resampledi = idf2.resample("d").pad()
resampledi = resampledi[resampledi.index > pd.datetime(2001,1,1)]

fedf = pd.read_csv("Downloads/FEDFUNDS.csv")
fedf['DATE'] = pd.to_datetime(fedf['DATE'], format = '%Y-%m-%d')
fedf = fedf.set_index(pd.DatetimeIndex(fedf['DATE']))
fedf = fedf.drop(columns = 'DATE')
fedf = fedf[fedf.index > pd.datetime(2000,11,30)]
resampledf = fedf.resample("d").pad()
resampledf = resampledf[resampledf.index > pd.datetime(2001,1,1)]
ind = pd.date_range("20201202", "20201231", freq="d")
series = pd.Series(.09, index = ind)
fedfadd = series.to_frame("FEDFUNDS")
resampledf = pd.concat([resampledf, fedfadd])
resampledf

file = "Downloads/EURUSD.txt"
df = clean(file)
resampled = df.resample('d').mean()
