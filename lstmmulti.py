import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def train_test_split(X, y, prct=.2):
    '''Splits the time series data into a train and test set
    Inputs: 
    X: The data to that used to make prediction
    y: The predicted variable
    prct:  Optional percentage of data to be used for testing
    Outputs: Four arrays, series or Dataframe contain X and y training and
    testing datasets.'''
    X_train, X_test = X[:int((1-prct)*len(X))], X[int((1-prct)*len(X)):]
    y_train, y_test = y[:int((1-prct)*len(y))], y[int((1-prct)*len(y)):]
    
    return X_train, X_test, y_train, y_test

def formatdata(past,pred, back=1):
    ''' Format the past data and prediction data in arrays that can be fed into LSTM 
    Inputs:
    Past: A dataframe or series containing the data used to make a prediction
    pred: A series that contains the predicted variable
    back: How much historical data to consider with each prediction, optional 
    defaults to just 1.
    Outputs: Numpy array of historical data and numpy array predicted variable
    '''
    pastX, predY = [], []
    for i in range(len(past)-back-1 -50): 
        pastX.append(past[i:(i+back), None])
        predY.append(pred[i + back:i + back + 50])
    return np.array(pastX), np.array(predY)

def chooseoptions(Xtrain, Xtest, ytrain, ytest, back):
    ''' Allows to choose options currently just the amount of past data to include
    Inputs:
    Xtrain: Training data
    Xtest: Testing data
    ytrain: Predicted variable for training
    ytest: Predicted variable for testing
    back: amount of previous periods to include in prediction
    Outputs: formated numpy arrays of training and testing data '''
    trainpast, trainpred = formatdata(Xtrain, ytrain, back)
    testpast,testpred = formatdata(Xtest, ytest, back)
    return trainpast, trainpred, testpast, testpred
    
def predict(model, trainpastr, testpastr):
    ''' Creates the prediction arrays
    Inputs:
    model:  The built model
    trainpastr: The training array 
    testpastr: thr testing array.
    Outputs:
    An array of predictions on the training data
    An array of predictions on the testing data
    
    By comparing test results on training and testing data it is possible to see
    if model is overfit''' 
    trainPredict = model.predict(trainpastr)
    testPredict = model.predict(testpastr)
    return trainPredict, testPredict

def buildmodel(pastr, pred, epoch = 10, bs = 256):
    ''' Builds model with inputs given
    Inputs:
    pastr: the data available for predictions
    pred: the predicted variables to train model on:
    epoch: Optional defaults to 10
    bs: Optional batch size defaults to 256
    Returns the fitted model'''
    model = Sequential()
    model.add(LSTM(100))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(25))
    model.add(Dense(5))
    model.add(Dense(25))
    model.add(Dense(50))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(pastr, pred, epochs=epoch, batch_size=bs)
    
    return model

def metrics(trainpred, trainPredict, testpred, testPredict):
    ''' Prints and returns the metric data which consists of RMSE
    Inputs:
    trainpred: actual predicted variable from training data set
    trainPredict: Predictions made on training data by the model.
    testpred: actual predicted variable from testting data set.
    testPredict: Predictions made on testing data by the model.
    Outputs:
    RMSE for training predictions and
    RMSE for testing predictions
    
    By comparing these numbers we can determine possible overfitting'''
    trainScore = math.sqrt(mean_squared_error(trainpred, trainPredict))
    print('Train Score: %.5f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testpred, testPredict))
    print('Test Score: %.5f RMSE' % (testScore))
    return trainScore, testScore    

#Code to take in file, create the appropriate inputs for the model, create model and results

file = "eurousddailywithrates.csv"
back = 250
df = pd.read_csv(file)
df = df.set_index(pd.DatetimeIndex(df['dt']))
df = df.drop(columns='dt')
df = df.dropna()
X = df.values
y = df["<CLOSE>"].values
X_train, X_test, y_train, y_test = train_test_split(X,y)
trpt, trpd, tstpt, tstpd = chooseoptions(X_train, X_test, y_train, y_test, back)
tstr = tstpt.reshape((-1,back,7))
trptr = trpt.reshape((-1,back,7))
model = buildmodel(trptr,trpd)
r1, r2 = predict(model, trptr, tstr)
model.summary()
metrics(trpd,r1,tstpd,r2)