
from unittest import result
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, SimpleRNN, GRU
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import mplfinance as fplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima.utils import ndiffs
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import datetime as dt
import random
import functions as sp


n_steps = 60
lookup_step = 10
epochs = 10
batch_size = 256

#load data
data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='close')
last_sequence = data["last_sequence"][-n_steps:]
# expand dimension
last_sequence = np.expand_dims(last_sequence, axis=0)

#multistep datasets
adjclose_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='adjclose')
volume_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='volume')
open_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='open')
high_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='high')
low_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='low')


#create models
#base lstm model
lstm_model = sp.create_model(n_steps, 6, n_layers=3)

#gru model
gru_model = sp.create_model(n_steps, 6, n_layers=3, cell=GRU)

#rnn model
rnn_model = sp.create_model(n_steps, 6, n_layers=3, cell=SimpleRNN)

#arima model
arima_model = sp.create_arima_model(data)

#multistep lookup models
adjclose_model = sp.create_model(n_steps, 6, n_layers=3)
close_model = sp.create_model(n_steps, 6, n_layers=3)
volume_model = sp.create_model(n_steps, 6, n_layers=3)
open_model = sp.create_model(n_steps, 6, n_layers=3)
high_model = sp.create_model(n_steps, 6, n_layers=3)
low_model = sp.create_model(n_steps, 6, n_layers=3)



#fit models
#fit lstm model
lstm_model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size)
#fit gru model
gru_model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size)
#fit rnn model
rnn_model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size)

#fit multistep models
close_model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size)
volume_model.fit(volume_data["X_train"], volume_data["y_train"], epochs=epochs, batch_size=batch_size)
open_model.fit(open_data["X_train"], open_data["y_train"], epochs=epochs, batch_size=batch_size)
high_model.fit(high_data["X_train"], high_data["y_train"], epochs=epochs, batch_size=batch_size)
low_model.fit(low_data["X_train"], low_data["y_train"], epochs=epochs, batch_size=batch_size)

#make predictions predictions
#store predictions 
predictions = []

#lstm model prediction
lstm_prediction = lstm_model.predict(last_sequence)
lstm_prediction = data["column_scaler"]["close"].inverse_transform(lstm_prediction)

#gru model predictions
gru_prediction = gru_model.predict(last_sequence)
gru_prediction = data["column_scaler"]["close"].inverse_transform(gru_prediction)

#rnnmodel predictions
rnn_prediction = rnn_model.predict(last_sequence)
rnn_prediction = data["column_scaler"]["close"].inverse_transform(rnn_prediction)

#arima model prediction
arima_prediction = arima_model.forecast()


#add predictions to  list for ensemble prediction
predictions.append(lstm_prediction[0][0])
predictions.append(gru_prediction[0][0])
predictions.append(rnn_prediction[0][0])
predictions.append(arima_prediction)

print("lstm prediction: " + str(lstm_prediction[0][0]))
print("gru prediction: " + str(gru_prediction[0][0]))
print("rnn prediction: " + str(rnn_prediction[0][0]))
print("arima prediction: " + str(arima_prediction))

#make multistep predictions

predictions = sp.multistep_prediction(adjclose_model, close_model, volume_model, open_model, high_model, low_model, adjclose_data, n_steps, k=10)

print('Multistep prediction:')
for x in predictions:
    print(x)
    
#make ensemble prediction
print("ensemble prediction:")
result = sp.ensemble_prediction(predictions)

print(result)