
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


data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='close')
last_sequence = data["last_sequence"][-n_steps:]
# expand dimension
last_sequence = np.expand_dims(last_sequence, axis=0)

lstm_model = sp.create_model(n_steps, 6, n_layers=3)

gru_model = sp.create_model(n_steps, 6, n_layers=3, cell=GRU)

rnn_model = sp.create_model(n_steps, 6, n_layers=3, cell=SimpleRNN)


arima_model = sp.create_arima_model(data)

lstm_model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size)
gru_model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size)
rnn_model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size)

predictions = []

lstm_prediction = lstm_model.predict(last_sequence)
lstm_prediction = data["column_scaler"]["close"].inverse_transform(lstm_prediction)

gru_prediction = gru_model.predict(last_sequence)
gru_prediction = data["column_scaler"]["close"].inverse_transform(gru_prediction)

rnn_prediction = rnn_model.predict(last_sequence)
rnn_prediction = data["column_scaler"]["close"].inverse_transform(rnn_prediction)

arima_prediction = arima_model.forecast()



predictions.append(lstm_prediction[0][0])
predictions.append(gru_prediction[0][0])
predictions.append(rnn_prediction[0][0])
predictions.append(arima_prediction)

result = sp.ensemble_prediction(predictions)

print(result)


    