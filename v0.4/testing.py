from unicodedata import bidirectional

import numpy as np
import functions as sp
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU

import datetime as dt

n_steps = 60
lookup_step = 10
epochs = 1
batch_size = 256

adjclose_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='adjclose')
adjclose_model = sp.create_model(n_steps, 6, n_layers=3)
adjclose_model.fit(adjclose_data["X_train"], adjclose_data["y_train"], epochs=epochs, batch_size=batch_size)

close_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='close')
close_model = sp.create_model(n_steps, 6, n_layers=3)
close_model.fit(close_data["X_train"], close_data["y_train"], epochs=epochs, batch_size=batch_size)

volume_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='volume')
volume_model = sp.create_model(n_steps, 6, n_layers=3)
volume_model.fit(volume_data["X_train"], volume_data["y_train"], epochs=epochs, batch_size=batch_size)

open_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='open')
open_model = sp.create_model(n_steps, 6, n_layers=3)
open_model.fit(open_data["X_train"], open_data["y_train"], epochs=epochs, batch_size=batch_size)

high_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='high')
high_model = sp.create_model(n_steps, 6, n_layers=3)
high_model.fit(high_data["X_train"], high_data["y_train"], epochs=epochs, batch_size=batch_size)

low_data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True, lookup_step=lookup_step, n_steps=n_steps, t='low')
low_model = sp.create_model(n_steps, 6, n_layers=3)
low_model.fit(low_data["X_train"], low_data["y_train"], epochs=epochs, batch_size=batch_size)



predictions = sp.multistep_prediction(adjclose_model, close_model, volume_model, open_model, high_model, low_model, adjclose_data, n_steps, k=10)

for x in predictions:
    print(x)

    