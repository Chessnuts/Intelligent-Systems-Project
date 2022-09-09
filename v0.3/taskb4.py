
from unicodedata import bidirectional

import stockprediction as sp
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU

import datetime as dt

#get data
data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime.now(), saving=False, split_by_date=True)

#Create our different models for testing

base_lstm_model = sp.create_model(60, 5, n_layers=3)

rnn_model = sp.create_model(60, 5, cell=SimpleRNN, n_layers=3)

gru_model = sp.create_model(60, 5, cell=GRU, n_layers=3)


double_layer_model = sp.create_model(60, 5, n_layers=6)

double_layer_size_model = sp.create_model(60, 5, units=512, n_layers=3)

double_epoch_model = sp.create_model(60, 5, n_layers=3)

double_batch_size_model = sp.create_model(60, 5, n_layers=3)

double_everything_model = sp.create_model(60, 5, units=512, n_layers=6)


#train our models

base_lstm_model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=64)

rnn_model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=64)

gru_model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=64)



double_layer_model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=64)

double_layer_size_model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=64)

double_epoch_model.fit(data["X_train"], data["y_train"], epochs=100, batch_size=64)

double_batch_size_model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=128)

double_everything_model.fit(data["X_train"], data["y_train"], epochs=100, batch_size=128)


#test
base_lstm_model_pp = base_lstm_model.predict(data["X_test"])
base_lstm_model_pp = data["column_scaler"]["adjclose"].inverse_transform(base_lstm_model_pp)

rnn_model_pp = rnn_model.predict(data["X_test"])
rnn_model_pp = data["column_scaler"]["adjclose"].inverse_transform(rnn_model_pp)

gru_model_pp = gru_model.predict(data["X_test"])
gru_model_pp = data["column_scaler"]["adjclose"].inverse_transform(gru_model_pp)


double_layer_model_pp = double_layer_model.predict(data["X_test"])
double_layer_model_pp = data["column_scaler"]["adjclose"].inverse_transform(double_layer_model_pp)

double_layer_size_model_pp = double_layer_size_model.predict(data["X_test"])
double_layer_size_model_pp = data["column_scaler"]["adjclose"].inverse_transform(double_layer_size_model_pp)

double_epoch_model_pp = double_epoch_model.predict(data["X_test"])
double_epoch_model_pp = data["column_scaler"]["adjclose"].inverse_transform(double_epoch_model_pp)

double_batch_size_model_pp = double_batch_size_model.predict(data["X_test"])
double_batch_size_model_pp = data["column_scaler"]["adjclose"].inverse_transform(double_batch_size_model_pp)

double_everything_model_pp = double_everything_model.predict(data["X_test"])
double_everything_model_pp = data["column_scaler"]["adjclose"].inverse_transform(double_everything_model_pp)

#plot
plt.plot(data["test_df"]["adjclose"].values, color="black", label=f"Actual Price")

plt.plot(base_lstm_model_pp, color="green", label=f"base lstm")
plt.plot(rnn_model_pp, color="blue", label=f"rnn")
plt.plot(gru_model_pp, color="red", label=f"gru")

plt.plot(double_layer_model_pp, color="cyan", label=f"double layer")
plt.plot(double_layer_size_model_pp, color="magenta", label=f"double layer size")
plt.plot(double_epoch_model_pp, color="#800000", label=f"double epoch")
plt.plot(double_batch_size_model_pp, color="#000080", label=f"double batch size")
plt.plot(double_everything_model_pp, color="#800080", label=f"double everything")

plt.title(f"Share Price")
plt.xlabel("Time")
plt.ylabel(f"Share Price")
plt.legend()
plt.show()