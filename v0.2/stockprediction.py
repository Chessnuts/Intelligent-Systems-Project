###########################################################################################################
# Stock Prediction Model
#
# Created by Ryan Chessum (102564760) for Intelligent Systems COS30018 @ Swinburne University of Technology
#
###########################################################################################################
import tensorflow as tf
from functions import shuffle_in_unison
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import mplfinance as fplt
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime as dt
import random

import functions

#Load data off the web to be used in the model
def load_data(ticker, start_date, end_date, saving=True, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):

    #store data we want to rerurn in this variable
    results = {}

    #check the data type of ticker
    # if ticker is a string use yahoo to get the stock data between the start and end date
    if(isinstance(ticker, str)):
        data_frame = si.get_data(ticker, start_date, end_date)
    elif(isinstance(ticker, pd.DataFrame)): #else if data passed in is already a pandas dataframe data type, set the data frame to be the tick
        data_frame = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    results['data_frame'] = data_frame.copy() #make a copy of the data to store in the results variable

    #check if the feature columns from the parameters are present in the data frame
    for column in feature_columns:
        assert column in data_frame.columns, f"'{column}' does not exist in the dataframe."

    #add a date column 
    if "date" not in data_frame.columns:
        data_frame["date"] = data_frame.index 


    #if scale variable is true, scale down data for use (like in v0.01)
    if scale:
        #dictionary for storing scalers 
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        #loop to cycle through the columns
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler() #Assign an new scaler
            #scale the data in the dataframe
            data_frame[column] = scaler.fit_transform(np.expand_dims(data_frame[column].values, axis=1))
            #store the scaler in the ditionary
            column_scaler[column] = scaler 

        # add the MinMaxScaler instances to the result returned
        results["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    data_frame['future'] = data_frame['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(data_frame[feature_columns].tail(lookup_step))
    
    # drop NaNs
    data_frame.dropna(inplace=True) 
    #inplace true means that the the dropping is done on the same object instead of making and returning a new dataframe with the values dropped

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(data_frame[feature_columns + ["date"]].values, data_frame['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    results['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X) #last 60 days
    y = np.array(y) #future day

    #split up the data into train and test data based on the parameters
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X)) #int value for tranng the data based on the test size and length of X
        results["X_train"] = X[:train_samples] # training data
        results["y_train"] = y[:train_samples] # prediction training data
        results["X_test"]  = X[train_samples:] # testing data
        results["y_test"]  = y[train_samples:] # prediction testing data
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(results["X_train"], results["y_train"]) 
            shuffle_in_unison(results["X_test"], results["y_test"])   
    else:    
        # split the dataset randomly
        results["X_train"], results["X_test"], results["y_train"], results["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)

    # get the list of test set dates
    dates = results["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    results["test_df"] = results["data_frame"].loc[dates]
    # remove duplicated dates in the testing dataframe
    results["test_df"] = results["test_df"][~results["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    results["X_train"] = results["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    results["X_test"] = results["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    ticker_data_filename = "B3TickerData"
    scaler_data_filename = "B3ScalerData"

    if saving:
        #save the data
        results['data_frame'].to_csv(ticker_data_filename) #save dataframe in csv format
        if scale:
            results["column_scaler"].to_csv(scaler_data_filename) #save the scaler

    return results 

def candlestick_data(data, n=60):

    #make sure that n is not less than 1
    if n < 1:
        n = 1
    
    #refactors the data with a new frequency 
    plot_data = data.asfreq(str(n) + 'D') 
    #uses pandas dataframe function asfreq
    #returns a new dataframe with adjusted values to the specified new frequency
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asfreq.html
    #string formatting for n + days
    #we could also specify weeks or years if we wanted. To do this change 'D' to 'W' or 'Y'



    #Plot the graph
    fplt.plot(plot_data, type='candle', title='facebook stock price', ylabel='Price ($)',
            style='charles', volume=True, ylabel_lower='shares\nTraded',
            show_nontrading=False)
    #   data is the data loaded from the load_data function
    #   type is the type of graph, since we want a candle graph we specify candle
    #   title is the heading of our graph, Keeping it simple with facebook stock
    #   y label is the label on the Y axis of our graph
    #   style specifies a style preset, 'charles' seemed like a nice one as it represents the data as red or green depending on how the price changed
    #   volume adds a bar chart at the bottom of the graph which shows how much stock was traded on that day
    #   y label lowe adds another label at the bottom of the y axis, we can use this to label the trade volume
    #   show nontrading puts a gap on days when the market is closed
    
    return

def boxplot_data(data, n=60, rolling=60):
    
    #make sure that n is not less than 1
    if n < 1:
        n = 1

    #plot data
    plot_data = data['close'].to_frame()

    #add a column for the average
    #we can use the rolling and mean functions to get the moving average
    plot_data['moving_average'] = plot_data.rolling(rolling).mean() # in is our parameter of how many days we want to use for our moving average

    #because we need multiple days to find the moving average we end up with NaN values. 
    #so we need to drop them
    plot_data.dropna(inplace=True)

    #only show last n days
    plot_data = plot_data.iloc[len(plot_data['moving_average']) - n: , : ]

    #print(plot_data)

    #show plots as a window turned on
    plt.ion()
    #plot the moving window data
    plt.boxplot(plot_data['moving_average'])

    #input so I can see the plot before it dissapears
    input()

    return
