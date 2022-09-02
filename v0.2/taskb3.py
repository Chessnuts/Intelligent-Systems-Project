
import stockprediction as sp

import datetime as dt

data = sp.load_data("META", dt.datetime(2012, 1, 1), dt.datetime(2020, 1, 1), False)

sp.candlestick_data(data['data_frame'], 60)

sp.boxplot_data(data['data_frame'])
