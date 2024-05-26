from polygon import RESTClient
import getpass
from datetime import date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
 
# Returns the current local date
today = date.today()
dateOfMonth = today.day
monthNumber = today.month
yearNumber = today.year

import ftplib
import io
import pandas as pd
import pandas_ta as ta
import requests
import requests_html
import numpy as np

# from pycaret.classification import *
# from pycaret.regression import *

import yfinance as yf
from yahoo_fin.stock_info import get_data, get_top_crypto, get_analysts_info
import yahoo_fin.stock_info as si
import yahoo_fin.options as ops
from yahoo_fin.stock_info import *

# import tensorflow as tf
import altair as alt

import mplfinance as mpf

import requests_cache

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

import metricFxns
from metricFxns import *
import vizFxns
from vizFxns import *

from sklearn import metrics
from sklearn import datasets

from yahooquery import Ticker

import ssl
import certifi
import urllib

from yahoo_fin import stock_info as si

otherTickers = si.tickers_other(include_company_data = True)
nasdaqTickers = si.tickers_nasdaq(include_company_data = True)
allStocks = pd.concat([otherTickers,nasdaqTickers])
stockMarketTickers = list(allStocks['ACT Symbol'].unique())
nasdaqTicks = list(nasdaqTickers['Symbol'].unique())

currentDate = str(monthNumber) + "/" + str(dateOfMonth) + "/" + str(yearNumber)
yearPrior = str(monthNumber) + "/" + str(dateOfMonth+1) + "/" + str(yearNumber-1)
# tickerLst = ['NVDA','ADBE','UBER','MSFT','AMZN']
tickerLst = ['UBER']

def sarimaForecast (tickerLst):

    print('History data collection @' + str(datetime.now()))
    historyDfs = []   
    for ticker in tickerLst:
        try: 
            yahooTickerData = yf.Ticker(ticker) #, session=session
            yahooHistory = yahooTickerData.history(period="5y", auto_adjust=False, back_adjust=False) #"max"
            yahooHistory['Ticker']=ticker
            historyDfs.append(yahooHistory)
        except:
            print(ticker)
            pass
    historyDfFull = pd.concat(historyDfs)

    print('RSI calculation @' + str(datetime.now()))
    rsiDfs = []
    for ticker in tickerLst:#current_portfolio:
        histDf = historyDfFull[historyDfFull['Ticker']==ticker]
        add_rsi(histDf)
        add_macd(histDf)
        rsiDfs.append(histDf)
    rsiDfFull = pd.concat(rsiDfs)

    endWeekDate = str(yearNumber)+"-"+str(monthNumber)+"-"+str(dateOfMonth)
    startWeekDate = str(yearNumber-5)+"-"+str(monthNumber)+"-"+str(dateOfMonth+1)
    rsiDfFull = rsiDfFull.dropna(subset=['RSI'])

    print('Resampling @' + str(datetime.now()))
    resampleDfs = []
    for ticker in tickerLst:#current_portfolio:
        resampleDf = rsiDfFull[rsiDfFull['Ticker']==ticker]
        
        resampleDfWeekly = resampleDf[startWeekDate:endWeekDate].resample('W').mean()
        resampleDfWeekly.index.freq = pd.infer_freq(resampleDfWeekly.index)
        resampleDfWeekly['Ticker'] = ticker
        resampleDfs.append(resampleDfWeekly)

    resampleRsiDfFull = pd.concat(resampleDfs)

    for ticker in tickerLst:
        y = resampleRsiDfFull[resampleRsiDfFull['Ticker']==ticker]['Close']
        y.index.freq = pd.infer_freq(resampleDfWeekly.index)
        print(y.index.freq)
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
        ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
        ax.set_ylabel('Close')
        ax.legend()

        print('Seasonal decomposition @' + str(datetime.now()))
        seasonal_decompose (y)
        pd.options.display.float_format = '{:.8f}'.format
        print('Stationarity testing @' + str(datetime.now()))
        test_stationarity(y,'raw data')

        # Detrending
        y_detrend =  (y - y.rolling(window=12).mean())/y.rolling(window=12).std()

        test_stationarity(y_detrend,'de-trended data')
        ADF_test(y_detrend,'de-trended data')

        # Detrending + Differencing

        y_12lag_detrend =  y_detrend - y_detrend.shift(12)

        test_stationarity(y_12lag_detrend,'12 lag differenced de-trended data')
        ADF_test(y_12lag_detrend,'12 lag differenced de-trended data')

        current_date = datetime.datetime.strptime(currentDate, "%m/%d/%Y")

        # Subtract 4 months from the current date
        endTrainingDate = current_date - relativedelta(months=4)
        # Add 4 days to the current date
        endTrainingDate += timedelta(days=4)

        # Set the start of the test week to be one week after the end training date
        startTestWeekDate = endTrainingDate + timedelta(weeks=1)

        # Generate formatted date strings
        endTrainingWeekDateStr = endTrainingDate.strftime("%Y-%m-%d")
        startTestWeekNearestWeekStr = startTestWeekDate.strftime("%Y-%m-%d")
        startTestWeekDateStr = startTestWeekDate.strftime("%Y-%m-%d")

        # Find the closest dates in y.index
        endTrainingDate = y.index[y.index.get_loc(endTrainingDate, method='nearest')]
        startTestWeekDate = y.index[y.index.get_loc(startTestWeekDate, method='nearest')]

        # Slicing the series using the Timestamp objects
        y_to_train = y[:endTrainingDate]  # dataset to train
        y_to_val = y[startTestWeekDate:]  # last X months for test

        # Calculate the number of data points for the test set
        predict_date = len(y) - len(y[:startTestWeekDate])

        print('Holt win sea @' + str(datetime.now()))
        vizFxns.holt_win_sea(y, y_to_train,y_to_val,'additive',52, predict_date)
        print('Sarima grid search @' + str(datetime.now()))
        metricFxns.sarima_grid_search(y,52)

        startTestWeekNearestWeekReal = y_to_val.index[0]
        print('Sarima eva @' + str(datetime.now()))
        model = vizFxns.sarima_eva(y,(1, 1, 1),(1, 1, 0, 52),52,startTestWeekNearestWeekReal,y_to_val)

        print('Forecast @' + str(datetime.now()))
        final_table = forecast(model,52,y,ticker)

        return final_table

sarimaForecast (tickerLst)