import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import mplfinance as mpf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
import pandas as pd

def plot_candlestick(df, symbol, rng, start, end):
    mpf.plot(
        df[start:end],
        type="candle",
        title=f"{symbol} Price, {rng}",
        ylabel="Price ($)",
        volume=True,
        ylabel_lower="Volume",
        show_nontrading=False,
        mav=(4),
        figsize=(8, 3),
        style="yahoo",
        datetime_format="%Y-%m-%d")

def plot_crossover(ticker, df):
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df[["Close", "SMA_50", "SMA_200"]].plot(
        title=f"{ticker}: SMA Crossover",
        style=["-", "-", "--"])
    return df


def plot_bollinger(ticker, df):
    df[["Close", "SMA_20", "UpperBB", "LowerBB"]].plot(
        title=f"{ticker} with Bollinger Bands",
        style=["-", "--", "-", "-"])

def plot_rsi(df, symbol, periods=14):
    fig, ax = plt.subplots(
        nrows=2, ncols=1,
        sharex=True, tight_layout=True,
        figsize=(9, 6))
    ax[0].set_title(f"{symbol} price")
    ax[0].plot(df["Close"])
    ax[1].set_title(f"RSI ({periods}-day moving average)")
    ax[1].set_ylim(0, 100)
    ax[1].plot(df["RSI"])
    ax[1].axhline(70, color="r", ls="--")
    ax[1].axhline(30, color="g", ls="--")
    custom_lines = [
        Line2D([0], [0], color="r", lw=4),
        Line2D([0], [0], color="g", lw=4)
    ]
    ax[1].legend(
        custom_lines, ["Overbought", "Oversold"], loc="best")

def plot_macd(df, symbol, periods=14):
    fig, ax = plt.subplots(
        nrows=2, ncols=1,
        sharex=True, tight_layout=True,
        figsize=(9, 6))
    ax[0].set_title(f"{symbol} price")
    ax[0].plot(df["Close"])
    ax[1].set_title(f"MACD")
    ax[1].plot(df["MACD"], color="gray") # slow signal
    ax[1].plot(df["MACD-s"], color="orange") # fast signal
    ax[1].bar(df.index, height=df["MACD-h"], color="black")
    custom_lines = [
        Line2D([0], [0], color="gray", lw=4),
        Line2D([0], [0], color="orange", lw=4),
        Line2D([0], [0], color="black", lw=4)
    ]
    ax[1].legend(
        custom_lines, ["MACD", "Signal", "Diff"], loc="best")

def holt_win_sea(y,y_to_train,y_to_test,seasonal_type,seasonal_period,predict_date):
    
    y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    if seasonal_type == 'additive':
        fit1 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add', use_boxcox=True).fit()
        fcast1 = fit1.forecast(predict_date).rename('Additive')
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse1), 2)))
        
        fit2 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add', damped=True, use_boxcox=True).fit()
        fcast2 = fit2.forecast(predict_date).rename('Additive+damped')
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive damped trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse2), 2)))
        
        fit1.fittedvalues.plot(style='--', color='red')
        fcast1.plot(style='--', marker='o', color='red', legend=True)
        fit2.fittedvalues.plot(style='--', color='green')
        fcast2.plot(style='--', marker='o', color='green', legend=True)
    
    elif seasonal_type == 'multiplicative':  
        fit3 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul', use_boxcox=True).fit()
        fcast3 = fit3.forecast(predict_date).rename('Multiplicative')
        mse3 = ((fcast3 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, multiplicative seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse3), 2)))
        
        fit4 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul', damped=True, use_boxcox=True).fit()
        fcast4 = fit4.forecast(predict_date).rename('Multiplicative+damped')
        mse4 = ((fcast3 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive damped trend, multiplicative seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse4), 2)))
        
        fit3.fittedvalues.plot(style='--', color='red')
        fcast3.plot(style='--', marker='o', color='red', legend=True)
        fit4.fittedvalues.plot(style='--', color='green')
        fcast4.plot(style='--', marker='o', color='green', legend=True)
        
    else:
        print('Wrong Seasonal Type. Please choose between additive and multiplicative')

    plt.show()

def yahoo_plot(df):
    df = df["2022-06-03":"2023-06-02"]
    mpf.plot(df, mav=(50, 200), style="yahoo")

