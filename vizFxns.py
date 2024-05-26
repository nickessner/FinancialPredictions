import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import mplfinance as mpf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
import pandas as pd
import statsmodels.api as sm
plt.style.use('seaborn')
import matplotlib.ticker as ticker

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
        fit1 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add').fit(use_boxcox=True)
        fcast1 = fit1.forecast(predict_date).rename('Additive')
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse1), 2)))
        
        fit2 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
        fcast2 = fit2.forecast(predict_date).rename('Additive+damped')
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive damped trend, additive seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse2), 2)))
        
        fit1.fittedvalues.plot(style='--', color='red')
        fcast1.plot(style='--', marker='o', color='red', legend=True)
        fit2.fittedvalues.plot(style='--', color='green')
        fcast2.plot(style='--', marker='o', color='green', legend=True)
    
    elif seasonal_type == 'multiplicative':  
        fit3 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul').fit(use_boxcox=True)
        fcast3 = fit3.forecast(predict_date).rename('Multiplicative')
        mse3 = ((fcast3 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of additive trend, multiplicative seasonal of '+ 
              'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse3), 2)))
        
        fit4 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
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

def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',period=int((len(y)/7)))
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

### plot for Rolling Statistic for testing Stationarity
def test_stationarity(timeseries, title):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean() 
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label= title)
    ax.plot(rolmean, label='rolling mean');
    ax.plot(rolstd, label='rolling std (x10)');
    ax.legend()

# Call this function after pick the right(p,d,q) for SARIMA based on AIC               
def sarima_eva(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
    # fit the model 
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])
    
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    
    # The dynamic=False argument ensures that we produce one-step ahead forecasts, 
    # meaning that forecasts at each point are generated using the full history up to that point.
    try:
        pred = results.get_prediction(start=pd.to_datetime(pred_date).tz_localize('America/New_York'), dynamic=False)
    except:
        pred = results.get_prediction(start=pred_date, dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))

    ax = y.plot(label='observed')
    y_forecasted.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    plt.legend()
    plt.show()

    # A better representation of our true predictive power can be obtained using dynamic forecasts. 
    # In this case, we only use information from the time series up to a certain point, 
    # and after that, forecasts are generated using values from previous forecasted time points.
    try:
        pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date).tz_localize('America/New_York'), dynamic=True, full_results=True)
    except:
        pred_dynamic = results.get_prediction(start=pred_date, dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))

    ax = y.plot(label='observed')
    y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax,figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')

    plt.legend()
    plt.show()
    
    return (results)


def forecast(model, predict_steps, y, ticker_nm):
    pred_uc = model.get_forecast(steps=predict_steps)
    pred_ci = pred_uc.conf_int()

    fig, ax = plt.subplots(figsize=(13, 4))
    
    # Plot observed data with specified color
    y.plot(ax=ax, label='Observed', color='#D08C60')
    
    # Plot forecasted data with specified color
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color='#797D62')

    # Fill between confidence intervals
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)

    # Customize plot appearance
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_title(f'{ticker_nm} 12 Month Forecast', loc='left', fontsize=20, fontweight='bold', color='#696969', pad=20)
    ax.xaxis.labelpad = 0  # Remove padding between label and plot
    ax.yaxis.labelpad = 0  # Remove padding between label and plot
    ax.yaxis.set_major_formatter('${:,.0f}'.format)

    # Remove background and gridlines
    ax.set_facecolor('none')
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Set color for axis labels and tick labels
    ax.xaxis.label.set_color('#696969')
    ax.yaxis.label.set_color('#696969')
    ax.tick_params(axis='x', colors='#696969')
    ax.tick_params(axis='y', colors='#696969')

    # Show legend
    ax.legend()

    # Specify the directory path
    directory = "C:/Financial Outputs/Overviews"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Specify the path and filename for the PNG image
    forecast_file = os.path.join(directory, "forecast_chart")

    plt.savefig(forecast_file)

    # Show plot
    plt.show()

    # Produce the forecasted tables 
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['Date', 'Predicted_Mean']
    pci = pred_ci.reset_index()
    pci.columns = ['Date', 'Lower Bound', 'Upper Bound']
    final_table = pm.join(pci.set_index('Date'), on='Date')

    return final_table
