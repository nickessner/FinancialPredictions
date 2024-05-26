# Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

def add_bollinger_bands(df):
    sma_20 = df["Close"].rolling(20)
    mean = sma_20.mean()
    two_sigmas = 2 * sma_20.std()
    df["SMA_20"] = mean
    df["UpperBB"] = mean + two_sigmas
    df["LowerBB"] = mean - two_sigmas

def add_rsi(df, periods=14):
    close_diff = df["Close"].diff()
    up = close_diff.clip(lower=0)
    down = -1 * close_diff.clip(upper=0)
    ma_up = up.ewm(
        com=periods-1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(
        com=periods-1, adjust=True, min_periods=periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    df["RSI"] = rsi

def add_macd(df):
    ema_12 = df["Close"].ewm(
        span=12, adjust=False, min_periods=12).mean()
    ema_26 = df["Close"].ewm(
        span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(
        span=9, adjust=False, min_periods=9).mean()
    diff = macd - signal
    
    df["MACD"] = macd
    df["MACD-s"] = signal
    df["MACD-h"] = diff


def ADF_test(timeseries, dataDesc):
    print(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))

import itertools

def sarima_grid_search(y,seasonal_period):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
    
    mini = float('+inf')
    
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                
                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal

#                 print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))





