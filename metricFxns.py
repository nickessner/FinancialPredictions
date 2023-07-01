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



