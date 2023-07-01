comb_df = pd.DataFrame({
    "^IXIC": ixic.history(start=start, end=end,
             auto_adjust=False, back_adjust=False)["Close"],
    "AAPL": aapl.history(start=start, end=end,
            auto_adjust=False, back_adjust=False)["Close"],
    "AMZN": amzn.history(start=start, end=end,
            auto_adjust=False, back_adjust=False)["Close"],
    "GOOG": goog.history(start=start, end=end,
            auto_adjust=False, back_adjust=False)["Close"],
    "MSFT": msft.history(start=start, end=end,
            auto_adjust=False, back_adjust=False)["Close"],
    "TSLA": tsla.history(start=start, end=end,
            auto_adjust=False, back_adjust=False)["Close"],
})

#Normalize prices
norm_df = comb_df.div(comb_df.iloc[0])
norm_df.head()

norm_df.plot(title="2021 Q4 prices, selected NASDAQ stocks")