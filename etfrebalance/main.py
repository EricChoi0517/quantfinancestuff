from collections import namedtuple

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["SPY"]
tickerData = yf.download(tickers, start="2000-01-01")["Close"]

tickerData.plot()
plt.show()

returns = tickerData.pct_change().dropna()
#print(returns.head())

PortfolioWeight = [1.0]

weeklyData = tickerData.resample("W").last()
WeeklyPct = weeklyData.pct_change().dropna()

portfolioReturn = (WeeklyPct @ PortfolioWeight) #might need it eventually if have multiple tickers
  
