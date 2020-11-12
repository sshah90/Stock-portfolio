import yfinance as yf
import json

tsla = yf.Ticker("TSLA")

print (json.dumps(tsla.info, sort_keys=True, indent=4))

# print(tsla.info)

# get historical market data
# hist = tsla.history(period="ytd",interval = "1d",)

# print(hist.head())