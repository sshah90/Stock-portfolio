import pandas_datareader.data as reader
import datetime as dt
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

end = dt.datetime.now()
start = dt.datetime(end.year - 20,end.month,end.day)
start
df = reader.get_data_yahoo('GOOG',start,end)
df

from fbprophet import Prophet

m = Prophet()
m
df = df.reset_index()
df
df[['ds','y']] = df[['Date','Adj Close']]
df

m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
m.plot(forecast)
plt.show()
