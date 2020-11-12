import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import datetime
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

"""
Person is able to put in different weights on the stocks selected and it spits out the
annulaized mean return and volatility of that portfolio
"""
portfolio_df = pd.read_excel('Sample stocks acquisition dates_costs.xlsx')
list_stock = portfolio_df['Ticker'].values.tolist()
stocks = sorted(list_stock)

# set array holding portfolio weights of each stock
weights = np.asarray([0.5,0.1,0.15,0.1,0.05,0.02,0.02,0.06])

#download daily price data for each of the stocks in the portfolio
data = web.DataReader(stocks,data_source='yahoo',start='2013-01-01')['Adj Close']

data.sort_index(inplace=True)

#convert daily stock prices into daily returns
returns = data.pct_change()

#calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# calculate annualised portfolio return
portfolio_return = round(np.sum(mean_daily_returns * weights) * 252,2)
# calculate annualised portfolio volatility
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)

print ('The annualised mean return of stock is', portfolio_return, 
       'and the annualised volatility is', portfolio_std_dev) 

"""
Different weights are given to the stocks in the portfolio to give the person different combinations 
for voltility and mean return. This would show a graph with all the variations. It would also
spit out the weight combination that had the highest mean return and weight combination that 
has the lowest volatility.
"""

portfolio_df = pd.read_excel('Sample stocks acquisition dates_costs.xlsx')
list_stock = portfolio_df['Ticker'].values.tolist()
stocks = sorted(list_stock)

#download daily price data for each of the stocks in the portfolio
data = web.DataReader(stocks,data_source='yahoo',start='2013-01-01')['Adj Close']

def monte(data):
    data.sort_index(inplace=True)
    
    #convert daily stock prices into daily returns
    returns = data.pct_change()

    #calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    #set number of runs of random portfolio weights
    num_portfolios = 25000

    #set up array to hold results
    #results = np.zeros((3,num_portfolios))
    results = np.zeros((4+len(stocks)-1,num_portfolios))


    for i in range(num_portfolios):
        #select random weights for portfolio holdings
        weights = np.random.random(len(stocks))
        #rebalance weights to sum to 1
        weights /= np.sum(weights)
        
        #calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # store results in results array
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2,i] = results[0,i] / results[1,i]
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j+3,i] = weights[j]
        

    variable = ['ret','stdev','sharpe']
    variable.extend(stocks)
    results_frame = pd.DataFrame(results.T,columns=variable)
    return results_frame


results_frame = monte(data)

# locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

#create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()

#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)
#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)


print('Summary of portfolio and weight of stocks where sharpe ratio is the highest is: ',"\n", max_sharpe_port) # print portfolio percentage with the highest sharpe value
print('Summary of portfolio and weight of stocks that has the low volatility is: ',"\n", min_vol_port) # print portfolio percentage with the lowest volatility




"""
Measuring portfolio performance compared to s&P500

"""

# Import the Sample worksheet with acquisition dates and initial cost basis:
portfolio_df = pd.read_excel('Sample stocks acquisition dates_costs.xlsx')


# Date Ranges for SP 500 and for all tickers
# Modify these date ranges each week.
# The below will pull back stock prices from the start date until end date specified.
start_sp = datetime.datetime(2013, 1, 1)
end_sp = datetime.datetime(2020, 9, 30)

# This variable is used for YTD performance.
end_of_last_year = datetime.datetime(2019, 12, 31)

# These are separate if for some reason want different date range than SP.
stocks_start = datetime.datetime(2013, 1, 1)
stocks_end = datetime.datetime(2020, 9, 30)




#getting sp500 data
sp500 = pdr.get_data_yahoo('^GSPC', start_sp, end_sp)
    
# Adj Close for the EOY in 2017 in order to run comparisons versus stocks YTD performances.
sp_500_adj_close = sp500[['Adj Close']].reset_index()

# Adj Close for the EOY in 2017 in order to run comparisons versus stocks YTD performances.
sp_500_adj_close_start = sp_500_adj_close[sp_500_adj_close['Date']==end_of_last_year]




# Generate a dynamic list of tickers to pull from Yahoo Finance API based on the imported file with tickers.
tickers = portfolio_df['Ticker'].unique()

# Stock comparison code
def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
    datas = map(data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
            
all_data = get(tickers, stocks_start, stocks_end)

 

# Also only pulling the ticker, date and adj. close columns for our tickers.
adj_close = all_data[['Adj Close']].reset_index()

# Grabbing the ticker close from the end of last year
adj_close_start = adj_close[adj_close['Date']==end_of_last_year]

# Grab the latest stock close price
#adj_close_latest = adj_close[adj_close['Date']==stocks_end] 
adj_close_latest = adj_close[adj_close['Date']==max(adj_close['Date'])]




def merged_portfolio_sp(adj_close_latest, portfolio_df, sp_500_adj_close):
    adj_close_latest.set_index(['Ticker'], inplace=True) # removing the indexes
    portfolio_df.set_index(['Ticker'], inplace=True)
    # Merge the portfolio dataframe with the adj close dataframe; they are being joined by their indexes.
    merged_portfolio = pd.merge(portfolio_df, adj_close_latest, left_index=True, right_index=True)
    
    # The below creates a new column which is the ticker return; takes the latest adjusted close for each position
    # and divides that by the initial share cost.
    merged_portfolio['ticker return'] = merged_portfolio['Adj Close'] / merged_portfolio['Unit Cost'] - 1
    merged_portfolio.reset_index(inplace=True)
    
    
    # Here we are merging the new dataframe with the sp500 adjusted closes since the sp start price based on 
    # each ticker's acquisition date and sp500 close date.
    merged_portfolio_sp = pd.merge(merged_portfolio, sp_500_adj_close, left_on='Acquisition Date', right_on='Date')
    # .set_index('Ticker')
    
    # We will delete the additional date column which is created from this merge.
    # We then rename columns to Latest Date and then reflect Ticker Adj Close and SP 500 Initial Close.
    
    del merged_portfolio_sp['Date_y']
    
    merged_portfolio_sp.rename(columns={'Date_x': 'Latest Date', 'Adj Close_x': 'Ticker Adj Close'
                                        , 'Adj Close_y': 'SP 500 Initial Close'}, inplace=True)

    # This new column determines what SP 500 equivalent purchase would have been at purchase date of stock.
    merged_portfolio_sp['Equiv SP Shares'] = merged_portfolio_sp['Cost Basis'] / merged_portfolio_sp['SP 500 Initial Close']
    
    return(merged_portfolio_sp)


merged_portfolio_sp = merged_portfolio_sp(adj_close_latest, portfolio_df, sp_500_adj_close)



def merged_portfolio_sp_latest(merged_portfolio_sp, sp_500_adj_close):
    # We are joining the developing dataframe with the sp500 closes again, this time with the latest close for SP.
    merged_portfolio_sp_latest = pd.merge(merged_portfolio_sp, sp_500_adj_close, left_on='Latest Date', right_on='Date')
    
    # Once again need to delete the new Date column added as it's redundant to Latest Date.  
    # Modify Adj Close from the sp dataframe to distinguish it by calling it the SP 500 Latest Close.
    
    del merged_portfolio_sp_latest['Date']
    
    merged_portfolio_sp_latest.rename(columns={'Adj Close': 'SP 500 Latest Close'}, inplace=True)
    
    # Percent return of SP from acquisition date of position through latest trading day.
    merged_portfolio_sp_latest['SP Return'] = merged_portfolio_sp_latest['SP 500 Latest Close'] / merged_portfolio_sp_latest['SP 500 Initial Close'] - 1
    
    # This is a new column which takes the tickers return and subtracts the sp 500 equivalent range return.
    merged_portfolio_sp_latest['Abs. Return Compare'] = merged_portfolio_sp_latest['ticker return'] - merged_portfolio_sp_latest['SP Return']
    
    # This is a new column where we calculate the ticker's share value by multiplying the original quantity by the latest close.
    merged_portfolio_sp_latest['Ticker Share Value'] = merged_portfolio_sp_latest['Quantity'] * merged_portfolio_sp_latest['Ticker Adj Close']
    
    # We calculate the equivalent SP 500 Value if we take the original SP shares * the latest SP 500 share price.
    merged_portfolio_sp_latest['SP 500 Value'] = merged_portfolio_sp_latest['Equiv SP Shares'] * merged_portfolio_sp_latest['SP 500 Latest Close']
    
    # This is a new column where we take the current market value for the shares and subtract the SP 500 value.
    merged_portfolio_sp_latest['Abs Value Compare'] = merged_portfolio_sp_latest['Ticker Share Value'] - merged_portfolio_sp_latest['SP 500 Value']
    
    # This column calculates profit / loss for stock position.
    merged_portfolio_sp_latest['Stock Gain / (Loss)'] = merged_portfolio_sp_latest['Ticker Share Value'] - merged_portfolio_sp_latest['Cost Basis']
    
    # This column calculates profit / loss for SP 500.
    merged_portfolio_sp_latest['SP 500 Gain / (Loss)'] = merged_portfolio_sp_latest['SP 500 Value'] - merged_portfolio_sp_latest['Cost Basis']
 
    return(merged_portfolio_sp_latest)

merged_portfolio_sp_latest = merged_portfolio_sp_latest(merged_portfolio_sp, sp_500_adj_close)




def merged_portfolio_sp_latest_YTD_sp(merged_portfolio_sp_latest, adj_close_start, sp_500_adj_close_start):

    # Merge the overall dataframe with the adj close start of year dataframe for YTD tracking of tickers.
    # Should not need to do the outer join;
    merged_portfolio_sp_latest_YTD = pd.merge(merged_portfolio_sp_latest, adj_close_start, on='Ticker')
    # , how='outer'
    
    # Deleting date again as it's an unnecessary column.  Explaining that new column is the Ticker Start of Year Close.
    
    del merged_portfolio_sp_latest_YTD['Date']
    
    merged_portfolio_sp_latest_YTD.rename(columns={'Adj Close': 'Ticker Start Year Close'}, inplace=True)
    
    
    # Join the SP 500 start of year with current dataframe for SP 500 ytd comparisons to tickers.
    merged_portfolio_sp_latest_YTD_sp = pd.merge(merged_portfolio_sp_latest_YTD, sp_500_adj_close_start,left_on='Start of Year', right_on='Date')
    
    del merged_portfolio_sp_latest_YTD_sp['Date']
    
    # Renaming so that it's clear this column is SP 500 start of year close.
    merged_portfolio_sp_latest_YTD_sp.rename(columns={'Adj Close': 'SP Start Year Close'}, inplace=True)
    
    # YTD return for portfolio position.
    merged_portfolio_sp_latest_YTD_sp['Share YTD'] = merged_portfolio_sp_latest_YTD_sp['Ticker Adj Close'] / merged_portfolio_sp_latest_YTD_sp['Ticker Start Year Close'] - 1
    
    # YTD return for SP to run compares.
    merged_portfolio_sp_latest_YTD_sp['SP 500 YTD'] = merged_portfolio_sp_latest_YTD_sp['SP 500 Latest Close'] / merged_portfolio_sp_latest_YTD_sp['SP Start Year Close'] - 1
    
    merged_portfolio_sp_latest_YTD_sp = merged_portfolio_sp_latest_YTD_sp.sort_values(by='Ticker', ascending=True)
    
    # Cumulative sum of original investment
    merged_portfolio_sp_latest_YTD_sp['Cum Invst'] = merged_portfolio_sp_latest_YTD_sp['Cost Basis'].cumsum()
    
    # Cumulative sum of Ticker Share Value (latest FMV based on initial quantity purchased).
    merged_portfolio_sp_latest_YTD_sp['Cum Ticker Returns'] = merged_portfolio_sp_latest_YTD_sp['Ticker Share Value'].cumsum()
    
    # Cumulative sum of SP Share Value (latest FMV driven off of initial SP equiv purchase).
    merged_portfolio_sp_latest_YTD_sp['Cum SP Returns'] = merged_portfolio_sp_latest_YTD_sp['SP 500 Value'].cumsum()
    
    # Cumulative CoC multiple return for stock investments
    merged_portfolio_sp_latest_YTD_sp['Cum Ticker ROI Mult'] = merged_portfolio_sp_latest_YTD_sp['Cum Ticker Returns'] / merged_portfolio_sp_latest_YTD_sp['Cum Invst']
    
    return(merged_portfolio_sp_latest_YTD_sp)


merged_portfolio_sp_latest_YTD_sp = merged_portfolio_sp_latest_YTD_sp(merged_portfolio_sp_latest, adj_close_start, sp_500_adj_close_start)





# Assessing where positions are at vs highest close

# Need to factor in that some positions were purchased much more recently than others.
# Join adj_close dataframe with portfolio in order to have acquisition date.


def adj_close_pivot(portfolio_df,adj_close):
    portfolio_df.reset_index(inplace=True)
    adj_close_acq_date = pd.merge(adj_close, portfolio_df, on='Ticker')
    
    del adj_close_acq_date['Quantity']
    del adj_close_acq_date['Unit Cost']
    del adj_close_acq_date['Cost Basis']
    del adj_close_acq_date['Start of Year']
    
    # Sort by these columns in this order in order to make it clearer where compare for each position should begin.
    adj_close_acq_date.sort_values(by=['Ticker', 'Acquisition Date', 'Date'], ascending=[True, True, True], inplace=True)
    
    # Anything less than 0 means that the stock close was prior to acquisition.
    adj_close_acq_date['Date Delta'] = adj_close_acq_date['Date'] - adj_close_acq_date['Acquisition Date']
    adj_close_acq_date['Date Delta'] = adj_close_acq_date[['Date Delta']].apply(pd.to_numeric)  
    
    
    # Modified the dataframe being evaluated to look at highest close which occurred after Acquisition Date (aka, not prior to purchase).
    adj_close_acq_date_modified = adj_close_acq_date[adj_close_acq_date['Date Delta']>=0]
    
    # This pivot table will index on the Ticker and Acquisition Date, and find the max adjusted close.
    adj_close_pivot = adj_close_acq_date_modified.pivot_table(index=['Ticker', 'Acquisition Date'], values='Adj Close', aggfunc=np.max)
    adj_close_pivot.reset_index(inplace=True)

    return(adj_close_pivot)

adj_close_pivot = adj_close_pivot(portfolio_df,adj_close)




# Merge the adj close pivot table with the adj_close table in order to grab the date of the Adj Close High (good to know).
adj_close_pivot_merged = pd.merge(adj_close_pivot, adj_close
                                             , on=['Ticker', 'Adj Close'])


# Merge the Adj Close pivot table with the master dataframe to have the closing high since you have owned the stock.
merged_portfolio_sp_latest_YTD_sp_closing_high = pd.merge(merged_portfolio_sp_latest_YTD_sp, adj_close_pivot_merged
                                             , on=['Ticker', 'Acquisition Date'])


# Renaming so that it's clear that the new columns are two year closing high and two year closing high date.
merged_portfolio_sp_latest_YTD_sp_closing_high.rename(columns={'Adj Close': 'Closing High Adj Close', 'Date': 'Closing High Adj Close Date'}, inplace=True)
merged_portfolio_sp_latest_YTD_sp_closing_high['Pct off High'] = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker Adj Close'] / merged_portfolio_sp_latest_YTD_sp_closing_high['Closing High Adj Close'] - 1 


"""
Charts needed on dashboard
"""
#FINAL
#YTD Chart

trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp['Share YTD'][0:10],
    name = 'Ticker YTD')

trace2 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp['SP 500 YTD'][0:10],
    name = 'SP500 YTD')
    
data = [trace1, trace2]

layout = go.Layout(title = 'YTD Return vs S&P 500 YTD'
    , barmode = 'group'
    , yaxis=dict(title='Returns', tickformat=".2%")
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.8,y=1)
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)

fig.write_html('first_figure.html', auto_open=True)


# FINAL
# Trailing stop Chart
# Current Share Price versus Closing High Since Purchased
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Pct off High'][0:10],
    name = 'Pct off High')
    
data = [trace1]

layout = go.Layout(title = 'Adj Close % off of High'
    , barmode = 'group'
    , yaxis=dict(title='% Below Adj Close High', tickformat=".2%")
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.8,y=1)
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig.write_html('second_figure.html', auto_open=True)


#FINAL
#Total Return Comparison high
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['ticker return'][0:10],
    name = 'Ticker Total Return')

trace2 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['SP Return'][0:10],
    name = 'SP500 Total Return')
    
data = [trace1, trace2]

layout = go.Layout(title = 'Total Return vs S&P 500'
    , barmode = 'group'
    , yaxis=dict(title='Returns', tickformat=".2%")
    , xaxis=dict(title='Ticker', tickformat=".2%")
    , legend=dict(x=.8,y=1)
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig.write_html('third_figure.html', auto_open=True)


#FINAL
#Cumulative returns over time
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Stock Gain / (Loss)'][0:10],
    name = 'Ticker Total Return ($)')

trace2 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['SP 500 Gain / (Loss)'][0:10],
    name = 'SP 500 Total Return ($)')

trace3 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['ticker return'][0:10],
    name = 'Ticker Total Return %',
    yaxis='y2')

data = [trace1, trace2, trace3]

layout = go.Layout(title = 'Gain / (Loss) Total Return vs S&P 500'
    , barmode = 'group'
    , yaxis=dict(title='Gain / (Loss) ($)')
    , yaxis2=dict(title='Ticker Return', overlaying='y', side='right', tickformat=".2%")
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.75,y=1)
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig.write_html('fourth_figure.html', auto_open=True)



#FINAL
#Total cumulative investment over time
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Invst'],
    # mode = 'lines+markers',
    name = 'Cum Invst')

trace2 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum SP Returns'],
    # mode = 'lines+markers',
    name = 'Cum SP500 Returns')

trace3 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Ticker Returns'],
    # mode = 'lines+markers',
    name = 'Cum Ticker Returns')

trace4 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Ticker ROI Mult'],
    # mode = 'lines+markers',
    name = 'Cum ROI Mult'
    , yaxis='y2')


data = [trace1, trace2, trace3, trace4]

layout = go.Layout(title = 'Total Cumulative Investments Over Time'
    , barmode = 'group'
    , yaxis=dict(title='Returns')
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.4,y=1)
    , yaxis2=dict(title='Cum ROI Mult', overlaying='y', side='right')               
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)

fig.write_html('fifth_figure.html', auto_open=True)
