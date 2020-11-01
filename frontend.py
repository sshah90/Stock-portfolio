import streamlit as st
import numpy as np
import pandas as pd
import time
import yfinance as yf

from dateutil.parser import parse

from datetime import datetime

from typing import List

#ToDo

# Push code
# Slider
# User Input (Store it) 
# Sector wise 
# metrics ? 


today = datetime.today().strftime("%Y-%m-%d")

name = "Smit"
f""" # Welcome {name}
    # Stock Portfolio
   ### (stock prices from *yahoo finance*) """

sp500_list = pd.read_csv("SP500_list.csv")


ticker = st.selectbox(
    "Select the ticker if present in the S&P 500 index", sp500_list["Symbol"], index=30
).upper()

checkbox_noSP = st.checkbox(
    "Select this box to write the ticker (if not present in the S&P 500 list). \
                            Deselect to come back to the S&P 500 index stock list"
)
if checkbox_noSP:
    ticker = st.text_input(
        "Write the ticker (check it in yahoo finance)", "MN.MI"
    ).upper()


start = st.text_input("Enter the start date in yyyy-mm-dd format:", today)
end = st.text_input("Enter the end date in yyyy-mm-dd format:", today)


def valid_dates(date_list: List) -> List:
    for date in date_list:
        try:
            convert = parse(date).date()
        except:
            st.error("Please enter correct date")
            return False

        if convert <= datetime(2019, 1, 1, 0, 0).date():
            st.error("Please insert a date posterior to 1st January 2019")
            return False
    return True

valid_dates([start,end])

tsla = yf.Ticker(ticker)

hist = tsla.history(start=start, end=end,interval = "1d",)
# print(hist[['Open','Close']])
# lowercase = lambda x: str(x).lower()
# hist.rename(lowercase, axis='columns', inplace=True)
# hist['Date'] = pd.to_datetime(hist['Date'])
# print(hist['close','open'])

st.line_chart(hist[['Close','Low','High']])
st.line_chart(hist['Volume'])


st.sidebar.header('Nice Header')
st.sidebar.subheader('Choose the option to visualize')

stock_info = st.sidebar.checkbox('Get Stock Info', value = True)

# hour_to_filter = st.slider('hour', 0, 365, 5)  # min: 0h, max: 23h, default: 17h


# if stock_info:
#     st.write("Company info")

# ticker_test = st.sidebar.selectbox(
#     "Select the ticker if present in the S&P 500 index", sp500_list["Symbol"], index=30
# ).upper()

# print(tsla['Date'])
# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)

# hist = tsla.history(period="7d",interval = "1d",)
# print(hist['Open'])

# st.line_chart(hist['Close'])
# st.line_chart(hist['Open'])
# def main():
#     st.header('Hello')
#     st.balloons()


# login_blocks = generate_login_block()
# password = login(login_blocks)

# if is_authenticated(password):
#     clean_blocks(login_blocks)
#     main()
# elif password:
#     st.info("Please enter a valid password")

# print(valid_dates([start, end]))


# st.title('Stock Portfolio')

# tsla = yf.Ticker("TSLA")

# get historical market data
# hist = tsla.history(period="ytd",interval = "1d",)

# st.write(hist)

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# Create a text element and let the reader know the data is loading.


# Load 10,000 rows of data into the dataframe.

# data = load_data(10000)

# Notify the reader that the data was successfully loaded.

# data_load_state.text("Done! (using st.cache)")

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')

# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

# st.bar_chart(hist_values)

# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)

# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)

# for i in range(1, 101):
#     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     chart.add_rows(new_rows)
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.05)

# progress_bar.empty()

# # Streamlit widgets automatically run the script from top to bottom. Since
# # this button is not connected to any other logic, it just causes a plain
# # rerun.
# st.button("Re-run")


# import yfinance as yf

# msft = yf.Ticker("TSLA")

# print(msft.info)

