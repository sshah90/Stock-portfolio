import json
import os.path
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from matplotlib.pyplot import rc
from pandas_datareader import data as pdr
from PIL import Image

from info import info_list
from news import news
from predict import predict
from analysis import analysis, monte

yf.pdr_override()
st.set_option("deprecation.showPyplotGlobalUse", False)


def infopage():
    if not os.path.exists("data/data.json"):
        name_text = st.text_input(
            "Please Enter your name",
        )
        load_button = st.button("Save")
        if load_button and len(name_text) > 3:
            with open("data/data.json", "w") as f:
                json.dump(
                    {
                        "name": name_text,
                    },
                    f,
                )
                st.experimental_rerun()
                return None


@st.cache(allow_output_mutation=True)
def load_data(ticker, start, end):
    data = pdr.get_data_yahoo(ticker, start=start, end=end, interval="1d")
    return data


@st.cache(allow_output_mutation=True)
def load_prediction_data(ticker):
    data = pdr.get_data_yahoo(
        ticker, start="2015-01-01", end=datetime.today(), interval="1d"
    )
    return data["Adj Close"]


def get_ticker_object(ticker):
    ticker = yf.Ticker(ticker)
    return ticker


def sort_dataframe(dataframe, column="Date"):
    return dataframe.sort_values(by=[column], ascending=False).head(10)


def summary_page_load(ticker):
    ticker_obj = get_ticker_object(ticker)
    ticker_info = ticker_obj.info
    response = requests.get(ticker_info["logo_url"])
    image = Image.open(BytesIO(response.content))

    st.image(image, caption="")
    st.title(ticker_info["longName"])
    st.markdown(ticker_info["longBusinessSummary"])

    needed_info = {info: ticker_info[info] for info in info_list}
    st.title("Summary")
    st.dataframe(pd.DataFrame(needed_info.items()).assign(hack="").set_index("hack"))

    recommendations_df = sort_dataframe(ticker_obj.recommendations)
    st.title("Top firm Recommendations")
    st.table(recommendations_df.assign(hack="").set_index("hack"))

    st.title("Major Holders")
    st.table(ticker_obj.major_holders.assign(hack="").set_index("hack"))


def valid_dates(date_list: List) -> List:
    for date in date_list:
        try:
            convert = parse(date).date()
        except:
            st.error("Please enter correct date")
            return False

        if convert <= datetime(2016, 1, 1, 0, 0).date():
            st.error("Please insert a date posterior to 1st January 2019")
            return False
    return True


def select_date_range():
    one_day = st.radio(
        "Select Range Preferences",
        [
            "Slider",
            "Custom Dates",
        ],
    )
    end_date = datetime.today()
    end_date_string = end_date.strftime("%Y-%m-%d")
    if one_day == "Slider":
        months = st.slider("", 3, 24, 3, 3, format="%d months")
        start_date = end_date + relativedelta(months=-int(months))
        return (start_date.strftime("%Y-%m-%d"), end_date_string)

    start_text = st.text_input(
        "Enter the start date in yyyy-mm-dd format:",
        (end_date - timedelta(3)).strftime("%Y-%m-%d"),
    )
    end_text = st.text_input(
        "Enter the end date in yyyy-mm-dd format:", end_date_string
    )
    if valid_dates([start_text, end_text]):
        return (start_text, end_text)


def set_pub():
    rc("font", weight="bold")  # bold fonts are easier to see
    rc("grid", c="0.5", ls="-", lw=0.5)
    rc("figure", figsize=(10, 8))
    plt.style.use("seaborn-whitegrid")
    rc("lines", linewidth=1.3, color="b")


def plotData(ticker):
    start, end = select_date_range()
    df_stockdata = load_data(ticker, start, end)["Adj Close"]
    df_stockdata.index = pd.to_datetime(df_stockdata.index)

    set_pub()
    fig, ax = plt.subplots(2, 1)

    ma1_checkbox = st.checkbox("Moving Average 1")
    ma2_checkbox = st.checkbox("Moving Average 2")

    ax[0].set_title("Adj Close Price %s" % ticker, fontdict={"fontsize": 15})
    ax[0].plot(df_stockdata.index, df_stockdata.values, "g-", linewidth=1.6)
    ax[0].set_xlim(ax[0].get_xlim()[0] - 10, ax[0].get_xlim()[1] + 10)
    ax[0].grid(True)

    if ma1_checkbox:
        days1 = st.slider("Business Days to roll MA1", 5, 120, 30)
        ma1 = df_stockdata.rolling(days1).mean()
        ax[0].plot(ma1, "b-", label="MA %s days" % days1)
        ax[0].legend(loc="best")
    if ma2_checkbox:
        days2 = st.slider("Business Days to roll MA2", 5, 120, 30)
        ma2 = df_stockdata.rolling(days2).mean()
        ax[0].plot(ma2, color="magenta", label="MA %s days" % days2)
        ax[0].legend(loc="best")

    ax[1].set_title("Daily Total Returns %s" % ticker, fontdict={"fontsize": 15})
    ax[1].plot(df_stockdata.index[1:], df_stockdata.pct_change().values[1:], "r-")
    ax[1].set_xlim(ax[1].get_xlim()[0] - 10, ax[1].get_xlim()[1] + 10)
    plt.tight_layout()
    ax[1].grid(True)
    st.pyplot(plt)


def homepage():
    sp500_list = pd.read_csv("static_data/SP500_list.csv")

    ticker = (
        st.selectbox(
            "Select the ticker if present in the S&P 500 index",
            sp500_list["Symbol"] + " (" + sp500_list["Name"] + ")",
            # format_func = format_of_list,
            index=30,
        )
        .split(" (")[0]
        .upper()
    )

    checkbox_noSP = st.checkbox(
        "Select this box to write the ticker (if not present in the S&P 500 list). \
                                Deselect to come back to the S&P 500 index stock list"
    )
    if checkbox_noSP:
        ticker = st.text_input(
            "Write the ticker (check it in yahoo finance)", "MN.MI"
        ).upper()

    return ticker


st.set_page_config(
    page_title="Stock Portfolio",
    initial_sidebar_state="collapsed",
    page_icon=":dollar:",
)

if os.path.exists("data/data.json"):
    f = open("data/data.json")
    data = json.load(f)
    st.sidebar.header(f"Welcome {data['name']}!\n")
else:
    st.sidebar.header(f"Welcome Stranger!\n")

image = Image.open("static_data/stock.jpeg")

st.sidebar.image(image, caption="", use_column_width=True)
st.sidebar.subheader("Choose the option to visualize")

choose_options = st.sidebar.radio(
    "",
    ["Home Page", "Personalized Portfolio", "Update Personal Stock List"],
)


if choose_options == "Home Page":
    infopage()
    ticker = homepage()
    choose_options = st.sidebar.radio(
        "Options",
        ["Company Analysis", "Close/Return", "News"],
    )
    if choose_options == "Company Analysis":
        summary_page_load(ticker)
    elif choose_options == "Close/Return":
        plotData(ticker)
    else:
        ticker_obj = get_ticker_object(ticker)
        ticker_info = ticker_obj.info
        # query = ticker_info['longName'].replace("Inc.","")
        n = news(f"{ticker_info['industry']}")
        all_news = n.cleanup_news()
        if all_news and len(all_news) > 0:
            for news in all_news:
                col1, col2 = st.beta_columns(2)
                try:
                    response = requests.get(news["urlToImage"])
                    image = Image.open(BytesIO(response.content))
                except:
                    image = Image.open("static_data/no_image.png")
                col1.image(
                    image, caption=f"Source : {news['source']}", use_column_width=True
                )
                col2.markdown(f"[{news['title']}]({news['url']})")
        else:
            st.title("Sorry! couldn't find any news for you.")

if choose_options == "Update Personal Stock List":
    uploaded_file = st.file_uploader(
        "Upload CSV file with stocks details", type=["csv"]
    )
    if uploaded_file:
        dataframe = pd.read_csv(uploaded_file)
        stock_list = dataframe.to_dict("records")
        with open("data/data.json", "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["stock_list"] = stock_list
            f.seek(0)
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4,
            )
            f.truncate()
        with st.spinner("Wait for it..."):
            time.sleep(1)
        st.success("Now your stock list is up to date!, please find your list below")
        st.write(dataframe)

if choose_options == "Personalized Portfolio":
    if os.path.exists("data/data.json"):
        f = open("data/data.json")
        all_stock_metadata = json.load(f)["stock_list"]
        all_tickers = [tickers["ticker"] for tickers in all_stock_metadata]
        choose_options_personalized = st.sidebar.radio(
            "Options",
            ["Predictions", "Analysis"],
        )
        if choose_options_personalized == "Predictions":
            ticker = st.selectbox(
                "Please Select the Ticker from your stock",
                all_tickers,
                index=0,
            ).upper()
            period = st.slider(
                "How many periods would you like to forecast into the future?",
                15,
                120,
                15,
                15,
                format="%d days",
            )
            ticker_obj = get_ticker_object(ticker)
            ticker_info = ticker_obj.info

            train_df = load_prediction_data(ticker)
            pr_ob = predict(train_df)
            fig, fig2, fig3 = pr_ob.prediction(int(period))
            fig.update_layout(
                title=f"Prediction for {ticker_info['longName']}",
                yaxis_title="Adj Close Price",
                xaxis_title="Date",
            )
            """
            #### The next visual shows the actual (black dots) and predicted (blue line) values over time.
            """
            st.plotly_chart(fig, use_container_width=True)

            """
            #### The next few visuals show a high level trend of predicted values.
            
            """
            st.write(fig2)
        if choose_options_personalized == "Analysis":
            """### Portfolio Summary"""
            fig_1, fig_2, fig_3, fig_4, fig_5 = analysis()
            st.plotly_chart(fig_1, use_container_width=True)
            st.plotly_chart(fig_2, use_container_width=True)
            st.plotly_chart(fig_3, use_container_width=True)
            st.plotly_chart(fig_4, use_container_width=True)
            st.plotly_chart(fig_5, use_container_width=True)

            checkbox_monte = st.checkbox(
                "Select this box if you want run  Monte Carlo simulation"
            )
            if checkbox_monte:
                simulation = st.slider(
                    "How many simulation  would you like to run?",
                    1000,
                    15000,
                    3000,
                    500,
                    format="%d days",
                )
                df_with_stocks = load_data(
                    all_tickers, start="2015-01-01", end=datetime.today()
                )["Adj Close"]
                fig, max_sharpe_port, min_vol_port = monte(
                    df_with_stocks, all_tickers, simulation
                )
                st.pyplot(fig)
                """
                ### Summary of portfolio and weight of stocks where sharpe ratio is the highest (Red Star)
                """
                st.dataframe(max_sharpe_port.to_frame().T)
                """
                ### Summary of portfolio and weight of stocks that has the low volatility (Green Star)
                """
                st.dataframe(min_vol_port.to_frame().T)
    else:
        st.title("Sorry! I couldn't find any data, please update personal stock list")
