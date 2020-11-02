import streamlit as st
import numpy as np
import pandas as pd
import time
import yfinance as yf

from dateutil.parser import parse

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from typing import List
import os.path
import json
from io import StringIO


def infopage():
    # if os.path.isfile("data.json"):
    #     homepage()
    #     return None
    f = open("data.json")
    data = json.load(f)
    if data["name"] == "Stranger":
        name_text = st.text_input("Please Enter your name",)
        load_button = st.button("Save")
        if load_button and len(name_text) > 3:
            print("I am here1")
            with open("data.json", "w") as f:
                json.dump({"name": name_text,}, f)
                st.experimental_rerun()
                return None
    homepage()


def load_data(ticker, start, end):
    tsla = yf.Ticker(ticker)
    hist = tsla.history(start=start, end=end, interval="1d",)
    st.line_chart(hist[["Close", "Low", "High"]])
    st.line_chart(hist["Volume"])


def homepage():
    # f"""# Welcome {data['name']}!\n ## Loading personalized Stock Portfolio"""

    sp500_list = pd.read_csv("SP500_list.csv")
    ticker = st.selectbox(
        "Select the ticker if present in the S&P 500 index",
        sp500_list["Symbol"],
        index=30,
    ).upper()

    checkbox_noSP = st.checkbox(
        "Select this box to write the ticker (if not present in the S&P 500 list). \
                                Deselect to come back to the S&P 500 index stock list"
    )
    if checkbox_noSP:
        ticker = st.text_input(
            "Write the ticker (check it in yahoo finance)", "MN.MI"
        ).upper()

    start, end = select_date_range()
    load_data(ticker, start, end)


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


def select_date_range():
    one_day = st.radio("Select Range Preferences", ["Slider", "Custom Dates",],)
    end_date = datetime.today()
    end_date_string = end_date.strftime("%Y-%m-%d")
    if one_day == "Slider":
        age = st.slider("", 3, 24, 3, 3, format="%d months")
        start_date = end_date + relativedelta(months=-int(age))
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


f = open("data.json")
data = json.load(f)
st.sidebar.header(f"Welcome {data['name']}!\n")
st.sidebar.subheader("Choose the option to visualize")

choose_options = st.sidebar.radio(
    "", ["Home Page", "Personal Page", "Update Personal Stock List"],
)

if choose_options == "Home Page":
    infopage()
elif choose_options == "Update Personal Stock List":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with stocks details", type=["csv"]
    )
    if uploaded_file:
        bytesData = uploaded_file.getvalue()
        s = str(bytesData, "utf-8").split("\n")
        stock_list = list(filter(None, [stock for stock in s]))
        data["stock_list"] = stock_list
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(
                data, f, ensure_ascii=False, indent=4,
            )
        with st.spinner("Wait for it..."):
            time.sleep(2)
        st.balloons()
        st.success("Now your stock list is up to date!, please find your list below")
        df = pd.DataFrame(data["stock_list"], columns=["Stock List"])
        st.write(df)
else:
    st.write("personal")
