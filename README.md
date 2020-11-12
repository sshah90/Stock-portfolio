# Stock-Portfolio 

[![python](https://img.shields.io/badge/python-3.7-brightgreen)](https://www.python.org/downloads/release/python-370/)
[![streamlit](https://img.shields.io/badge/streamlit-0.70-red)](https://www.streamlit.io/)

- Python app for Stock-Portfolio dashboard using [yahoo finance APIs](https://github.com/ranaroussi/yfinance) and [Streamlit](https://www.streamlit.io/)
- Dashboard will accept different parameters from user and give analysis for tickers.
- It will also show you news for selected ticker's industry using [news api](https://newsapi.org/).

## Getting Started

These instructions will cover usage information about running app locally. 

### Prerequisites

* GIT 
* [Docker](https://www.docker.com/products/docker-desktop)
* [Newsapi API key](https://newsapi.org/)

### Build Image
```bash
git clone https://github.com/sshah90/Stock-portfolio.git
cd Stock-portfolio/
docker build -t stock:1.0.0 .
```
### Run Image

```bash
docker run \
    -p 8501:8501 \
    -e API_KEY="YOUR_API_KEY"\
    -v /your/path:/data\
    stock:1.0.0 
```
If everything works fine then you should able to see dashboard at `localhost:8501`.

### Running without Docker

This code base only work with `Python >= 3.7.4`.

After cloning repo, run below commands

```bash
git clone https://github.com/sshah90/Stock-portfolio.git
cd Stock-portfolio/
pip install -r requirement.txt
streamlit run frontend.py
```
If you have any problem installing `fbprophet` then check installing documentation [here](https://github.com/facebook/prophet#installation-in-python)
