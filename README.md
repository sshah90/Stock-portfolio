# Stock-Portfolio 

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
git clone https://github.com/sshah90/stock-portfolio.git
cd stock-portfolio/
docker build -t stock:1.0.0 .
```
### Run Image (Tested in Mac only)

```bash
docker run \
    -p 8501:8501 \
    -e API_KEY="YOUR_API_KEY"\
    -v /your/path:/data\
    stock:1.0.0 
```
If everything works fine then you should able to see dashboard at `localhost:8501`.