import requests
from info import news_sources
from datetime import datetime, timedelta
import json
import os
class news(object):
    def __init__(self,keywords):
        self.api_key = os.environ['API_KEY']
        self.keywords = keywords
        self.url = "http://newsapi.org/v2/everything"
        self.from_date = (
            (datetime.now() - timedelta(days=15)).date().strftime("%Y-%m-%d")
        )

    def build_parameters(self):
        print(self.keywords)
        return {
            "q": self.keywords,
            # "qInTitle":self.keywords,
            "pageSize": 20,
            "apiKey": self.api_key,
            "sources": ",".join(news_sources),
            "sortBy": "popularity",
            "from": self.from_date,
            "language":"en",
        }

    def get_news(self):
        try:
            response = requests.get(self.url, params=self.build_parameters())
            return response.json()
        except ConnectionError as e:
            print("Issue with Connecting with news sources")
            return None

    def remove_dupe_dicts(self,news_list):
        list_of_strings = [
            json.dumps(news, sort_keys=True)
            for news in news_list
        ]
        list_of_strings = set(list_of_strings)
        
        return [
            json.loads(s)
            for s in list_of_strings
        ]


    def cleanup_news(self):
        all_news = self.get_news()
        if all_news:
            data = [
                {
                    "title": news["title"],
                    "url": news["url"],
                    "source":news["source"]["name"],
                    "urlToImage":news["urlToImage"]
                }
                for news in all_news["articles"]
            ]
            return data
        return None