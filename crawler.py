import re
import os
import pickle
import time
import datetime
import json
import ipdb

import requests
import pandas as pd
from bs4 import BeautifulSoup

VALID_TAGS = ['div', 'p']

def clean_tag(tag):
    return re.sub('<[^<>]*>', ' ', str(tag)).strip()

def clean_html(soup):
    doc = ""
    for tag in soup.findAll('p'):
        if tag.name in VALID_TAGS:
            doc += clean_tag(tag)
    return doc

def crawl_IEX_news(ticker, dir_path):
    url = "https://api.iextrading.com/1.0/stock/{0:s}/news/last/1000".format(ticker.lower())
    result = requests.get(url)

    pickle.dump(result, open(dir_path + "/request.p", "wb"))

    for index, news in enumerate(result.json()):
        time.sleep(0.5)
        file_name = "news_{0:04d}".format(index + 1)

        news_result = requests.get(news['url'])
        pickle.dump(news_result, open(dir_path + "/raw/" + file_name + ".p", "wb"))

        soup = BeautifulSoup(news_result.text, 'html5lib')
        news_doc = clean_html(soup)

        file_path = dir_path + "/doc/" + file_name + ".txt"
        with open(file_path, "w") as f:
            f.write(news_doc)

def crawl_IEX_price(ticker, dir_path):
    date = datetime.datetime.strptime('20180702', '%Y%m%d')
    end_date = datetime.datetime.today()
    one_day = datetime.timedelta(days=1)

    while date <= end_date:
        time.sleep(0.5)
        date_str = date.strftime('%Y%m%d')
        url = "https://api.iextrading.com/1.0/stock/{0:s}/chart/date/{1:s}".format(
            ticker.lower(), date_str
        )
        result = requests.get(url)
        if len(result.json()) > 0:
            file_name = "price_{0:s}.json".format(date_str)
            pickle.dump(result.json(), open(dir_path + "/price/" + file_name, "wb"))

        date = date + one_day

def crawl_reuters(dir_path):
    date = datetime.datetime.strptime('20180703', '%Y%m%d')
    end_date = datetime.datetime.today()
    one_day = datetime.timedelta(days=1)
    while date <= end_date:
        time.sleep(1)
        date_str = date.strftime('%Y%m%d')
        url = 'https://www.reuters.com/resources/archive/us/{0:s}.html'.format(date_str)
        result = requests.get(url)

        if result.status_code == 200:
            articles = []

            soup = BeautifulSoup(result.text, 'html.parser')
            tags = soup.find_all('div', "headlineMed")

            print("Date: {0:s}".format(date_str))
            for index, tag in enumerate(tags):
                time.sleep(1)
                try:
                    link = tag.contents[0]['href']
                    published_time = date_str + " " + tag.contents[1][1:8]
                    title = tag.contents[0].contents[0]

                    print("\t{0:05d}: {1:s}".format(index, title))
                    news_result = requests.get(link)
                    if news_result.status_code != 200:
                        continue

                    news_soup = BeautifulSoup(news_result.text, 'html5lib')
                    news_tag = news_soup.find('div', "ArticleHeader_channel")

                    if news_tag is None or not hasattr(news_tag, 'contents') or not hasattr(news_tag.contents[0], 'contents'):
                        continue

                    news_type = news_soup.find('div', "ArticleHeader_channel").contents[0].contents[0]
                    if "Business" in news_type:
                        article_body_tags = news_soup.find('div', "StandardArticleBody_body")
                        article = ""
                        for article_tag in article_body_tags:
                            if article_tag.name == 'p':
                                content = article_tag.contents[0]
                                if hasattr(content, 'contents'):
                                    content = content.contents[0]
                                article = article + " " + content
                        articles.append({'published_time':published_time, 'title':title, 'text':article})
                except Exception as err:
                    print("\t\t" + str(err))
                    pass

            if len(articles) > 0:
                print("Date: {0:s}, {1:d} business articles".format(date_str, len(articles)))
                file_name = "news_{0:s}.json".format(date_str)
                with open(dir_path + "/" + file_name, "w") as f:
                    f.write(json.dumps(articles))

        date = date + one_day

def main():
    snp = pd.read_csv('data/constituents.csv')
    for ticker in snp['Symbol']:
        time.sleep(1)
        dir_path = 'data/SNP/' + ticker
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            os.mkdir(dir_path + "/raw")
            os.mkdir(dir_path + "/doc")
            os.mkdir(dir_path + "/price")

        crawl_IEX_news(ticker, dir_path)
        crawl_IEX_price(ticker, dir_path)
        crawl_reuters("data/reuters")


if __name__ == '__main__':
    main()
