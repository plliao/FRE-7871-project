import re
import json
import os
import string
import pickle
import datetime

import ipdb
import pandas as pd

from reuter_data import logging

NEWS_MONTH = ['07', '08', '09', '10']
NEWS_NUMBER = [14793, 11978, 11337, 9743]

def clean_sentence(s):
    s = re.sub("\n", " ", s)
    s = re.sub("[" + string.punctuation + "]", " ", s)
    s = re.sub(" +", " ", s)
    return s.strip()

def collect_webhose_news():
    df = pd.DataFrame()
    for i, month in enumerate(NEWS_MONTH):
        for index in range(1, NEWS_NUMBER[i] + 1):
            logging('{0:s}: {1:05d}/{2:05d}'.format(month, index, NEWS_NUMBER[i]))
            news_path = "data/{0:s}/news_{1:07d}.json".format(month, index)
            with open(news_path) as f:
                datum_json = json.load(f)

            datum = pd.Series(
                data={
                    'text':datum_json['text'],
                    'published_time':datum_json['published'],
                    'country':datum_json['thread']['country'],
                    'title':datum_json['thread']['title'],
                    'site':datum_json['thread']['site']
                }
            )
            df = df.append(datum, ignore_index=True)
    df.to_csv('webhose_data.csv', index=False)
    ipdb.set_trace()

def read_news_dataframe(news_per_day):
    df = pd.DataFrame()
    date_news_count = {}
    for i, month in enumerate(NEWS_MONTH):
        for index in range(NEWS_NUMBER[i]):
            news_path = "data/{0:s}/news_{1:07d}.json".format(month, NEWS_NUMBER[i] - index)
            with open(news_path) as f:
                datum_json = json.load(f)

            publish_time = pd.to_datetime(datum_json['published'])
            date_str = publish_time.strftime('%Y-%m-%d')
            start_time = datetime.datetime.strptime('{0:s} 0930'.format(date_str), '%Y-%m-%d %H%M')
            end_time = datetime.datetime.strptime('{0:s} 1600'.format(date_str), '%Y-%m-%d %H%M')

            if date_str not in date_news_count:
                date_news_count[date_str] = 0

            if date_news_count[date_str] > news_per_day:
                continue

            if publish_time <= start_time or publish_time >= end_time:
                continue
            if datum_json['thread']['country'] != 'US' or 'finance' not in str(datum_json):
                continue

            text = clean_sentence(datum_json['text'])
            if len(text.split(' ')) < 100:
                continue

            date_news_count[date_str] += 1
            datum = pd.Series(
                data={
                    'text':text,
                    'date':date_str
                }
            )
            df = df.append(datum, ignore_index=True)

    #pickle.dump(date_news_count, open("data/date_news_count.p", "wb"))
    return df

def read_SNP_dataframe():
    snp = pd.read_csv("data/GSPC.csv")
    snp['target'] = pd.Series('2015-06-30').append(snp['Date'][0:-1], ignore_index=True)
    for offset in range(1, 6):
        name = 'previous_price_{0:d}'.format(offset)
        snp[name] = pd.Series([None for _ in range(offset)]).append(snp['Close'][0:-offset] - snp['Open'][0:-offset], ignore_index=True)
    return snp

def find_price(ticker, timestamp):
    date_str = timestamp.strftime('%Y%m%d')
    path = 'data/SNP/{0:s}/price/price_{1:s}.json'.format(ticker, date_str)

    if not os.path.exists(path):
        return None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    start_time = datetime.datetime.strptime(date_str + ' 09:30', '%Y%m%d %H:%M')
    offset = int((timestamp - start_time).total_seconds() // 60)
    return data[offset]

def read_reuter_csv():
    reuter = pd.read_csv('reuter_data.csv')
    reuter['published_time'] = pd.to_datetime(reuter['published_time'])
    reuter.sort_values('published_time', inplace=True)

    twenty_min = datetime.timedelta(minutes=20)

    df = pd.DataFrame()

    for _, article in reuter.iterrows():
        ticker = article['ticker']
        published_time = article['published_time']

        date_str = published_time.strftime('%Y%m%d')
        start_time = datetime.datetime.strptime('{0:s} 0930'.format(date_str), '%Y%m%d %H%M')
        end_time = datetime.datetime.strptime('{0:s} 1600'.format(date_str), '%Y%m%d %H%M')

        predicted_time = published_time + twenty_min

        if published_time > start_time and published_time < end_time and \
                predicted_time > start_time and predicted_time < end_time:
            price = find_price(ticker, published_time)
            predicted_price = find_price(ticker, predicted_time)

            if price is None or predicted_price is None:
                continue

            datum = pd.Series(
                data={
                    'text':clean_sentence(article['text']),
                    'publish_time':published_time,
                    'predicted_time':predicted_time,
                    'price':price['high'],
                    'predicted_price':price['low'],
                    'ticker':article['ticker'],
                    'name':article['name'],
                    'title':article['title']
                }
            )
            df = df.append(datum, ignore_index=True)
    df.to_csv('reuter_price.csv', index=False)
    ipdb.set_trace()


if __name__ == '__main__':
    collect_webhose_news()
