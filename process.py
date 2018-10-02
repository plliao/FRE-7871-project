import re
import json
import string
import pickle
import datetime

import ipdb
import pandas as pd

NEWS_MONTH = ['07', '08', '09', '10']
NEWS_NUMBER = [14793, 11978, 11337, 9743]

def clean_sentence(s):
    s = re.sub("\n", " ", s)
    s = re.sub("[" + string.punctuation + "]", " ", s)
    s = re.sub(" +", " ", s)
    return s.strip()

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
        name = 'previous_close_{0:d}'.format(offset)
        snp[name] = pd.Series([None for _ in range(offset)]).append(snp['Close'][0:-offset], ignore_index=True)
    return snp

