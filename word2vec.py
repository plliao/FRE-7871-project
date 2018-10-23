import ipdb

import pandas as pd
import gensim.downloader as api

from new_model import average_word2vec
from new_model import preprocess_news_df

MODEL_LIST = ['glove-twitter-25', 'glove-wiki-gigaword-300', 'glove-twitter-200', 'word2vec-google-news-300', 'conceptnet-numberbatch-17-06-300']

def encode(news_df, label, model):
    features = news_df['text'].apply(average_word2vec, model=model)
    file_name = 'word2vec/{0:s}_{1:s}.csv'.format(label, model_name)
    features.to_csv(file_name, index=False)

def read_news_df(path):
    news_df = pd.read_csv(path)
    return preprocess_news_df(news_df)

if __name__ == "__main__":
    reuters = read_news_df("data/reuter_price.csv")
    webhose = read_news_df("data/webhose_price_trend.csv")
    for model_name in MODEL_LIST:
        model = api.load(model_name)
        encode(reuters, 'reuters', model)
        encode(webhose, 'webhose', model)

