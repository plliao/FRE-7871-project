import ipdb
import datetime

import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt

def plot_freq_hist(data, title, xlabel, fname):
    rcParams.update({'figure.autolayout': True})
    plt.cla()
    ax = data.plot(kind='bar', title=title)
    ax.set(xlabel=xlabel, ylabel='Frequency')
    plt.savefig(fname)

def is_in_trading_time(published_time):
    date_str = published_time.strftime('%Y%m%d')
    start_time = datetime.datetime.strptime('{0:s} 0930'.format(date_str), '%Y%m%d %H%M')
    end_time = datetime.datetime.strptime('{0:s} 1600'.format(date_str), '%Y%m%d %H%M')
    return published_time >= start_time and published_time <= end_time

def map_trading_date(published_time):
    date_str = published_time.strftime('%Y%m%d')
    start_time = datetime.datetime.strptime('{0:s} 0930'.format(date_str), '%Y%m%d %H%M')
    end_time = datetime.datetime.strptime('{0:s} 1600'.format(date_str), '%Y%m%d %H%M')
    next_day = published_time + datetime.timedelta(days=1)

    if published_time < start_time:
        return date_str
    elif published_time > end_time:
        return next_day.strftime('%Y%m%d')
    else:
        return None

def plot_news_per_day(df, title, fname):
    df['datetime'] = pd.to_datetime(df['published_time'])
    trading_index = df['datetime'].apply(is_in_trading_time)
    trading = df[trading_index]
    trading_group = trading.groupby(trading['datetime'].apply(lambda x: x.strftime('%Y%m%d')))
    trading_day_news = trading_group.count()['datetime'].rename('open')

    non_trading = df[~trading_index]
    non_trading_group = non_trading.groupby(non_trading['datetime'].apply(lambda x: map_trading_date(x)))
    non_trading_day_news = non_trading_group.count()['datetime'].rename('close')

    day_news = pd.concat([trading_day_news, non_trading_day_news], axis=1, sort=True).dropna()
    day_news.index = pd.to_datetime(day_news.index)

    rcParams.update({'figure.autolayout': True})
    plt.cla()
    ax = day_news.plot(kind='line', title=title, legend=True)
    ax.set(xlabel='Date', ylabel='Frequency')
    plt.savefig(fname)

def plot_webhose():
    df = pd.read_csv('webhose_data.csv')

    plot_freq_hist(
        data=df['site'].value_counts()[:20],
        title='Top 20 Webhose.io source site',
        xlabel='Site name',
        fname='picture/webhose_site.png'
    )

    plot_freq_hist(
        data=df['country'].value_counts(),
        title='Webhose.io source country',
        xlabel='Country',
        fname='picture/webhose_country.png'
    )

    plot_news_per_day(df, 'News per day', 'picture/webhose_news_per_day.png')

def plot_webhose_500():
    df = pd.read_csv('webhose_label.csv')

    plot_freq_hist(
        data=df['site'].value_counts()[:20],
        title='Top 20 Webhose 500 source site',
        xlabel='Site name',
        fname='picture/webhose_500_site.png'
    )

    plot_freq_hist(
        data=df['country'].value_counts(),
        title='Webhose 500 source country',
        xlabel='Country',
        fname='picture/webhose_500_country.png'
    )

    plot_freq_hist(
        data=df['ticker'].value_counts()[:20],
        title='Top 20 Webhose 500 article ticker',
        xlabel='Ticker',
        fname='picture/webhose_500_ticker.png'
    )

    plot_news_per_day(df, 'News per day', 'picture/webhose_500_news_per_day.png')

def plot_reuters():
    df = pd.read_csv('data/reuter_data.csv')

    plot_freq_hist(
        data=df['ticker'].value_counts()[:20],
        title='Top 20 Reuters article ticker',
        xlabel='Ticker',
        fname='picture/reuter_ticker.png'
    )

    plot_news_per_day(df, 'News per day', 'picture/reuter_news_per_day.png')


if __name__ == '__main__':
    plot_webhose_500()

