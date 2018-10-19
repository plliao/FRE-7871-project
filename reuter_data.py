import json
import datetime
import re
import sys
import ipdb

import pandas as pd

def logging(message):
    sys.stderr.write('\r')
    sys.stderr.write(message)
    sys.stderr.flush()

def clean_company_name(name):
    company_token = [
        '^\"', '\"$', 'Inc\W', 'Inc$', 'Co\W', 'Co$',
        'Corp\W', 'Corp$', 'Technology', ',.*$',
        '\.com', 'Company', 'ltd\W',
        'Class A', 'Class C', 'Corporation', '& Co\.'
    ]
    company_pattern = '|'.join(company_token)
    return re.sub(company_pattern, '', name).strip()

def generate_news_label():
    snp = pd.read_csv('data/constituents.csv')

    for name in snp['Name']:
        sys.stdout.write(clean_company_name(name) + '\n')
        sys.stdout.flush()

    df = pd.DataFrame()
    date = datetime.datetime.strptime('20180702', '%Y%m%d')
    end_date = datetime.datetime.strptime('20180930', '%Y%m%d')
    one_day = datetime.timedelta(days=1)
    count = 0
    while date <= end_date:
        date_str = date.strftime('%Y%m%d')
        file_path = 'data/reuters/news_{0:s}.json'.format(date_str)

        with open(file_path, 'r') as f:
            data = json.load(f)

        title_map = {}
        total_articles = len(data)
        for index, item in enumerate(data):
            count += 1

            logging('{0:s}: {1:d}/{2:d}'.format(date_str, index + 1, total_articles))

            published_time = item['published_time']
            title = item['title']
            text = item['text']

            cleaned_title = re.sub(r'[^\w\s]', '', title)
            cleaned_title = re.sub(r' +', ' ', cleaned_title)
            if cleaned_title in title_map:
                continue
            title_map[cleaned_title] = 1

            for ticker, name in zip(snp['Symbol'], snp['Name']):

                ticker_pattern = '\W{0:s}\W'.format(ticker)

                cleaned_name = clean_company_name(name)
                name_pattern = '{0:s}\W'.format(cleaned_name)

                add_datum = False
                '''
                if len(ticker) != 1 and ticker != 'CA':
                    if re.search(ticker_pattern, title) is not None or \
                            re.search(ticker_pattern, text) is not None:
                        add_datum = True
                '''
                if cleaned_name != 'CA':
                    if '*' in cleaned_name:
                        name_pattern = re.sub('\\*', '\\\\*', name_pattern)
                    if '.' in cleaned_name:
                        name_pattern = re.sub('\\.', '\\\\.', name_pattern)

                    if re.search(name_pattern, title) is not None or \
                            re.search(name_pattern, title) is not None:
                        add_datum = True

                if (ticker == 'GOOGL' or ticker == 'GOOG') and 'Google' in text:
                    add_datum = True

                if add_datum:
                    sys.stdout.write('{0:s}: {1:s}, {2:s}, {3:s}\n'.format(date_str, cleaned_title, ticker, cleaned_name))
                    sys.stdout.write('\t{0:s}\n'.format(text))
                    sys.stdout.flush()

                    datum = pd.Series(
                        data={
                            'published_time':published_time,
                            'title':title,
                            'text':text,
                            'ticker':ticker,
                            'name':name
                        }
                    )
                    df = df.append(datum, ignore_index=True)
        date = date + one_day
    print('\nnews articles: ', count)
    return df


def label_webhose_data():
    webhose = pd.read_csv('webhose_data.csv')
    webhose['published_time'] = pd.to_datetime(webhose['published_time'])
    webhose.sort_values('published_time', inplace=True)

    snp = pd.read_csv('data/constituents.csv')

    df = pd.DataFrame()
    total_articles = len(webhose)

    for index, item in webhose.iterrows():
        #logging('{0:d}/{1:d}'.format(index + 1, total_articles))

        published_time = pd.to_datetime(item['published_time'])
        title = item['title']
        text = re.sub('Google Plus', '', item['text'])
        text = re.sub('Google\+', '', text)
        date_str = published_time.strftime('%Y%m%d')
        country = item['country']
        site = item['site']

        cleaned_title = re.sub(r'[^\w\s]', '', title)
        cleaned_title = re.sub(r' +', ' ', cleaned_title)

        for ticker, name in zip(snp['Symbol'], snp['Name']):
            cleaned_name = clean_company_name(name)
            name_pattern = '{0:s}\W'.format(cleaned_name)

            add_datum = False
            if cleaned_name != 'CA' and cleaned_name != 'News':
                if '*' in cleaned_name:
                    name_pattern = re.sub('\\*', '\\\\*', name_pattern)
                if '.' in cleaned_name:
                    name_pattern = re.sub('\\.', '\\\\.', name_pattern)

                if re.search(name_pattern, title) is not None or \
                        re.search(name_pattern, title) is not None:
                    add_datum = True

            if cleaned_name == 'News':
                if name in text:
                    add_datum = True

            if (ticker == 'GOOGL' or ticker == 'GOOG') and 'Google' in text:
                add_datum = True

            if add_datum:
                sys.stdout.write('{0:s}: {1:s}, {2:s}, {3:s}\n'.format(date_str, cleaned_title, ticker, cleaned_name))
                sys.stdout.write('\t{0:s}\n'.format(re.sub('\n', '', text)))
                sys.stdout.flush()

                datum = pd.Series(
                    data={
                        'published_time':published_time,
                        'title':title,
                        'text':text,
                        'ticker':ticker,
                        'name':name,
                        'country':country,
                        'site':site
                    }
                )
                df = df.append(datum, ignore_index=True)
    return df


if __name__ == '__main__':
    df = label_webhose_data()
    df.to_csv('webhose_label.csv', index=False)
