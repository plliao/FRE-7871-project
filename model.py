import re
import string
import ipdb

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from process import read_news_dataframe
from process import read_SNP_dataframe

NEWS_PER_DAY = 5

def clean_sentence(s):
    s = re.sub("\n", " ", s)
    s = re.sub("[" + string.punctuation + "]", " ", s)
    s = re.sub("[0-9]+", " ", s)
    s = re.sub(" +", " ", s)
    return s.strip()

def generate_bag_of_words(news_df, stopwords, min_df, max_df):
    vectorizer = CountVectorizer(stop_words=stopwords, min_df=min_df, max_df=max_df)
    bag_of_words = vectorizer.fit_transform(news_df['text'].apply(clean_sentence)).toarray()
    word_list = vectorizer.get_feature_names()
    print(len(word_list))
    print(word_list)

    words = pd.DataFrame(bag_of_words, columns=word_list)
    bow_df = pd.concat(
            [
                news_df.rename(columns={
                    'date': '_date_',
                    'text': '_text_'
                }),
                words
            ],
            axis=1
    )
    bow_df = bow_df.drop('_text_', axis=1).groupby('_date_').mean()
    return bow_df, vectorizer

def generate_price_features(data):
    price_feature_name = ['Open'] + ['previous_close_{0:d}'.format(d) for d in range(1, 2)]
    price_features = data[price_feature_name].values
    return price_features

def generate_classification_label(data):
    y = np.zeros(data.shape[0], np.float)
    y[data['Close'] > data['Open']] = 1.0
    return y

def generate_regression_label(data):
    return (data['Close'] - data['Open']).values

def evaluate_return(open_price, y_hat, y):
    invest_amount = 1000
    invest_times = 0
    revenue = 0
    index = 1
    for price, predict, actual in zip(open_price, y_hat, y):
        if predict >= 0.0001 * price:
            invest_times += 1
            print("Trade on {0:d}: ".format(index), price, predict, actual)
            revenue += (invest_amount * actual / price)
        index += 1

    if invest_times == 0:
        return 0
    else:
        return 100 * revenue / invest_amount / invest_times

def get_bow(news_df, start_date, split_date, end_date):
    news_date = news_df['date'].apply(pd.to_datetime)
    all_data = news_df[(news_date >= start_date) & (news_date <= end_date)]
    train = news_df[(news_date >= start_date) & (news_date < split_date)]
    test = news_df[(news_date >= split_date) & (news_date <= end_date)]

    vectorizer = CountVectorizer(stop_words="english", min_df=10, max_df=100)
    train_bag_of_words = vectorizer.fit_transform(train['text'].apply(clean_sentence)).toarray()
    test_bag_of_words = vectorizer.transform(test['text'].apply(clean_sentence)).toarray()

    word_list = vectorizer.get_feature_names()
    print("word size", len(word_list))
    #print(word_list)

    train_df = pd.DataFrame(train_bag_of_words, index=train.index, columns=word_list)
    test_df = pd.DataFrame(test_bag_of_words, index=test.index, columns=word_list)
    bag_of_words_df = pd.concat([train_df, test_df], axis=0)

    date_df = all_data.drop('text', axis=1).rename(columns={'date':'_date_'})
    return date_df.join(bag_of_words_df).groupby('_date_').sum(), vectorizer

def run(news_df, snp_df, price_index, split):
    train_result_list = []
    test_result_list = []
    train_result_list_bay = []
    test_result_list_bay = []

    data = news_df.groupby('date').sum().join(snp_df.set_index('Date')).dropna()
    X_temp = data.values
    y = generate_regression_label(data)

    tscv = TimeSeriesSplit(n_splits=split)
    for train_index, test_index in tscv.split(X_temp):
        print("index: ", train_index, test_index)
        start_date = data.index[train_index[0]]
        split_date = data.index[test_index[0]]
        end_date = data.index[test_index[-1]]

        bag_of_words, vectorizer = get_bow(news_df, start_date, split_date, end_date)

        X = data[['Open']].join(bag_of_words, how='inner').drop('Open', axis=1).values

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_price = data['Open'].values
        X_train_price = X_price[train_index]
        X_test_price = X_price[test_index]

        scaler = StandardScaler()
        scaler.fit(X_train_price.reshape(-1, 1))
        x_train_price_t = scaler.transform(X_train_price.reshape(-1, 1))
        x_test_price_t = scaler.transform(X_test_price.reshape(-1, 1))
        y_train_t = scaler.transform(y_train.reshape(-1, 1)).reshape(-1, )
        '''
        y_scaler = StandardScaler()
        y_scaler.fit(y_train.reshape(-1, 1))
        '''

        #train_open_price = X_train[:, price_index]
        #test_open_price = X_test[:, price_index]

        #x_train_t = x_scaler.transform(X_train)
        #x_test_t = x_scaler.transform(X_test)
        x_train_t = np.concatenate((normalize(X_train), x_train_price_t), axis=1)
        x_test_t = np.concatenate((normalize(X_test), x_test_price_t), axis=1)



        #ipdb.set_trace()
        #y_train_t = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1, )

        #clf = LogisticRegression(penalty='l2', C=0.00005, verbose=1, max_iter=100)
        clf = SVR(kernel='linear', C=0.0005, verbose=1)
        clf.fit(x_train_t, y_train_t)
        y_train_clf = clf.predict(x_train_t)
        y_test_clf = clf.predict(x_test_t)
        y_train_hat = scaler.inverse_transform(y_train_clf)
        y_test_hat = scaler.inverse_transform(y_test_clf)
        '''
        bayes = MultinomialNB()
        bayes.fit(X_train, y_train)

        y_train_bay = bayes.predict(X_train)
        y_test_bay = bayes.predict(X_test)
        '''

        train_mse = mean_squared_error(y_train, y_train_hat)
        test_mas = mean_squared_error(y_test, y_test_hat)
        print(train_mse, test_mas)
        train_return = evaluate_return(X_train_price, y_train_hat, y_train)
        test_return = evaluate_return(X_test_price, y_test_hat, y_test)
        print(train_return, test_return)

        '''
        train_acc = accuracy_score(y_train, y_train_clf)
        test_acc = accuracy_score(y_test, y_test_clf)
        train_result_list.append(train_acc)
        test_result_list.append(test_acc)
        '''

        '''
        train_acc_bay = accuracy_score(y_train, y_train_bay)
        test_acc_bay = accuracy_score(y_test, y_test_bay)
        train_result_list_bay.append(train_acc_bay)
        test_result_list_bay.append(test_acc_bay)
        '''

        #print(train_return, test_return)
        #print(train_acc, test_acc)
        #print('\t', train_acc_bay, test_acc_bay)

        important_terms, not_important_terms = analysis(x_train_t.shape[1], clf)
        print("Important terms: ", vectorizer.inverse_transform(important_terms[:-1].reshape(1, -1))[0])
        print("Not important terms: ", vectorizer.inverse_transform(not_important_terms[:-1].reshape(1, -1))[0])

        #A = analysis_bay(X_train, y_train)
        #show_terms(['pos', 'neg'], vectorizer, A[0], A[1])

    #print(np.average(train_result_list), np.average(test_result_list))
    #print('\t', np.average(train_result_list_bay), np.average(test_result_list_bay))
    #print(np.average(test_return_list))

    #clf = LogisticRegression(penalty='l2', C=0.005, verbose=1, max_iter=100)
    #clf.fit(X, y)

def analysis_bay(X, y):
    clf = MultinomialNB()
    clf.fit(X, y)

    important_terms = np.zeros(X.shape, np.int64)
    not_important_terms = np.zeros(X.shape, np.int64)
    for class_index, class_prob_array in enumerate(clf.feature_log_prob_):
        maximum_prob_index = class_prob_array.argsort()[-5:][::-1]
        minimum_prob_index = class_prob_array.argsort()[:5]
        important_terms[class_index][maximum_prob_index] = 1
        not_important_terms[class_index][minimum_prob_index] = 1
    return important_terms, not_important_terms

def show_terms(class_labels, vectorizer, important_terms, not_important_terms):
    print("Word Analysis:")
    for class_index, class_label in enumerate(class_labels):
        print("class: ", class_label)
        print("Important terms: ", vectorizer.inverse_transform(important_terms[class_index])[0])
        print("Not important terms: ", vectorizer.inverse_transform(not_important_terms[class_index])[0])

def analysis(feature_size, clf):
    important_terms = np.zeros(feature_size, np.int64)
    not_important_terms = np.zeros(feature_size, np.int64)
    maximum_weight_index = clf.coef_.argsort()[0][-20:][::-1]
    minimum_weight_index = clf.coef_.argsort()[0][:20]
    print(clf.coef_[0][maximum_weight_index])
    print(clf.coef_[0][minimum_weight_index])
    important_terms[maximum_weight_index] = 1
    not_important_terms[minimum_weight_index] = 1
    return important_terms, not_important_terms

def exp_with_price_features(snp_df):
    X = generate_price_features(snp_df)
    y = generate_label(snp_df)
    run(X, y, 10)

def exp_with_bag_of_words(news_df, snp_df, stopwords):
    #doc_number = news_df.shape[0]
    #bow_df, vectorizer = generate_bag_of_words(news_df, stopwords, 10, 100)

    #selected_features = ['Open', 'Date', 'Close'] + ['previous_close_{0:d}'.format(d) for d in range(1, 1)]
    #data = bow_df.join(snp_df[selected_features].set_index('Date')).dropna()

    #y = generate_label(data)
    #X = data.drop(['Open', 'Close'], axis=1).values
    run(news_df, snp_df, -1, 5)

def main():
    news_df = read_news_dataframe(NEWS_PER_DAY)
    snp_df = read_SNP_dataframe()

    #print("Price feature")
    #exp_with_price_features(snp_df.dropna())

    #print("With stopwords")
    #exp_with_bag_of_words(news_df, snp_df, None)

    print("Without stopwords")
    exp_with_bag_of_words(news_df, snp_df, "english")


if __name__ == '__main__':
    main()
