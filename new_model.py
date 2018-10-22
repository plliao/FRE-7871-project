import re
import string
import ipdb

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
from wordcloud import WordCloud

MIN_DF = 10
MAX_DF = 100
WORD_CLOUD_NUMBER = 50

def clean_sentence(s):
    s = re.sub("\n", " ", s)
    s = re.sub("[" + string.punctuation + "]", " ", s)
    s = re.sub("[0-9]+", " ", s)
    s = re.sub(" +", " ", s)
    return s.strip()

def generate_bag_of_words(news_df, start_index, split_index, end_index, stopwords):
    all_data = news_df[start_index:end_index]
    train = news_df[start_index:split_index]
    test = news_df[split_index:end_index]

    vectorizer = CountVectorizer(stop_words=stopwords, min_df=MIN_DF, max_df=MAX_DF)
    train_bag_of_words = vectorizer.fit_transform(train['text'].apply(clean_sentence)).toarray()
    test_bag_of_words = vectorizer.transform(test['text'].apply(clean_sentence)).toarray()

    word_list = vectorizer.get_feature_names()
    print("word size", len(word_list))
    #print(word_list)

    train_text_df = pd.DataFrame(train_bag_of_words, index=train.index, columns=word_list)
    test_text_df = pd.DataFrame(test_bag_of_words, index=test.index, columns=word_list)
    bag_of_words_df = pd.concat([train_text_df, test_text_df], axis=0)
    return bag_of_words_df, vectorizer

def generate_price_features(data):
    price_feature_name = ['previous_price_{0:d}'.format(d) for d in range(1, 6)]
    price_features = data[price_feature_name].values
    return price_features

def generate_classification_label(data):
    y = np.zeros(data.shape[0], np.float)
    y[data['predicted_price'] > data['price']] = 1.0
    return y

def generate_regression_label(data):
    return (data['predicted_price'] - data['price']).values

def evaluate_return(open_price, y_hat, y):
    revenue = 0
    index = 0
    buy_action = []

    for price, predict, actual in zip(open_price, y_hat, y):
        if predict >= 0.0 * price:
            revenue += actual
            buy_action.append(index)
        index += 1

    return revenue, buy_action

def run(data, split, stopwords, exp_label):
    published_time = pd.to_datetime(data['published_time'])
    y = generate_regression_label(data)
    y_class = generate_classification_label(data)
    X_price = data['price'].values


    record = {
        'classification':{
            'train':pd.DataFrame(),
            'test':pd.DataFrame()
        },
        'regression':{
            'train':pd.DataFrame(),
            'test':pd.DataFrame()
        },
        'pnl':{
            'train':pd.DataFrame(),
            'test':pd.DataFrame()
        },
        'buy_actions':{
        }
    }

    fold_index = 0
    tscv = TimeSeriesSplit(n_splits=split)
    for train_index, test_index in tscv.split(data.values):
        fold_index += 1

        start_index = data.index[train_index[0]]
        split_index = data.index[test_index[0]]
        end_index = data.index[test_index[-1]] + 1

        bag_of_words, vectorizer = generate_bag_of_words(data, start_index, split_index, end_index, stopwords)
        X = bag_of_words.values

        word_size = X.shape[1]
        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]
        y_class_train, y_class_test = y_class[train_index], y_class[test_index]

        X_train_price = X_price[train_index]
        X_test_price = X_price[test_index]

        # Normalization and Scaling
        scaler = RobustScaler()
        scaler.fit(y_train.reshape(-1, 1))
        y_train_t = scaler.transform(y_train.reshape(-1, 1)).reshape(-1, )

        x_train_t = normalize(X_train)
        x_test_t = normalize(X_test)

        # Modeling
        classifiers_dict = {
            'Logistic Regression':LogisticRegression(penalty='l2', C=0.05, verbose=0, max_iter=10000)
        }

        regressors_dict = {
            'SVR':SVR(kernel='linear', C=1.0, verbose=0),
            'Ridge Regression':linear_model.Ridge(alpha=5.0)
        }

        train_class_err = {}
        test_class_err = {}
        train_regre_err = {}
        test_regre_err = {}
        train_pnl_err = {}
        test_pnl_err = {}
        test_buy_times = []

        for label, clf in classifiers_dict.items():
            clf.fit(x_train_t, y_class_train)

            y_class_train_pred = clf.predict(x_train_t)
            y_class_test_pred = clf.predict(x_test_t)

            # classification error
            train_acc = accuracy_score(y_class_train, y_class_train_pred)
            test_acc = accuracy_score(y_class_test, y_class_test_pred)
            train_class_err[label] = train_acc
            test_class_err[label] = test_acc

            # PNL error
            train_return, train_buy_action = evaluate_return(X_train_price, y_class_train_pred, y_train)
            test_return, test_buy_action = evaluate_return(X_test_price, y_class_test_pred, y_test)
            train_pnl_err[label] = train_return
            test_pnl_err[label] = test_return

            if label not in record['buy_actions']:
                record['buy_actions'][label] = []
            for action_time in test_buy_action:
                record['buy_actions'][label].append(action_time + len(X_train))

        for label, clf in regressors_dict.items():
            clf.fit(x_train_t, y_train_t)

            y_train_pred = clf.predict(x_train_t)
            y_test_pred = clf.predict(x_test_t)

            # classification error
            y_class_train_pred = np.zeros(y_train_pred.shape[0], np.float)
            y_class_train_pred[y_train_pred >= 0.0] = 1.0
            y_class_test_pred = np.zeros(y_test_pred.shape[0], np.float)
            y_class_test_pred[y_test_pred >= 0.0] = 1.0

            train_acc = accuracy_score(y_class_train, y_class_train_pred)
            test_acc = accuracy_score(y_class_test, y_class_test_pred)
            train_class_err[label] = train_acc
            test_class_err[label] = test_acc

            # regression error
            y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).reshape(-1, )
            y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(-1, )
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)

            train_regre_err[label] = train_mse
            test_regre_err[label] = test_mse

            # PNL error
            train_return, train_buy_action = evaluate_return(X_train_price, y_train_pred, y_train)
            test_return, test_buy_action = evaluate_return(X_test_price, y_test_pred, y_test)
            train_pnl_err[label] = train_return
            test_pnl_err[label] = test_return

            if label not in record['buy_actions']:
                record['buy_actions'][label] = []
            for action_time in test_buy_action:
                record['buy_actions'][label].append(action_time + len(X_train))

        record['classification']['train'] = record['classification']['train'].append(pd.Series(data=train_class_err), ignore_index=True)
        record['classification']['test'] = record['classification']['test'].append(pd.Series(data=test_class_err), ignore_index=True)
        record['regression']['train'] = record['regression']['train'].append(pd.Series(data=train_regre_err), ignore_index=True)
        record['regression']['test'] = record['regression']['test'].append(pd.Series(data=test_regre_err), ignore_index=True)
        record['pnl']['train'] = record['pnl']['train'].append(pd.Series(data=train_pnl_err), ignore_index=True)
        record['pnl']['test'] = record['pnl']['test'].append(pd.Series(data=test_pnl_err), ignore_index=True)

        # Words analysis
        if fold_index == split:
            plot_word_coef_in_model_dict(classifiers_dict, vectorizer, exp_label)
            plot_word_coef_in_model_dict(regressors_dict, vectorizer, exp_label)

            bayes_result = analysis_bay(X_train, y_class_train, ['negative', 'positive'], vectorizer)
            plot_word_analysis_result(bayes_result, 'bayes', exp_label)
    return record

def plot_word_coef_in_model_dict(model_dict, vectorizer, exp_label):
    for clf_name, clf in model_dict.items():
        coef = clf.coef_
        if len(coef) == 1:
            coef = coef[0]

        result = analysis(coef, vectorizer)
        plot_word_analysis_result(result, clf_name, exp_label)

def plot_word_analysis_result(result, model, exp_label):
    model_name = re.sub(' +', '_', model.lower())
    exp_label = re.sub(' +', '_', exp_label.lower())
    for class_name, freq in result.items():
        label = '{0:s}_{1:s}_{2:s}'.format(model_name, class_name, exp_label)
        plot_word_cloud(label, freq)

def plot_word_cloud(label, freq):
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=freq)
    plt.clf()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    path = "picture/wordCloud/{0:s}.png".format(label)
    plt.savefig(path)

def analysis_bay(X, y, class_labels, vectorizer):
    clf = MultinomialNB()
    clf.fit(X, y)

    result = {}
    for class_index, class_prob_array in enumerate(clf.feature_log_prob_):
        result[class_labels[class_index]] = {}
        maximum_prob_index = class_prob_array.argsort()[-WORD_CLOUD_NUMBER:][::-1]

        for index in maximum_prob_index:
            word = vectorizer.get_feature_names()[index]
            result[class_labels[class_index]][word] = class_prob_array[index]
    return result

def analysis(clf_coef, vectorizer):
    result = {
        'positive':{},
        'negative':{}
    }
    maximum_weight_index = clf_coef.argsort()[-WORD_CLOUD_NUMBER:][::-1]
    minimum_weight_index = clf_coef.argsort()[:WORD_CLOUD_NUMBER]

    for positive_index, negative_index in zip(maximum_weight_index, minimum_weight_index):
        positive_weight = clf_coef[positive_index]
        negative_weight = clf_coef[negative_index]

        if positive_weight > 0:
            positive_term = vectorizer.get_feature_names()[positive_index]
            result['positive'][positive_term] = positive_weight

        if negative_weight < 0:
            negative_term = vectorizer.get_feature_names()[negative_index]
            result['negative'][negative_term] = -negative_weight
    return result

def plot_record(record, label):
    measure_map = {
        'classification':'Accuracy',
        'regression':'MSE',
        'pnl':'Dollar'
    }

    rcParams.update({'figure.autolayout': True})
    for task, item in record.items():
        if task != 'buy_actions':
            for sample, dataframe in item.items():
                label_filename = re.sub(' +', '_', label)
                title = '{0:s} task on {1:s} dataset {2:s}'.format(task, sample, label)
                path = 'picture/experiment/{0:s}_{1:s}_{2:s}.png'.format(task, sample, label_filename)
                plt.cla()
                ax = dataframe.plot(kind='line', style='.-', xticks=range(len(dataframe)), title=title, legend=True)
                ax.set(xlabel='Fold number', ylabel=measure_map[task])
                plt.savefig(path)


def main():
    news_df = pd.read_csv('webhose_price_trend.csv')
    data_label = '(webhose)'
    #news_df = pd.read_csv('data/reuter_price.csv')
    #data_label = '(reuter)'

    null_text_index = news_df[news_df['text'].isnull()].index
    news_df.drop(null_text_index, inplace=True)
    news_df['published_time'] = pd.to_datetime(news_df['published_time'])
    news_df.sort_values('published_time', inplace=True)
    news_df = news_df.reset_index()
    #print("Price feature")
    #run(news_df, snp_df, 5, None, False)

    num_split = 10
    exp_label = "with stopwords {0:s}".format(data_label)
    print(exp_label)
    record = run(news_df, num_split, None, exp_label)
    plot_record(record, exp_label)

    exp_label = "without stopwords {0:s}".format(data_label)
    print(exp_label)
    record = run(news_df, num_split, "english", exp_label)
    plot_record(record, exp_label)


if __name__ == '__main__':
    main()
