import re
import string
import ipdb
import pickle

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from nltk import pos_tag, word_tokenize
import gensim.downloader as api

MIN_DF = 5
MAX_DF = 50
WORD_CLOUD_NUMBER = 50

BOW = "bow"
TFIDF = "tfidf"
WORD2VEC = "word2vec"
SKIPTHOUGHT = "skipThought"

def select_by_pos_tag(sentence, tags):
    word_tokens = word_tokenize(sentence)
    tagged_word_token = pos_tag(word_tokens)

    selected_words = [word for word, tag in tagged_word_token if tag in tags]
    return ' '.join(selected_words)

def clean_sentence(s):
    s = re.sub("\n", " ", s)
    s = re.sub("[" + string.punctuation + "]", " ", s)
    s = re.sub("[0-9]+", " ", s)
    s = re.sub(" +", " ", s)
    return s.strip()

def generate_bag_of_words(train, test, feature_args):
    vectorizer = CountVectorizer(min_df=MIN_DF, max_df=MAX_DF, **feature_args)
    train_bag_of_words = vectorizer.fit_transform(train['text'].apply(clean_sentence)).toarray()
    test_bag_of_words = vectorizer.transform(test['text'].apply(clean_sentence)).toarray()

    train_bag_of_words = normalize(train_bag_of_words)
    test_bag_of_words = normalize(test_bag_of_words)
    word_list = vectorizer.get_feature_names()

    train_text_df = pd.DataFrame(train_bag_of_words, index=train.index, columns=word_list)
    test_text_df = pd.DataFrame(test_bag_of_words, index=test.index, columns=word_list)
    bag_of_words_df = pd.concat([train_text_df, test_text_df], axis=0)
    return bag_of_words_df, vectorizer

def generate_tfidf(train, test, feature_args):
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, **feature_args)
    train_bag_of_words = vectorizer.fit_transform(train['text'].apply(clean_sentence)).toarray()
    test_bag_of_words = vectorizer.transform(test['text'].apply(clean_sentence)).toarray()

    word_list = vectorizer.get_feature_names()

    train_text_df = pd.DataFrame(train_bag_of_words, index=train.index, columns=word_list)
    test_text_df = pd.DataFrame(test_bag_of_words, index=test.index, columns=word_list)
    bag_of_words_df = pd.concat([train_text_df, test_text_df], axis=0)
    return bag_of_words_df, vectorizer

def average_word2vec(sentence, model):
    sentence = clean_sentence(sentence)
    word2vecs = []
    for word in sentence.split(" "):
        word = word.lower()
        if word in model:
            word2vecs.append(model[word])
    return pd.Series(np.average(word2vecs, axis=0))

def generate_word2vec(train, test, feature_args):
    model = feature_args['model']
    features = pd.concat([train, test], axis=0)['text'].apply(average_word2vec, model=model)
    return features, None

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

def run(data, split, feature_args, exp_label):
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
        },
        'words':{
        }
    }

    feature_list = [BOW, TFIDF, WORD2VEC, SKIPTHOUGHT]
    feature_functions = {
        BOW:generate_bag_of_words,
        TFIDF:generate_tfidf,
        WORD2VEC:generate_word2vec,
        SKIPTHOUGHT:None
    }

    fold_index = 0
    tscv = TimeSeriesSplit(n_splits=split)
    for train_index, test_index in tscv.split(data.values):
        fold_index += 1

        start_index = data.index[train_index[0]]
        split_index = data.index[test_index[0]]
        end_index = data.index[test_index[-1]] + 1
        train = data[start_index:split_index]
        test = data[split_index:end_index]

        X_list = []
        for feature_name in feature_list:
            if feature_name in feature_args:
                features, vectorizer = feature_functions[feature_name](train, test, feature_args[feature_name])
                X_list.append(features)

        if len(X_list) > 1:
            array_list = [features.values for features in X_list]
            X = np.concatenate(array_list, axis=1)
        else:
            X = X_list[0].values

        feature_size = X.shape[1]
        print("feature size:", feature_size)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_class_train, y_class_test = y_class[train_index], y_class[test_index]

        X_train_price = X_price[train_index]
        X_test_price = X_price[test_index]

        # Normalization and Scaling
        scaler = RobustScaler()
        scaler.fit(y_train.reshape(-1, 1))
        y_train_t = scaler.transform(y_train.reshape(-1, 1)).reshape(-1, )

        x_train_t = X_train
        x_test_t = X_test

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

        '''
        if str(fold_index) not in record['words']:
            record['words'][str(fold_index)] = []
        record['words'][str(fold_index)].append(len(vectorizer.get_feature_names()))

        # Words analysis
        if fold_index == split:
            plot_word_coef_in_model_dict(classifiers_dict, vectorizer, exp_label)
            plot_word_coef_in_model_dict(regressors_dict, vectorizer, exp_label)

            bayes_result = analysis_bay(X_train, y_class_train, ['negative', 'positive'], vectorizer)
            plot_word_analysis_result(bayes_result, 'bayes', exp_label)
        '''
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

def plot_record(record, label, selected_tasks):
    measure_map = {
        'classification':'Accuracy',
        'regression':'MSE',
        'pnl':'Dollar'
    }

    rcParams.update({'figure.autolayout': True})
    for task, item in record.items():
        if task in selected_tasks:
            for sample, dataframe in item.items():
                label_filename = re.sub(' +', '_', label)
                title = '{0:s} task on {1:s} dataset {2:s}'.format(task, sample, label)
                path = 'picture/experiment/{0:s}_{1:s}_{2:s}.png'.format(task, sample, label_filename)
                plt.cla()
                ax = dataframe.plot(kind='line', style='.-', xticks=range(len(dataframe)), title=title, legend=True)
                ax.set(xlabel='Fold number', ylabel=measure_map[task])
                plt.savefig(path)

def do_exp(data, num_split, feature_args, data_label, feature_label):
    if feature_args['stop_words'] is None:
        exp_label = "with stop_words ({0:s}, {1:s})".format(data_label, feature_label)
    else:
        exp_label = "without stop_words ({0:s}, {1:s})".format(data_label, feature_label)
    print(exp_label)
    print(feature_args)

    selected_tasks = ['classification', 'regression', 'pnl']

    file_name_label = re.sub(' +', '_', exp_label)
    record = run(data, num_split, feature_args, exp_label)
    #plot_record(record, exp_label, selected_tasks)
    #pickle.dump(record, open('result/record_{0:s}.p'.format(file_name_label), 'wb'))
    return record

def plot_records(records, model):
    task_list = ['classification', 'regression', 'pnl']
    merged_record = {}
    for task in task_list:
        train_record_list = []
        test_record_list = []

        for feature_label, record in records.items():
            if model in record[task]['train']:
                train_record_list.append(record[task]['train'][model].rename(feature_label))
                test_record_list.append(record[task]['test'][model].rename(feature_label))

        if len(train_record_list) > 0:
            merged_record[task] = {}
            merged_record[task]['train'] = pd.concat(train_record_list, axis=1)
            merged_record[task]['test'] = pd.concat(test_record_list, axis=1)

    model_file_name = re.sub(' +', '_', model.lower())
    plot_record(merged_record, 'comparison_{0:s}'.format(model_file_name), task_list)

def preprocess_news_df(news_df):
    null_text_index = news_df[news_df['text'].isnull()].index
    news_df.drop(null_text_index, inplace=True)
    news_df['published_time'] = pd.to_datetime(news_df['published_time'])
    news_df.sort_values('published_time', inplace=True)
    news_df = news_df.reset_index()
    return news_df

def main():
    #news_df = pd.read_csv('webhose_price_trend.csv')
    #data_label = 'webhose'
    news_df = pd.read_csv('data/reuter_price.csv')
    data_label = 'reuter'

    news_df = preprocess_news_df(news_df)
    raw_text = news_df['text'].copy()

    records = {}

    num_split = 5

    feature_label = "BOW stop_words"
    feature_param = {
        'stop_words':None,
        BOW:{}
    }
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    feature_label = "BOW"
    feature_param['stop_words'] = "english"
    feature_param[BOW]['stop_words'] = "english"
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)
    '''
    feature_label = "BOW tri-gram"
    feature_param[BOW]['ngram_range'] = (1, 3)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    tags = ['NNP', 'NNPS']
    feature_label = "BOW proper noun"
    del feature_param[BOW]['ngram_range']
    news_df['text'] = raw_text.apply(select_by_pos_tag, tags=tags)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    '''
    tags = ['NNP', 'NNPS', 'NN', 'NNS']
    feature_label = "BOW noun"
    news_df['text'] = raw_text.apply(select_by_pos_tag, tags=tags)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    '''
    tags = ['NNP', 'NNPS', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    feature_label = "BOW noun and verb"
    news_df['text'] = raw_text.apply(select_by_pos_tag, tags=tags)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    tags = ['NNP', 'NNPS', 'NN', 'NNS', 'JJ', 'JJR', 'JJS']
    feature_label = "BOW noun and adj"
    news_df['text'] = raw_text.apply(select_by_pos_tag, tags=tags)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    tags = ['NNP', 'NNPS', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
    feature_label = "BOW noun, verb and adj"
    news_df['text'] = raw_text.apply(select_by_pos_tag, tags=tags)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    feature_label = "BOW and TFIDF"
    feature_param[TFIDF] = {
        'stop_words':"english"
    }
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    tags = ['NNP', 'NNPS', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    feature_label = "BOW, TFIDF noun and verb"
    news_df['text'] = raw_text.apply(select_by_pos_tag, tags=tags)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)
    '''

    feature_label = "TFIDF"
    del feature_param[BOW]
    feature_param[TFIDF] = {
        'stop_words':"english"
    }
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    '''
    tags = ['NNP', 'NNPS', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    feature_label = "TFIDF noun and verb"
    news_df['text'] = raw_text.apply(select_by_pos_tag, tags=tags)
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)
    '''

    feature_label = "word2vec"
    del feature_param[TFIDF]
    feature_param[WORD2VEC] = {
        'model':api.load("word2vec-google-news-300")
    }
    records[feature_label] = do_exp(news_df, num_split, feature_param, data_label, feature_label)

    plot_records(records, 'Ridge Regression')
    #plot_records(records, 'Logistic Regression')
    #plot_records(records, 'SVR')

if __name__ == '__main__':
    main()
