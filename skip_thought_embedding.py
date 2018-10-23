import ipdb

import numpy as np
import pandas as pd
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

from new_model import preprocess_news_df
from new_model import clean_sentence

def encode_with_unidirectional(data):
    vocab_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
    embedding_matrix_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
    checkpoint_path = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424.data-00000-of-00001"
    return encode(data, vocab_file, embedding_matrix_file, checkpoint_path)

def encode_with_bidirectional(data):
    vocab_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/vocab.txt"
    embedding_matrix_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/embeddings.npy"
    checkpoint_path = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008.data-00000-of-00001"
    return encode(data, vocab_file, embedding_matrix_file, checkpoint_path)

def encode(data, vocab_file, embedding_matrix_file, checkpoint_path):
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(
        configuration.model_config(),
        vocabulary_file=vocab_file,
        embedding_matrix_file=embedding_matrix_file,
        checkpoint_path=checkpoint_path
    )
    ipdb.set_trace()
    encodings = encoder.encode(data)
    return encodings

def encode_news_df(file_path, label):
    news_df = preprocess_news_df(pd.read_csv(file_path))
    text = news_df['text'].apply(clean_sentence)

    encoded_text_with_uni = encode_with_unidirectional(text)
    #encoded_text_with_bi = encode_with_bidirectional(text)

    encoded_text_with_uni_df = pd.DataFrame(encoded_text_with_uni, index=news_df.index)
    encoded_text_with_uni_df.to_csv('skip_thoughts_embedding/{0:s}_uni_embedding.csv'.format(label))

    #encoded_text_with_bi_df = pd.DataFrame(encoded_text_with_bi, index=news_df.index)
    #encoded_text_with_bi_df.to_csv('skip_thoughts_embedding/{0:s}_bi_embedding.csv'.format(label))

if __name__ == '__main__':
    encode_news_df("data/reuter_price.csv", 'reuters')
