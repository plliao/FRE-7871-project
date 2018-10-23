import ipdb

import numpy as np
import pandas as pd
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

from new_model import preprocess_news_df
from new_model import clean_sentence

def load_with_unidirectional_model():
    vocab_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
    embedding_matrix_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
    checkpoint_path = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"
    return load_model(vocab_file, embedding_matrix_file, checkpoint_path, False)

def load_with_bidirectional_model():
    vocab_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/vocab.txt"
    embedding_matrix_file = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/embeddings.npy"
    checkpoint_path = "/home/plliao/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"
    return load_model(vocab_file, embedding_matrix_file, checkpoint_path, True)

def load_model(vocab_file, embedding_matrix_file, checkpoint_path, bidirectional_encoder):
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(
        configuration.model_config(bidirectional_encoder=bidirectional_encoder),
        vocabulary_file=vocab_file,
        embedding_matrix_file=embedding_matrix_file,
        checkpoint_path=checkpoint_path
    )
    return encoder

def text_preprocessing(sentence):
    sentence = clean_sentence(sentence)
    word_tokens = sentence.split(" ")
    word_number = 100
    if len(word_tokens) < word_number:
        word_number = len(word_tokens)

    return ' '.join(word_tokens[:word_number])

def encode_news_df(file_path, label, encoder):
    news_df = preprocess_news_df(pd.read_csv(file_path))
    text = news_df['text'].apply(text_preprocessing)

    encoded_text = encoder.encode(text.tolist(), batch_size=256)
    encoded_text_df = pd.DataFrame(encoded_text, index=news_df.index)
    encoded_text_df.to_csv('skip_thoughts_embedding/{0:s}_embedding.csv'.format(label), index=False)


if __name__ == '__main__':
    #uni_encoder = load_with_unidirectional_model()
    #encode_news_df("data/reuter_price.csv", 'reuters_uni', uni_encoder)
    #encode_news_df("data/webhose_price_trend.csv", 'webhose_uni', uni_encoder)

    bi_encoder = load_with_bidirectional_model()
    encode_news_df("data/reuter_price.csv", 'reuters_bi', bi_encoder)
    encode_news_df("data/webhose_price_trend.csv", 'webhose_bi', bi_encoder)
