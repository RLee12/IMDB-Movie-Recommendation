from collections import defaultdict

import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

import nltk


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def split_word(x):
    if pd.notnull(x):
        return nltk.word_tokenize(x)
    else:
        return "None"


user_table = pd.read_csv("df.csv", encoding='ISO-8859-1')
imdb_table = pd.read_csv("9066.csv")
imdb_table_sample = imdb_table.sample(n=100).reset_index(drop=True)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
w2v = {w: vec for w, vec in zip(model.index2word, model.syn0)}
pipeline = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), ("linear svr", SVR(kernel="linear"))])
user_table_splitted = user_table.groupby("userId")
user_all = list(user_table_splitted.grouper)
result_dict = []

for user in user_all:
    print(user)
    temp_table = user_table_splitted.get_group(user)
    merged_table = pd.merge(temp_table, imdb_table)
    merged_table.insert(merged_table.shape[1], "overview_splitted", "")
    merged_table['overview_splitted'] = merged_table['overview'].apply(split_word)
    model = pipeline.fit(merged_table['overview_splitted'], merged_table['rating'])
    predicted = model.predict(imdb_table_sample['overview'].apply(split_word))
    index = [imdb_table_sample.iloc[i[0]]['imdbId'] for i in sorted(enumerate(predicted), key=lambda x: x[1], reverse=True)][:10]
    result_dict.append({'userId': user, 'best_imdbId': index})

pd.DataFrame.from_dict(result_dict).to_csv("bestImdbId.csv", index=False, encoding='utf-8')