# -*- coding:utf-8 -*-
#author: ASUS
#date  : 2020/4/23 23:38
#FILE  : word2vec_build.py
#IDE   : PyCharm
import src.config as config
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import gensim


def sentence_data():
    data = []
    with open(config.train_x_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            data.append(line)
    with open(config.train_y_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            data.append(line)
    with open(config.test_x_path, 'r', encoding='utf-8') as f3:
        for line in f3:
            data.append(line)
    with open(config.sentence_data_path, 'w', encoding='utf-8') as f4:
        for line in data:
            f4.write(line)


def build_word2vec():
    w2v = Word2Vec(sg=1, sentences=LineSentence(config.sentence_data_path),
                   size=256, window=5, min_count=100, iter=5)
    w2v.wv.save_word2vec_format(config.w2v_bin_path, binary=True)


if __name__ == "__main__":
    # sentence_data()
    # build_word2vec()
    w2v = gensim.models.KeyedVectors.load_word2vec_format(config.w2v_bin_path, binary=True)

    print(w2v.most_similar("技师"))

