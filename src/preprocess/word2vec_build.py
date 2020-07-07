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
import numpy as np
import pandas as pd


def build_word2vec(sentences_path, w2v_bin_path):
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentences_path),
                   size=config.embed_dim, window=5, min_count=5, iter=10)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)

def load_word2vec(w2v_bin_path):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    return w2v


if __name__ == "__main__":
    build_word2vec(config.sentences_path, config.w2v_bin_path)
    w2v = load_word2vec(config.w2v_bin_path)
    print(len(w2v.wv.vocab))
    print(w2v.most_similar("美女"))

