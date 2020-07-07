# -*- coding:utf-8 -*-
#author: ASUS
#date  : 2020/4/24 0:04
#FILE  : index_word.py
#IDE   : PyCharm
import src.config as config
from collections import defaultdict

def save_wordscounts(sentences_path, wordcounts_path):
    d = defaultdict(int)
    with open(sentences_path, "r", encoding='utf-8') as f:
        for line in f:
            for w in line.strip().split():
                d[w.lower()] += 1
    dic = sorted(d.items(), key=lambda x:x[1], reverse=True)
    with open(wordcounts_path, 'w', encoding='utf-8') as f:
        for w, c in dic:
            f.write(' '.join([str(w), str(c)]) + "\n")


def build_word2idx_and_idx2word(sentences_path, wordcounts_path, idx2word_path, word2idx_path, sort=True,  min_count=5):
    """过滤低频词，构建word2idx和idx2word"""
    save_wordscounts(sentences_path, wordcounts_path)
    wordcounts = {}
    with open(wordcounts_path, 'r', encoding='utf-8') as f:
        for line in f:
            wordcounts[line.split()[0]] = int(line.split()[1])
    words = []
    if sort:
        for k, v in wordcounts.items():
            key = k
            if min_count and min_count > v:
                continue
            words.append(key)
    else:
        with open(sentences_path, "r", encoding='utf-8') as f:
            for line in f:
                words.extend(line.strip().lower().split())
            words = set(words)
    idx2word = [(idx, w) for idx, w in enumerate(words)]
    word2idx = [(w, idx) for idx, w in enumerate(words)]
    save_word2idx_and_idx2word(idx2word, word2idx, idx2word_path, word2idx_path)
    return idx2word, word2idx

def save_word2idx_and_idx2word(idx2word, word2idx, idx2word_path,  word2idx_path):
    with open(idx2word_path, 'w', encoding='utf-8') as f:
        for x, y in idx2word:
            f.write(' '.join([str(x),str(y)])+"\n")
    with open(word2idx_path, 'w', encoding='utf-8') as f:
        for x, y in word2idx:
            f.write(' '.join([str(x), str(y)]) + "\n")

def build_vocab(sentences_path, wordcounts_path,idx2word_path, word2idx_path ):
    save_wordscounts(sentences_path, wordcounts_path)
    build_word2idx_and_idx2word(sentences_path, wordcounts_path,idx2word_path, word2idx_path)


if __name__=="__main__":
    build_vocab(config.sentences_path, config.wordcounts_path, config.index2word_path, config.word2index_path)