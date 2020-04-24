# -*- coding:utf-8 -*-
#author: ASUS
#date  : 2020/4/24 0:04
#FILE  : index_word.py
#IDE   : PyCharm
import src.config as config

def build_index2word():
    words = []
    with open(config.sentence_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            words.extend(line.split())
    words = set(words)
    index2word = [(str(x), y) for x,y in enumerate(words)]
    word2index = [(y, str(x)) for x, y in enumerate(words)]
    with open(config.index2word_path, "w", encoding='utf-8') as f2:
        for line in index2word:
            f2.write(' '.join(line)+'\n')
    with open(config.word2index_path, "w", encoding='utf-8') as f3:
        for line in word2index:
            f3.write(' '.join(line)+'\n')


if __name__=="__main__":
    build_index2word()