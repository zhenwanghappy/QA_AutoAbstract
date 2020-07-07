# -*- coding:utf-8 -*-
#author: ASUS
#date  : 2020/4/23 23:17
#FILE  : dataclean.py
#IDE   : PyCharm

# -*- coding:utf-8 -*-
#author: ASUS
#date  : 2020/4/23 21:17
#FILE  : dropwords.py
#IDE   : PyCharm
import numpy as np
import pandas as pd
import src.config as config
import jieba
from jieba import posseg

"""
    清洗数据：去除停用词，去除空行
"""

def cut(sentence, cut_type = 'word',  pos = False):
    if pos:
        if cut_type == 'word':
            words_seq, pos_seq = [], []
            for w, p in posseg.cut(sentence):
                words_seq.append(w)
                pos_seq.append(p)
            return words_seq, posseg
        else:
            words_seq = list(sentence)
            pos_seq = []
            for w in words_seq:
                pos_seq.append(posseg.lcut(w)[0].flag)
            return words_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        else:
            return list(sentence)

def parse_clean(train_path, test_path):
    '''
    去除无用的数据
    :param train_path:
    :param test_path:
    :return:
    '''
    train_set = pd.read_csv(train_path, encoding="utf-8")
    print(train_set.info())
    train_set.dropna(subset=['Report'], how='any', inplace=True)
    train_set.fillna('', inplace=True)
    train_x = train_set.Question.str.cat(train_set.Dialogue)
    train_y = train_set.Report
    assert len(train_x) == len(train_y)

    test_set = pd.read_csv(test_path, encoding="utf-8")
    test_set.fillna('', inplace=True)
    test_x = test_set.Question.str.cat(test_set.Dialogue)
    test_y = []
    return train_x, train_y, test_x, test_y

def read_stopwords(path):
    '''
    加载停用词
    :param path:
    :return:
    '''
    lines = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines

def filter_data(train_x, train_y, test_x, test_y):
    """
    去除无用词
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """
    remove_words = ['|', '[', ']', '语音', '图片']
    stop_words = read_stopwords(config.stopwords_path)
    sentences = []
    with open(config.train_x_path, 'w', encoding='utf-8') as f1:
        i = 0
        for line in train_x:
            word_list = jieba.cut(line.strip())
            word_list = [word for word in word_list if word not in stop_words and word != ' ']
            word_list = [word for word in word_list if word not in remove_words]
            if len(word_list) == 0:
                i += 1
            sentences.append(word_list)
            f1.write(' '.join(word_list)+"\n")
    with open(config.train_y_path, 'w', encoding='utf-8') as f2:
        j = 0
        for line in train_y:
            word_list = jieba.cut(line.strip())
            word_list = [word for word in word_list if word not in stop_words and word != ' ']
            word_list = [word for word in word_list if word not in remove_words]
            if len(word_list) == 0:
                word_list = ['随时', '联系']
            sentences.append(word_list)
            f2.write(' '.join(word_list)+"\n")
    with open(config.test_x_path, 'w', encoding='utf-8') as f3:
        for line in test_x:
            word_list = jieba.cut(line.strip())
            word_list = [word for word in word_list if word not in stop_words and word != ' ']
            word_list = [word for word in word_list if word not in remove_words]
            sentences.append(word_list)
            f3.write(' '.join(word_list)+"\n")
    print(i, j)
    assert i == j
    return sentences

def save_sentences(sentences, sentences_path):
    with open(sentences_path, 'w',encoding='utf-8') as f:
        for line in sentences:
            f.write(' '.join(line)+"\n")

# def generate_data():
#     """
#     将预处理好的文本书籍转换成索引数据
#     :return:
#     """
#     index2word = {}
#     for line in open(config.index2word_path, encoding='utf-8'):
#         (key, value) = line.strip().split()
#         index2word[int(key)] = value
#
#     word2index = {}
#     for line in open(config.word2index_path, encoding='utf-8'):
#         (key, value) = line.strip().split()
#         word2index[key] = int(value)
#
#     with open(config.train_x_path, 'r', encoding='utf-8') as f1:
#         data = [[word2index[word] for word in sentence.strip().split()] for sentence in f1]
#     with open(config.train_x_index_path, 'w', encoding='utf-8') as f1:
#         f1.write(str(data))
#     with open(config.train_y_path, 'r', encoding='utf-8') as f2:
#         data = [[word2index[word] for word in sentence.strip().split()] for sentence in f2]
#     with open(config.train_y_index_path, 'w', encoding='utf-8') as f2:
#         f2.write(str(data))
#     with open(config.test_x_path, 'r', encoding='utf-8') as f3:
#         data = [[word2index[word] for word in sentence.strip().split()] for sentence in f3]
#     with open(config.test_x_index_path, 'w', encoding='utf-8') as f3:
#         f3.write(str(data))

def generate_sentences_for_word2vec(trainset_path, testset_path, sentences_path):
    """
    生成数据
    :return:
    """
    train_x, train_y, test_x, test_y = parse_clean(trainset_path, testset_path)
    sentences = filter_data(train_x, train_y, test_x, test_y)
    save_sentences(sentences, sentences_path)

if __name__ == '__main__':
    generate_sentences_for_word2vec()
