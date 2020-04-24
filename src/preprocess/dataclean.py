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


def parse_clean(train_path, test_path):
    train_set = pd.read_csv(train_path, encoding="utf-8")
    print(train_set.info())
    train_set.dropna(subset=['Report'], how='any', inplace=True)
    train_set.fillna('', inplace=True)
    train_x = train_set.Question.str.cat(train_set.Dialogue)
    train_y = train_set.Report
    assert len(train_x) == len(train_y)

    test_set = pd.read_csv(test_path, encoding="utf-8")
    test_set.fillna('',inplace=True)
    test_x = test_set.Question.str.cat(test_set.Dialogue)
    test_y = []
    return train_x, train_y, test_x, test_y

def read_stopwords(path):
    lines = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines

def filter_data(train_x, train_y, test_x):
    remove_words = ['|', '[', ']', '语音', '图片']
    stop_words = read_stopwords(config.stopwords_path)
    with open(config.train_x_path, 'w', encoding='utf-8') as f1:
        for line in train_x:
            word_list = jieba.cut(line)
            word_list = [word for word in word_list if word not in stop_words]
            word_list = [word for word in word_list if word not in remove_words]
            f1.write(' '.join(word_list)+"\n")
    with open(config.train_y_path, 'w', encoding='utf-8') as f2:
        for line in train_x:
            word_list = jieba.cut(line)
            word_list = [word for word in word_list if word not in stop_words]
            word_list = [word for word in word_list if word not in remove_words]
            f2.write(' '.join(word_list)+"\n")
    with open(config.test_x_path, 'w', encoding='utf-8') as f3:
        for line in train_x:
            word_list = jieba.cut(line)
            word_list = [word for word in word_list if word not in stop_words]
            word_list = [word for word in word_list if word not in remove_words]
            f3.write(' '.join(word_list)+"\n")


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = parse_clean(config.trainset_path, config.testset_path)
    filter_data(train_x, train_y, test_x)