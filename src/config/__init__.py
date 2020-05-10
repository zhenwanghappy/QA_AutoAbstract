# -*- coding:utf-8 -*-
#author: ASUS
#date  : 2020/4/23 15:36
#FILE  : __init__.py.py
#IDE   : PyCharm
import os

parentpath = os.path.abspath("../../.")
trainset_path = parentpath+"/resources/dataset/AutoMaster_TrainSet.csv"
testset_path = parentpath+"/resources/dataset/AutoMaster_TestSet.csv"
train_x_path = parentpath+"/resources/dataset/train_x.csv"
train_y_path = parentpath+"/resources/dataset/train_y.csv"
test_x_path = parentpath+"/resources/dataset/test_x.csv"
stopwords_path = parentpath+"/resources/dataset/stop_words.txt"
sentence_data_path = parentpath+"/resources/dataset/sentence_data.txt"
w2v_bin_path = parentpath+"/resources/dataset/w2v.bin"
index2word_path = parentpath+"/resources/dataset/index2word.txt"
word2index_path = parentpath+"/resources/dataset/vocab.txt"
embbeding_matrix_path = parentpath+"/resources/dataset/embbeding_matrix.txt"
