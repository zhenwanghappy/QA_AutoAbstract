# -*- coding:utf-8 -*-
#author: ASUS
#date  : 2020/4/23 15:36
#FILE  : __init__.py.py
#IDE   : PyCharm
import os

parentpath = os.path.abspath(os.path.dirname(__file__))+"/../.."
trainset_path = parentpath+"/resources/dataset/AutoMaster_TrainSet.csv"
testset_path = parentpath+"/resources/dataset/AutoMaster_TestSet.csv"
test_results_path = parentpath+"/resources/dataset/test_results.csv"
results_path = parentpath+"/resources/dataset/results.csv"

train_x_path = parentpath+"/resources/dataset/train_x.txt"
train_y_path = parentpath+"/resources/dataset/train_y.txt"
test_x_path = parentpath+"/resources/dataset/test_x.txt"
stopwords_path = parentpath+"/resources/dataset/stop_words.txt"
sentences_path = parentpath+"/resources/dataset/sentences.txt"
w2v_bin_path = parentpath+"/resources/dataset/w2v.bin"
index2word_path = parentpath+"/resources/dataset/index2word.txt"
word2index_path = parentpath+"/resources/dataset/vocab.txt"
embbeding_matrix_path = parentpath+"/resources/dataset/embbeding_matrix.pkl"
wordcounts_path = parentpath+"/resources/dataset/wordcounts.txt"

min_x_path = parentpath+"/resources/dataset/min_x.txt"
min_y_path = parentpath+"/resources/dataset/min_y.txt"
min_t_x_path = parentpath+"/resources/dataset/min_t_x.txt"
min_t_x_csv_path = parentpath+"/resources/dataset/min_t_x.csv"

vocab_size = 30000
embed_dim = 256
enc_units = 64
dec_units = 64
batch_sz = 19
attn_units = 64
max_enc_len = 400
max_dec_len = 40
steps_per_epoch = 3
epochs = 50
learning_rate = 0.001
mode = "test"
model = "SequenceToSequence"
num_to_test = 10

seq2seq_model_dir = parentpath+"/ckpt/seq2seq"