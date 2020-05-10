import src.config as config
from gensim.models import Word2Vec
import gensim
import numpy as np
import pandas as pd

def build_embedding_matrix(model_path):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(config.w2v_bin_path, binary=True)
    vocab = pd.read_csv(config.word2index_path, encoding="utf-8", delimiter=" ", header=None, index_col=1)
    vocab.index.astype("int")
    matrix = np.zeros((len(vocab), w2v.wv.vector_size))

    for i, row in vocab.iterrows():
        if row[0] in w2v.wv:
            matrix[i, :]=w2v[row[0]]
    np.savetxt(config.embbeding_matrix_path, matrix)


if __name__ == "__main__":
    build_embedding_matrix(config.w2v_bin_path)  #57084 技师
    # vocab = pd.read_csv(config.word2index_path, encoding="utf-8", delimiter=" ", header=None, index_col=1)
    # vocab.index.astype("int")
    # matrix = np.loadtxt(config.embbeding_matrix_path)
    # word = vocab.iloc[57084][0]
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(config.w2v_bin_path, binary=True)
    # print(w2v["技师"])
    # print(matrix[57084])


