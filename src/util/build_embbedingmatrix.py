import src.config as config
import numpy as np
from src.util.batch_utils import Vocab
import os
import pickle
from src.preprocess import word2vec_build


def build_embedding_matrix(w2v_path, vocab_path, embedding_path):
    w2v = word2vec_build.load_word2vec(w2v_path)
    vocab = Vocab(vocab_path, config.vocab_size)
    embedding = {}
    for word, id in vocab.word2id.items():
        if word in w2v.vocab:
            embedding[id] = w2v.wv[word]
        else:
            embedding[id] = np.random.uniform(-0.025, 0.025, (w2v.vector_size))
    dump_pkl(embedding, embedding_path, True)
    # return embedding
    # np.savetxt(config.embbeding_matrix_path, matrix)

def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)

def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result

def load_embbedding(embbeding_matrix_path):
    embbeding_dict = load_pkl(config.embbeding_matrix_path)
    embedding_matrix = np.zeros((config.vocab_size, config.embed_dim))
    for i, v in embbeding_dict.items():
        embedding_matrix[i] = v
    return embedding_matrix


if __name__ == "__main__":
    # build_embedding_matrix(config.w2v_bin_path, config.word2index_path, config.embbeding_matrix_path)  #57084
    # embbeding = load_pkl(config.embbeding_matrix_path)
    # print(embbeding[0])
    embbeding = load_embbedding(config.embbeding_matrix_path)
    print(embbeding[0])


