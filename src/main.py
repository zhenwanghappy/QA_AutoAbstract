from src.preprocess import dataclean
from src.preprocess import index_word
from src.preprocess import word2vec_build
from src import config
from src.util import build_embbedingmatrix
from src.util import batch_utils
from src.model import Seq2Seq
from src.util import train_helper
from src.util import test_helper
from src import testing
from src import training

def preprocess():
    print(config.trainset_path)
    dataclean.generate_sentences_for_word2vec(config.trainset_path, config.testset_path, config.sentences_path)
    index_word.build_vocab(config.sentences_path, config.wordcounts_path, config.index2word_path, config.word2index_path)
    word2vec_build.build_word2vec(config.sentences_path, config.w2v_bin_path)
    build_embbedingmatrix.build_embedding_matrix(config.w2v_bin_path, config.word2index_path, config.embbeding_matrix_path)
    embbeding = build_embbedingmatrix.load_pkl(config.embbeding_matrix_path)

if __name__ == '__main__':
    preprocess()
    # if config.mode == "train":
    #     training.train()
    # elif config.mode == "test":
    #     testing.test_and_save()

