import tensorflow as tf
from src.attention.Encoder import Encoder
from src.attention.Decoder import Decoder
from src.attention.Decoder import BahdanauAttention
from src import config
from src.util import build_embbedingmatrix
from src.util import batch_utils
import numpy as np
from src.attention.Decoder import Pointer
from tensorflow import keras

class PGN(keras.Model):
    def __init__(self):
        super(PGN, self).__init__()
        self.embedding_matrix = build_embbedingmatrix.load_embbedding(config.embbeding_matrix_path)
        self.encoder = Encoder(vocab_size = config.vocab_size,
                               embedding_dim = config.embed_dim,
                               embedding_matrix = self.embedding_matrix,
                               enc_units = config.enc_units,
                               batch_sz = config.batch_sz)

        self.attention = BahdanauAttention(units=config.attn_units)

        self.decoder = Decoder(vocab_size =  config.vocab_size,
                               embedding_dim = config.embed_dim,
                               embedding_matrix = self.embedding_matrix,
                               dec_units = config.dec_units,
                               batch_sz = config.batch_sz)
        self.pointer = Pointer(config.vocab_size, config.embed_dim, self.embedding_matrix)

    def call(self, dec_input, dec_hidden, enc_output, dec_target, _enc_extended_inp, enc_mask, batch_oov_len):
        # dec_target.shape:[batch_sz, max_dec_len]
        # dec_hidden.shape:[batch_sz, enc_units}
        # enc_output.shape:[batch, enc_len, enc_units]
        # print(enc_output.shape, dec_target.shape)
        predictions = []
        attentions = []
        p_gens = []
        coverages = []
        prev_coverage = tf.zeros((enc_output.shape[0], enc_output.shape[1], 1))

        for t in range(0, dec_target.shape[1]):
            # pred.shape [batch_sz, vocab_size]
            coverages.append(prev_coverage)
            context_vector, attn, prev_coverage = self.attention(dec_hidden, enc_output, enc_mask, prev_coverage, True)
            pred, dec_hidden = self.decoder(dec_input,
                                            # dec_hidden,
                                            # enc_output,
                                            context_vector,
                                            True)
            p_gen = self.pointer(context_vector, dec_hidden, dec_input)
            predictions.append(pred)
            attentions.append(attn)
            p_gens.append(p_gen)
            # context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

        # tf.concat与tf.stack这两个函数作用类似，
        # 都是在某个维度上对矩阵(向量）进行拼接，
        # 不同点在于前者拼接后的矩阵维度不变，后者则会增加一个维度。
        predictions = tf.stack(predictions, axis=1)
        attentions = tf.stack(attentions, axis=1)
        p_gens = tf.stack(p_gens, axis=1)
        coverages = tf.stack(coverages, axis=1)
        coverages = tf.squeeze(coverages, -1)
        # tf.stack(predictions, 1).shape [batch_sz, max_dec_len-1, vocab_size]
        final_dist = self._calc_final_dist(_enc_extended_inp,
                                      predictions,
                                      attentions,
                                      p_gens,
                                      batch_oov_len,
                                      config.vocab_size,
                                      config.batch_sz)

        # final_dist (batch_size, dec_len, vocab_size+batch_oov_len)
        # (batch_size, dec_len, enc_len)
        return final_dist, [attentions, coverages]

    def _calc_final_dist(self, _enc_batch_extend_inp, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size,
                     batch_size):
        """
        :param _enc_batch_extend_vocab:   (batch_sz, enc_len)
        :param vocab_dists:               (batch_sz, dec_len, embbed_dim)
        :param attn_dists:                (batch_sz, dec_len, enc_len)
        :param p_gens:                    (batch_sz, dec_len, 1)
        :param batch_oov_len:             the maximum (over the batch) size of the extended vocabulary
        :param vocab_size:
        :param batch_size:
        :return:
        """
        # 先计算公式的左半部分
        # _vocab_dists_pgn (batch_size, dec_len, vocab_size)
        _vocab_dists_pgn = vocab_dists * p_gens
        # 根据oov表的长度补齐原词表
        # _extra_zeros (batch_size, dec_len, batch_oov_len)
        # if batch_oov_len != 0:
        _extra_zeros = tf.zeros((batch_size, p_gens.shape[1], batch_oov_len))
        # 拼接后公式的左半部分完成了
        # _vocab_dists_extended (batch_size, dec_len, vocab_size+batch_oov_len)
        _vocab_dists_extended = tf.concat([_vocab_dists_pgn, _extra_zeros], axis=-1)

        # 公式右半部分
        # _attn_dists_pgn (batch_size, dec_len, enc_len)
        _attn_dists_pgn = attn_dists * (1 - p_gens)
        # 拓展后的长度
        _extended_vocab_size = vocab_size + batch_oov_len

        # 要更新的数组 _attn_dists_pgn
        # 更新之后数组的形状与 公式左半部分一致
        # shape=[batch_size, dec_len, vocab_size+batch_oov_len]
        shape = _vocab_dists_extended.shape

        enc_len = tf.shape(_enc_batch_extend_inp)[1]
        dec_len = tf.shape(_vocab_dists_extended)[1]

        # batch_nums (batch_size, )
        batch_nums = tf.range(0, limit=batch_size)
        # batch_nums (batch_size, 1)
        batch_nums = tf.expand_dims(batch_nums, 1)
        # batch_nums (batch_size, 1, 1)
        batch_nums = tf.expand_dims(batch_nums, 2)

        # tile 在第1,2个维度上分别复制batch_nums dec_len,enc_len次
        # batch_nums (batch_size, dec_len, enc_len)
        batch_nums = tf.tile(batch_nums, [1, dec_len, enc_len])
        # (dec_len, )
        dec_len_nums = tf.range(0, limit=dec_len)
        # (1, dec_len)
        dec_len_nums = tf.expand_dims(dec_len_nums, 0)
        # (1, dec_len, 1)
        dec_len_nums = tf.expand_dims(dec_len_nums, 2)
        # tile是用来在不同维度上复制张量的
        # dec_len_nums (batch_size, dec_len, enc_len)
        dec_len_nums = tf.tile(dec_len_nums, [batch_size, 1, enc_len])
        # _enc_batch_extend_vocab_expand (batch_size, 1, enc_len)
        _enc_batch_extend_vocab_expand = tf.expand_dims(_enc_batch_extend_inp, 1)
        # _enc_batch_extend_vocab_expand (batch_size, dec_len, enc_len)
        _enc_batch_extend_vocab_expand = tf.tile(_enc_batch_extend_vocab_expand, [1, dec_len, 1])

        # 因为要scatter到一个3D tensor上，所以最后一维是3
        # indices (batch_size, dec_len, enc_len, 3)
        indices = tf.stack((batch_nums,
                            dec_len_nums,
                            _enc_batch_extend_vocab_expand),
                           axis=3)

        # attn_dists_projected （batch_size, enc_len, vocab_size+batch_oov_len）
        attn_dists_projected = tf.scatter_nd(indices, _attn_dists_pgn, shape)  # scatter_nd包括了指向同一个词的attention的累加
        # 至此完成了公式的右半边
        # 计算最终分布
        final_dists = _vocab_dists_extended + attn_dists_projected
        # print("final_dists", final_dists.shape)
        return final_dists


if __name__ == '__main__':
    _enc_batch_extend_vocab = tf.Variable([[3, 4, 5, 6, 8], [1, 9, 5, 6, 10]])

    vocab_dists = tf.Variable([[[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], ],
                               [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]
                               ])
    attn_dists = tf.Variable([[[0.2, 0.2, 0.2, 0.2, 0.2],
                               [0.2, 0.2, 0.2, 0.2, 0.2],
                               [0.2, 0.2, 0.2, 0.2, 0.2], ],
                              [[0.2, 0.2, 0.2, 0.2, 0.2],
                               [0.2, 0.2, 0.2, 0.2, 0.2],
                               [0.2, 0.2, 0.2, 0.2, 0.2]]])
    p_gens = tf.Variable([[[0.8], [0.6], [0.5]], [[0.7], [0.8], [0.6]]])
    batch_oov_len = 2
    model = PGN()
    final_dist = model._calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size=8,
                           batch_size=2)
    print(final_dist.shape)

