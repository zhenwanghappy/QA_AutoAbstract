from tensorflow import keras
import tensorflow as tf
from src.attention.Encoder import Encoder
from src.attention.Decoder import Decoder
from src.attention.Decoder import BahdanauAttention
from src import config
from src.util import build_embbedingmatrix
from src.util import batch_utils
from src.model import losses
import numpy as np
np.set_printoptions(threshold=np.inf)

class Seq2Seq(keras.Model):
    def __init__(self):
        super(Seq2Seq, self).__init__()
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

    # def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
    #     # context_vector ()
    #     # attention_weights ()
    #     context_vector, attention_weights = self.attention(dec_hidden, enc_output)
    #
    #     # pred ()
    #     pred, dec_hidden = self.decoder(dec_input,
    #                                     None,
    #                                     None,
    #                                     context_vector)
    #     return pred, dec_hidden, context_vector, attention_weights

    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        # dec_target.shape:[batch_sz, max_dec_len]
        # dec_hidden.shape:[batch_sz, enc_units}
        # enc_output.shape:[batch, enc_len, enc_units]
        # print(enc_output.shape, dec_target.shape)
        predictions = []
        attentions = []

        for t in range(0, dec_target.shape[1]):
            # pred.shape [batch_sz, vocab_size]
            context_vector, attn = self.attention(dec_hidden, enc_output)
            attentions.append(attn)
            pred, dec_hidden = self.decoder(dec_input,
                                            # dec_hidden,
                                            # enc_output,
                                            context_vector)
            predictions.append(pred)
            # context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

            #tf.concat与tf.stack这两个函数作用类似，
            # 都是在某个维度上对矩阵(向量）进行拼接，
            # 不同点在于前者拼接后的矩阵维度不变，后者则会增加一个维度。
            # tf.stack(predictions, 1).shape [batch_sz, max_dec_len-1, vocab_size]
        return tf.stack(predictions, 1), [dec_hidden]


# def train_step(enc_inp, dec_tar, pad_index):
#     print(dec_tar.shape)
#     enc_output, enc_hidden = seq2seq.encoder(enc_inp)
#     # 第一个decoder输入 开始标签
#     # dec_input (batch_size, 1)
#     # dec_input = tf.expand_dims([start_index], 1)
#     dec_input = tf.expand_dims([start_index] * config.batch_sz, 1)
#     dec_hidden = enc_hidden
#     predictions, _ = seq2seq(dec_input, dec_hidden, enc_output, dec_tar)
#     loss = losses.loss_function(dec_tar[:, 1:],                    # 为什么要从第二个decoder算起
#                          predictions, pad_index)

if __name__ == '__main__':
    vocab = batch_utils.Vocab(config.word2index_path, 30000)
    print(len(vocab.id2word), len(vocab.word2id))
    txt = [ 3726,    33 , 1979,   163 ,  195 , 1249 ,  163 ,   23 ,   34,   129 ,  125 , 3093,
            15,   453  ,  60   , 33, 30000  , 163 ,  933 ,  117 , 1106 ,  134  ,   3]
    print([vocab.id2word[_] for _ in txt])
    # print([vocab.id2word[_] for _ in [1028, 2837, 9294, 131, 308, 128, 5, 2734, 40, 11, 192, 5, \
    #                                    2734, 87, 160, 321, 31, 87, 160, 6, 4, 16, 5, 325, \
    #                                    9, 13, 490, 33, 54, 158, 5, 131, 13, 17, 50, 6, \
    #                                    4, 490, 33, 551, 5, 131, 12, 251, 9, 1932, 1034, 103, \
    #                                    7, 4, 2747, 131, 0, 681, 9, 11, 192, 5, 22, 49, \
    #                                    1932, 18, 6, 4, 9, 5, 12, 131, 9, 164, 511, 131, \
    #                                    342, 131, 447, 121, 92, 245, 4, 251, 23, 121, 612, 66, \
    #                                    32, 512, 858, 5, 16, 17, 192, 20, 534, 550, 12, 10, \
    #                                   794, 145, 5, 198, 139, 12, 1430, 15, 120, 251, 9855, 4119, \
    #                                   7, 4, 7234, 46, 16, 5, 442, 5325, 8, 6, 4, 15, \
    #                                   137, 823, 504, 633]])
    # start_index = vocab.word_to_id('[START]')
    # pad_index = vocab.word_to_id('[PAD]')
    # seq2seq = Seq2Seq()
    # data = batch_utils.batcher(vocab)
    # batch = next(iter(data))
    # # print(batch[0]["enc_input"].shape, batch[1]["dec_target"].shape)
    # print(batch[0]["enc_input"])
    # print("fuck ",batch[0]["enc_input"][0].shape, batch[0]["enc_input"][1].shape,batch[0]["enc_input"][2].shape,)
    # batch_loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
    #                         batch[1]["dec_target"], pad_index)
