import tensorflow as tf
from tensorflow import keras
from src import config
from src.util import build_embbedingmatrix

class Encoder(keras.models.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.use_bi_gru = True
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        self.gru = keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.bi_gru = keras.layers.Bidirectional(self.gru)

    def call(self, enc_input):
        # (batch_size, enc_len, embedding_dim)
        enc_input_embedded = self.embedding(enc_input)
        initial_state = self.gru.get_initial_state(enc_input_embedded)

        if self.use_bi_gru:
            # 是否使用双向GRU
            output, forward_state, backward_state = self.bi_gru(enc_input_embedded, initial_state=initial_state * 2)
            enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)

        else:
            # 单向GRU
            output, enc_hidden = self.gru(enc_input_embedded, initial_state=initial_state)

        return output, enc_hidden

    def initialize_hidden_state(self):
        return tf.zeros(self.batch_sz, self.enc_units)

if __name__ == '__main__':
    embedding_matrix = build_embbedingmatrix.load_embbedding(config.embbeding_matrix_path)
    encoder = Encoder(vocab_size = config.vocab_size,
                               embedding_dim = config.embed_dim,
                               embedding_matrix = embedding_matrix,
                               enc_units = config.enc_units,
                               batch_sz = config.batch_sz)
    encoder.embedding(tf.range(10))
    print(encoder.embedding.embeddings[0])