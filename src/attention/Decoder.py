import tensorflow as tf
from tensorflow import keras


class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        # self.W2 = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_mask = None,prev_coverage=None, use_coverage=False):
        # 一次计算一个dec_hidden对应的attention_vector
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        """
        :param dec_hidden:  (batch_size, hidden size)
        :param enc_output:  (batch_size, enc_len, enc_units)
        :param enc_mask:
        :param prev_coverage: (batch_size, enc_len, 1)
        :param use_coverage:
        :return: context_vector: (batch_sz, hidden size)
                 attention_weights (batch_sz, enc_len)
                 coverage: (batch_sz, enc_len,1)
        """
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)
        if use_coverage:
            tmp = tf.tile(hidden_with_time_axis, multiples=(1, enc_output.shape[1], 1))
            # score (batch_size, max_length, 1)
            score = self.V(tf.nn.tanh(self.W(tf.concat([tmp, enc_output], axis=-1)) + self.W_c(prev_coverage)))
            mask = tf.cast(enc_mask, score.dtype)
            # mask shape:(batch_sz, enc_len)
            # mask_socre shape:(batch_sz, enc_len, 1)
            mask_score = score +(1- tf.expand_dims(mask, -1))*-1.e8
            attention_weights = tf.nn.softmax(mask_score, axis=1)
            # print(attention_weights)
            coverage = attention_weights + prev_coverage
            context_vector = attention_weights * enc_output
            # context_vector shape: (batch_sz, enc_hidden)
            context_vector = tf.reduce_sum(context_vector, axis=1)
            assert context_vector.shape == (dec_hidden.shape[0], enc_output.shape[-1])
            # context_vector (batch_sz, enc_len)
            attention_weights = tf.squeeze(attention_weights, -1)
            # print(context_vector.shape, attention_weights.shape, coverage.shape)
            return context_vector, attention_weights, coverage

        else:
            tmp = tf.tile(hidden_with_time_axis, multiples=(1, enc_output.shape[1], 1))
            # we are doing this to perform addition to calculate the score
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is
            score = self.V(tf.nn.tanh(self.W(tf.concat([tmp, enc_output], axis=-1))))
            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)
            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * enc_output
            context_vector = tf.reduce_sum(context_vector, axis=1)
            assert context_vector.shape == (dec_hidden.shape[0], enc_output.shape[-1])
            # context_vector (batch_sz, enc_len)
            attention_weights = tf.squeeze(attention_weights, -1)
            return context_vector, attention_weights



class Decoder(keras.Model):
    def __init__(self, vocab_size,  embedding_dim, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        self.gru = keras.layers.GRU(dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = keras.layers.Dense(vocab_size)
        self.fc0 = tf.keras.layers.Dense(self.dec_units * 2)

    def call(self, x, context_vector, use_coverage=False):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # context_vector.shape ==(batch_size, enc_units)
        # x shape == (batch_size, 1, 1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # print(x)
        # pred:
        x = self.embedding(x)
        if use_coverage:
            dec_output, dec_hidden = self.gru(x)
            dec_output = tf.concat([tf.expand_dims(context_vector, 1), dec_output], axis=-1)
            dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
            pred = self.fc(self.fc0(dec_output))
        else:
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            dec_output, dec_hidden = self.gru(x)
            dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
            # pred shape == (batch_size, vocab)
            pred = self.fc(dec_output)
        return pred, dec_hidden


class Pointer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super(Pointer, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, x):
        # context_vector.shape ==(batch_size, enc_units)
        # dec_hidden.shape == (batch_size, dec_units)
        # dec_hidden.shape == (batch_size, embbed_dim)
        dec_inp = self.embedding(x)
        dec_inp = tf.squeeze(dec_inp, 1)
        r = tf.nn.sigmoid(self.w_s_reduce(dec_hidden) +
                      self.w_c_reduce(context_vector) +
                      self.w_i_reduce(dec_inp))
        return r

if __name__ == '__main__':
    enc = tf.random.normal([3, 28, 16])
    dec = tf.random.normal([3, 20])
    attention = BahdanauAttention(15)
    context, _ = attention(dec, enc)
    print("_________")
    decoder = Decoder(100, 32, tf.random.normal([100, 32]), 32, 10)
    x = tf.reshape(tf.range(3), (3, 1))
    hidden = tf.random.normal([16])

    decoder(x, context, use_coverage=True)

