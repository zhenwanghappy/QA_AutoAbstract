import tensorflow as tf
from tensorflow import keras


class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        # self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output):
        # dec_hidden shape == (batch_size, hidden size)
        # enc_output (batch_size, enc_len, enc_units)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)
        # we are doing this to perform addition to calculate the score
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        tmp = tf.tile(hidden_with_time_axis, multiples=(1, enc_output.shape[1], 1))
        score = self.V(tf.nn.tanh(self.W(tf.concat([tmp, enc_output], axis=-1))))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        assert context_vector.shape == (dec_hidden.shape[0], enc_output.shape[-1])
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

    def call(self, x, context_vector):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # print(x)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        dec_output, dec_hidden = self.gru(x)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
        # pred shape == (batch_size, vocab)
        pred = self.fc(dec_output)
        return pred, dec_hidden


if __name__ == '__main__':
    enc = tf.random.normal([3, 28, 16])
    dec = tf.random.normal([3, 20])
    attention = BahdanauAttention(15)
    context, _ = attention(dec, enc)
    print("_________")
    decoder = Decoder(100, 32, tf.random.normal([100, 32]), 32, 10)
    x = tf.reshape(tf.range(10), (10, 1))
    hidden = tf.random.normal([16])

    decoder(x, context)

