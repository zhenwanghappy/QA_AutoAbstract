import tensorflow as tf
import numpy as np
from src.util.batch_utils import output_to_words
from src import config


def batch_beam_search(model, enc_data, vocab, max_dec_len, beam_width=2):
    pass


def decode_one_step(model, enc_extended_inp, batch_oov_len, dec_input, context_vector,
                    enc_pad_mask, prev_coverage, use_coverage=True):
    # 开始decoder
    context_vector, dec_hidden, \
    dec_x, pred, attn, coverage = model.decoder(dec_input,
                                                # dec_hidden,
                                                # enc_output,
                                                context_vector,
                                                use_coverage)

    # 计算p_gen
    p_gen = model.pointer(context_vector, dec_hidden, dec_x)

    # 保证pred attn p_gen的参数为3D的
    final_dist = model._calc_final_dist(enc_extended_inp,
                                  tf.expand_dims(pred, 1),
                                  tf.expand_dims(attn, 1),
                                  tf.expand_dims(p_gen, 1),
                                  batch_oov_len,
                                  config.vocab_size,
                                  config.batch_sz)

    return final_dist, dec_hidden, coverage


def batch_greedy_decode(model, enc_data, vocab, max_dec_len):
    # 判断输入长度
    # print(enc_data)
    # global outputs
    batch_data = enc_data[0]["enc_input"]
    batch_size = enc_data[0]["enc_input"].shape[0]
    enc_mask = enc_data[0]["sample_encoder_pad_mask"]
    batch_oov_len = enc_data[0]['max_oov_len']
    _enc_extended_inp = enc_data[0]["extended_enc_input"]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data

    enc_output, enc_hidden = model.encoder(inputs)
    dec_hidden = enc_hidden
#这里解释下为什么要有一个batch_size,因为训练得时候是按照一个batch size扔进去得，所以得到得模型得输入结构也是如此，因此在测试得时候相当于将单个样本
#乘以batch size那么多遍，然后再得到结果，结果区list得第一个即可，当然理论上list得内容是一样得
    dec_input = tf.constant([vocab.word_to_id('[START]')] * batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)
    # print('enc_output shape is :',enc_output.get_shape())
    # print('dec_hidden shape is :', dec_hidden.get_shape())
    # print('inputs shape is :', inputs.get_shape())
    # print('dec_input shape is :', dec_input.get_shape())
    if config.model =="SequenceToSequence":
        context_vector, _ = model.attention(dec_hidden, enc_output)
        for t in range(max_dec_len):
            # 单步预测
            # predictions.shape (batch_size, embbed_dim)
            predictions, dec_hidden = model.decoder(dec_input,
                                                    # dec_hidden,
                                                    # enc_output,
                                                    context_vector)
            predicted_ids = tf.argmax(predictions, axis=1).numpy()
            for index, predicted_id in enumerate(predicted_ids):
                predicts[index] += vocab.id_to_word(predicted_id) + ' '
            # dec_input = tf.expand_dims(predicted_ids, 1)
            context_vector, _ = model.attention(dec_hidden, enc_output)
            dec_input = tf.expand_dims(predicted_ids, 1)
    elif config.model == "PGN":
        prev_coverage = tf.zeros((enc_output.shape[0], enc_output.shape[1], 1))
        context_vector, attn, prev_coverage = model.attention(dec_hidden, enc_output, enc_mask, prev_coverage, True)
        for t in range(max_dec_len):
            # 单步预测
            # final_dist (batch_size, 1, vocab_size+batch_oov_len)
            pred, dec_hidden = model.decoder(dec_input,
                                             # dec_hidden,
                                             # enc_output,
                                             context_vector,
                                             True)
            p_gen = model.pointer(context_vector, dec_hidden, dec_input)
            # final_dist.shape (batch_size, embbedim+batch_oov_len)
            final_dist = model._calc_final_dist(_enc_extended_inp,
                                         tf.expand_dims(pred, 1),
                                         tf.expand_dims(attn, 1),
                                         tf.expand_dims(p_gen, 1),
                                         batch_oov_len,
                                         config.vocab_size,
                                         config.batch_sz)
            final_dist = tf.squeeze(final_dist, 1)
            # print(final_dist)
            predicted_ids = tf.argmax(final_dist, axis=1).numpy()
            for index, predicted_id_and_oov in enumerate(zip(predicted_ids, enc_data[0]['article_oovs'])):
                predicted_id, oovs = predicted_id_and_oov[0], predicted_id_and_oov[1]
                if predicted_id < config.vocab_size:
                    predicts[index] += vocab.id_to_word(predicted_id) + ' '
                else:
                    predicts[index] += oovs[predicted_id-config.vocab_size].numpy().decode()+" "
            # dec_input = tf.expand_dims(predicted_ids, 1)
            context_vector, attn, prev_coverage = model.attention(dec_hidden, enc_output, enc_mask, prev_coverage, True)
            dec_input = tf.expand_dims(predicted_ids, 1)
    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if '[STOP]' in predict:
            # 截断stop
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    return results



