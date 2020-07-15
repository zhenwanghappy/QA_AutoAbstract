import tensorflow as tf
import time
from src.model.losses import loss_function
from src.model.losses import pgn_loss
import  numpy as np
from src import config

def train_model(model, dataset, ckpt_manager, vocab):
    start_index = vocab.word_to_id('[START]')
    pad_index = vocab.word_to_id('[PAD]')

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=config.learning_rate)

    # @tf.function()
    def train_step(enc_inp, dec_tar,pad_index, _enc_extended_inp, enc_mask=None, dec_mask=None, batch_oov_len=None):
        with tf.GradientTape() as tape:
            # print('enc_inp shape is final for model :', enc_inp.get_shape())
            enc_output, enc_hidden = model.encoder(enc_inp)
            # 第一个decoder输入 开始标签
            # dec_input (batch_size, 1)
            # dec_input = tf.expand_dims([start_index], 1)
            dec_input = tf.expand_dims([start_index] * config.batch_sz, 1)
            dec_hidden = enc_hidden
            # print(dec_tar)
            if config.model == "SequenceToSequence":
                predictions, _ = model(dec_input, dec_hidden, enc_output, dec_tar)
            elif config.model == "PGN":
                predictions, _ = model(dec_input, dec_hidden, enc_output, dec_tar, _enc_extended_inp, enc_mask, batch_oov_len)          #error
            if config.model == "SequenceToSequence":
                loss = pgn_loss(dec_tar[:,0:],
                                 predictions, dec_mask)
            elif config.model == "PGN":
                loss = pgn_loss(dec_tar[:, 0:],
                                     predictions, dec_mask)
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        if config.gradient_clip:
            gradients, _ = tf.clip_by_global_norm(gradients, 1)
            if _ >= 1:
                print(_)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    for epoch in range(config.epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        # print(len(dataset.take(params['steps_per_epoch'])))
        for step, batch in enumerate(dataset.take(config.steps_per_epoch)):
            #讲设你的样本数是1000，batch size10,一个epoch，我们一共有100次，200， 500， 40，20.
            # print("mask",batch[0]["sample_encoder_pad_mask"])
            batch_loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                                batch[1]["dec_target"],
                                pad_index,
                                batch[0]["extended_enc_input"],
                                batch[0]['sample_encoder_pad_mask'],
                                batch[1]['sample_decoder_pad_mask'],
                                batch[0]['max_oov_len'])  # shape=(16, 50),去除不带oov的摘要
            total_loss += batch_loss
            step += 1
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, batch_loss.numpy()))

        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, total_loss/step))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss/step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))
            #学习率的衰减，按照训练的次数来更新学习率（tf1.x）
            # lr = config.learning_rate * np.power(0.9,epoch+1)
            # optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            # print("learning_rate=", optimizer.get_config()["learning_rate"])