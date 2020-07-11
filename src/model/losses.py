import tensorflow as tf


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# 定义损失函数
def loss_function(real, pred, pad_index):
    # used for baseline
    # print("real", real)
    # print("shape",real.shape, pred.shape)
    mask = tf.math.logical_not(tf.math.equal(real, pad_index))
    # print("mask", mask)
    loss_ = loss_object(real, pred)
    # print("loss",loss_.shape)
    mask = tf.cast(mask, dtype=loss_.dtype) #转换为和loss_类型相同的张量
    loss_ *= mask
    return tf.reduce_mean(loss_)


def _coverage_loss(attentions, coverages, dec_mask):
    """
    计算coverage loss
    :param attentions: shape (batch_size, dec_len, enc_len)
    :param coverages: shape (batch_size, dec_len, enc_len)
    :param dec_mask: shape (batch_size, dec_len)
    :return: cov_loss
    """
    # cov_loss (batch_size, dec_len, enc_len)
    cov_loss = tf.minimum(attentions, coverages)
    # mask
    cov_loss = tf.expand_dims(dec_mask, -1) * cov_loss

    # 对enc_len的维度求和
    cov_loss = tf.reduce_sum(cov_loss, axis=2)
    cov_loss = tf.reduce_mean(cov_loss)
    return cov_loss


def pgn_loss(target, pred, dec_mask):
    # used for pgn
    """
        计算log_loss
        :param target: shape (batch_size, dec_len)
        :param pred:  shape (batch_size, dec_len, vocab_size)
        :param dec_mask: shape (batch_size, dec_len)
        :return: log loss
        """
    loss_ = loss_object(target, pred)
    # 注batcher产生padding_mask时，数据类型需要指定成tf.float32可以少下面这行代码
    # mask = tf.cast(padding_mask, dtype=loss_.dtype)
    # print(dec_mask.shape, loss_.shape)
    loss_ *= dec_mask
    loss_ = tf.reduce_mean(loss_)
    return loss_


def calc_loss(target, pred, dec_mask, attentions, coverages, cov_loss_wt=1, use_coverage=True):
    if use_coverage:
        log_loss = pgn_loss(target, pred, dec_mask)
        cov_loss = _coverage_loss(attentions, coverages, dec_mask)
        return log_loss + cov_loss_wt * cov_loss, log_loss, cov_loss
    else:
        return pgn_loss(target, pred, dec_mask), 0, 0