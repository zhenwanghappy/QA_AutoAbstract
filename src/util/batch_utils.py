import tensorflow as tf
from src import config


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

class Vocab:
    def __init__(self, vocab_file, max_size):
        self.word2id = {UNKNOWN_TOKEN: 0, PAD_TOKEN: 1, START_DECODING: 2, STOP_DECODING: 3}
        self.id2word = {id: w for w, id in self.word2id.items()}
        self.count = 4

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning : incorrectly formatted line in vocabulary file : %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(r'<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, '
                                    r'but %s is' % w)
                self.word2id[w] = self.count
                self.id2word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                          % (max_size, self.count))
                    break

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder,
    and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer
    than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id
    (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


def example_generator(vocab, train_x_path, train_y_path, test_x_path, max_enc_len, max_dec_len, mode, batch_size):
    if mode == "train":
        dataset_train_x = tf.data.TextLineDataset(train_x_path)
        dataset_train_y = tf.data.TextLineDataset(train_y_path)
        train_dataset = tf.data.Dataset.zip((dataset_train_x, dataset_train_y))
        train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat()
        # i = 0
        for raw_record in train_dataset:
            article = raw_record[0].numpy().decode("utf-8")
            abstract = raw_record[1].numpy().decode("utf-8")

            start_decoding = vocab.word_to_id(START_DECODING)
            stop_decoding = vocab.word_to_id(STOP_DECODING)

            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)
            # print('enc_inp shape is final for dataset:',enc_len)
            # 添加mark标记
            # print('enc_len is', enc_len)
            sample_encoder_pad_mask = [0.] * max_enc_len
            sample_encoder_pad_mask[:enc_len] = [1.]*enc_len
            # print('sample_encoder_pad_mask is', sample_encoder_pad_mask)

            enc_input = [vocab.word_to_id(w) for w in article_words]
            # print('enc_inp shape is final for dataset:', len(enc_input))
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

            abstract_sentences = [""]
            abstract_words = abstract.split()
            abs_ids = [vocab.word_to_id(w) for w in abstract_words]
            abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
            dec_input, target = get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)
            # if config.model == "PGN":
            #    dec_input, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)

            dec_len = len(dec_input)
            # 添加mark标记
            # print('dec_len ids ', dec_len)
            sample_decoder_pad_mask = [0.] * max_dec_len
            sample_decoder_pad_mask[:dec_len] = [1.] * dec_len
            # print('sample_decoder_pad_mask is ', sample_decoder_pad_mask)
            output = {
                "enc_len": enc_len,                                        # min{摘要长度, max_enc_len}
                "enc_input": enc_input,                                    # 不带有oov的文章，oov用<unk>表示
                "enc_input_extend_vocab": enc_input_extend_vocab,          # 具有oov的文章
                "article_oovs": article_oovs,                              # 文章的oov列表
                "dec_input": dec_input,                                    # seq2seq中不帶oov的摘要句子/pgn中是带有oov的句子,以<start>开始
                "target": target,                                          # 在seq2seq中是没有oov的摘要句子/pgn中是带有oov的摘要句子，以<end>结尾
                "dec_len": dec_len,                                        # min{摘要长度+1, max_dec_len}
                "article": article,
                "abstract": abstract,
                "abstract_sents": abstract_sentences,
                "sample_decoder_pad_mask": sample_decoder_pad_mask,
                "sample_encoder_pad_mask": sample_encoder_pad_mask,
            }
            assert enc_len <= max_enc_len
            yield output

    if mode == "test":
        test_dataset = tf.data.TextLineDataset(test_x_path)
        for raw_record in test_dataset:
            # print('raw_record', raw_record)
            # print('raw_record length is:',raw_record.get_shape())
            article = raw_record.numpy().decode("utf-8")
            # print('article length is:', len(article)) #277
            # print(article)
            # print('max_enc_len value is :',max_enc_len)
            # print('article.split() length is:', len(article.split()))
            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)

            enc_input = [vocab.word_to_id(w) for w in article_words]
            # print('enc_input length in generator',len(enc_input)) #99
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

            sample_encoder_pad_mask = [1 for _ in range(enc_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": [],
                "target": [],
                "dec_len": 40,
                "article": article,
                "abstract": '',
                "abstract_sents": [],
                "sample_decoder_pad_mask": [],
                "sample_encoder_pad_mask": sample_encoder_pad_mask,
            }
            # print('output is ', output)
            for _ in range(batch_size):
                yield output

            # yield output


def batch_generator(generator, vocab, train_x_path, train_y_path,
                    test_x_path, max_enc_len, max_dec_len, batch_size, mode):
    dataset = tf.data.Dataset.from_generator(lambda: generator(vocab, train_x_path, train_y_path, test_x_path,
                                                               max_enc_len, max_dec_len, mode, batch_size),
                                             output_types={
                                                 "enc_len": tf.int32,
                                                 "enc_input": tf.int32,
                                                 "enc_input_extend_vocab": tf.int32,
                                                 "article_oovs": tf.string,
                                                 "dec_input": tf.int32,
                                                 "target": tf.int32,
                                                 "dec_len": tf.int32,
                                                 "article": tf.string,
                                                 "abstract": tf.string,
                                                 "abstract_sents": tf.string,
                                                 "sample_decoder_pad_mask": tf.float32,
                                                 "sample_encoder_pad_mask": tf.float32,
                                             },
                                             output_shapes={
                                                 "enc_len": [],
                                                 "enc_input": [None],
                                                 "enc_input_extend_vocab": [None],
                                                 "article_oovs": [None],
                                                 "dec_input": [None],
                                                 "target": [None],
                                                 "dec_len": [],
                                                 "article": [],
                                                 "abstract": [],
                                                 "abstract_sents": [None],
                                                 "sample_decoder_pad_mask": [None],
                                                 "sample_encoder_pad_mask": [None],
                                             })

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=({"enc_len": [],
                                                   "enc_input": [max_enc_len],                        # modified
                                                   "enc_input_extend_vocab": [max_enc_len],           # modified
                                                   "article_oovs": [None],
                                                   "dec_input": [max_dec_len],
                                                   "target": [max_dec_len],
                                                   "dec_len": [],
                                                   "article": [],
                                                   "abstract": [],
                                                   "abstract_sents": [None],
                                                   "sample_decoder_pad_mask": [max_dec_len],
                                                   "sample_encoder_pad_mask": [max_enc_len]}),
                                   padding_values={"enc_len": -1,
                                                   "enc_input": 1,
                                                   "enc_input_extend_vocab": 1,
                                                   "article_oovs": b'',
                                                   "dec_input": 1,
                                                   "target": 1,
                                                   "dec_len": -1,
                                                   "article": b'',
                                                   "abstract": b'',
                                                   "abstract_sents": b'',
                                                   "sample_decoder_pad_mask": 0.,
                                                   "sample_encoder_pad_mask": 0.},
                                   drop_remainder=True)

    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "max_oov_len": tf.shape(entry["article_oovs"])[1],
                 "sample_encoder_pad_mask": entry["sample_encoder_pad_mask"]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"],
                 "sample_decoder_pad_mask": entry["sample_decoder_pad_mask"]})

    dataset = dataset.map(update)
    return dataset


def batcher(vocab):
    dataset = batch_generator(example_generator, vocab, config.min_x_path, config.min_y_path,
                              config.test_x_path, config.max_enc_len,
                              config.max_dec_len, config.batch_sz, config.mode)
    # dataset = batch_generator(example_generator, vocab, config.train_x_path, config.train_y_path,
    #                           config.test_x_path, config.max_enc_len,
    #                           config.max_dec_len, config.batch_sz, config.mode)
    return dataset


def output_to_words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id_to_word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. " \
                                             "This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this '
                                 'example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words



if __name__ == '__main__':
    vocab = Vocab(config.word2index_path, 3000)
    print(vocab[0])
    from src.util.build_embbedingmatrix import load_embbedding
    embbeding = load_embbedding(config.embbeding_matrix_path)
    print(embbeding[0])

    # print("fuckyou", article_to_ids("阿卡麗 你好 师傅 银河系 火星文".split()[:10], vocab))
    # print([vocab.word_to_id(w) for w in "阿卡麗 你好 师傅 银河系 火星文".split()[:10]])
    # a, b = article_to_ids(["阿卡麗", "你好",'师傅',"银河系","火星文"], vocab)
    # sample_encoder_pad_mask = [1 for _ in range(5)]
    # print("mask",sample_encoder_pad_mask)
    # print(a, b)
    # abs_ids_extend_vocab = abstract_to_ids(['你好', "火星文"], vocab, b)
    abs_ids = [vocab.word_to_id(w) for w in ['你好', "火星文"]]
    start_decoding = vocab.word_to_id(START_DECODING)
    stop_decoding = vocab.word_to_id(STOP_DECODING)
    dec_input, target = get_dec_inp_targ_seqs(abs_ids, 40, start_decoding, stop_decoding)
    print(abs_ids, start_decoding, stop_decoding)
    print(dec_input, target)                       # [2, 24, 0] [24, 0, 3]
    # print(abs_ids_extend_vocab)
    # _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, 40, start_decoding, stop_decoding)
    # print(_, target)
    # print([1 for _ in range(len(dec_input))])

    # iter = example_generator(vocab, "x.txt", "y.txt", "t_x.txt", 400, 40, "test", 2)
    # for ex in iter:
    #     print(ex)
    #     enc_input = ex['enc_input']
    #     print(len(enc_input))

    # data = batcher(vocab)
    # i = 0
    # for x in data:
    #     if i>=1:
    #         break
    #     print(i, x)
    #     i+=1


    # s1 =[ 923,    0,  491,  301,  463, 1813,    6,    4,   16,   12,   73,
    #       56,  159,   26,  501,    7,    4,   10,    6,    4,  622,   27,
    #        7,    4,   29,  214,  249,   26,  163,  167,    9,  814,  221,
    #      937,   58,  133,   19,   26,   29,   87,  221,  155,   28,    0,
    #        5,   26,    9,  428,   12,   26,    7,    4, 1180,  202,    7,
    #        4,  167,    0,  239,    6,    4,    7,    4,   12,   51,   29,
    #        5,  155,  814,  937,   10,  176,  482,   18,    6,    4,    7,
    #        4,   74,   32,  428,  103,  491,  301,    0,    0,   51, 1404,
    #      785,   51, 1533,  785,   51,    0, 1345,    6,    4,    7,    4,
    #       66,  489,  322,   37,  240,   47,    5, 1592,   26,   10,  176,
    #      199,   50,    6,    4,    7,    4,   12,   10,  388,  159,  277,
    #     2143,   18,    6,    4,    7,    4,   46,    6,    4,   15,  137,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1]
    # s2 =[1721,   72,  372,  124,   94, 1148,  141,  635,  237,   49,  308,
    #      105,   84,  324,  426, 1202,  202,    5,    0,   74,   84,    0,
    #       82,  838,  407,   23,  120,  187,  635,   15,   86,   94,  257,
    #      216,  539,    0,    0,  784,   13,   17,    8,  200, 1013,   49,
    #     1672, 1340,   82,    6,    4,   24,   89,   94,   38, 1148,  141,
    #     1388,  162,   35,   86,   71,  138, 2576,  155,   32,   84,  883,
    #        8,   25,  260,   68,  598,   84,   71,  184,   84,  426,  440,
    #       89,   17, 1168,  162,   35,   86,   13,   61, 2327,  426, 1202,
    #       79,   36,    0,   13,   61,  532,   38,  483, 1805, 1672,  372,
    #        0,   32,    0,    8,  184,   84,   90,  838,  407,   23,  120,
    #        5,  842,  407,   25,  443,  206,   56,  372,  184,   84,   16,
    #       17, 1218,   70,  427,   79,  799,   16,    7,    4,   89,   30,
    #      407,   25,   56,  184,   84,  885,    0,  158,    5,   83,    0,
    #      511,    5,    7,    4, 1202,  520,  532,    8, 1672,    9,   20,
    #      201,    7, 1611,   78,   18,    6,    4,   24,   89,    0,  407,
    #      742,   71,   25,  184,   84, 1227,  508,    0,   37, 1029,  407,
    #       68,  598,   84,   71,   25,  443,  206,   56,  260,  184,   84,
    #     1202,  532,    8, 1672,   89,    9,   20, 1202, 1843,    5,    0,
    #       78,   17,  553, 1672]
    # s3 = [ 506, 1035,  334,  312,  506,   23,   31,    8,   44,  120,    6,
    #        4,    7,    4,  158,    5,   23,   31,    8,    7,    4,  334,
    #      312,  506,    6,    4,    7,    4,   31,    8,  506,  573,    5,
    #       10,    7,    4,    6,    4,    7,    4,   10,  334,   32, 1035,
    #      675,   49,  184,   50,    6,    4,   15,   25,    6,    4,   16,
    #       66,    9,   33,    5,  698,   10,   14,    7,    4,   98,  536,
    #        8, 1835,  266,   13,  486, 1035,    5,  104,    6,    4,    9,
    #      750, 1035,   18,    7,    4,    9,    5,    0,    6,    4,    7,
    #        4,   11,  524,   22,    9,  772,    7,    9,  290,   41,  456,
    #     1035, 1961,   50,    0,  312,   31,    8,  168,  133,   53,  471,
    #      686,    0,    8,   58,   95,  100,   13,   15,  124, 1131,   37,
    #        7, 2248,    6,    4,    6,    4]
    # print([vocab.id2word[i] for i in s1])
    # print([vocab.id2word[i] for i in s2])
    # print([vocab.id2word[i] for i in s3])
    # t1 = [   2,  408,  482,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    #        1,    1,    1,    1,    1,    1,    1]
    # t2 = [   2,   24,   89,   94,   38,  742,   71,  260,  162,   35,   86,
    #       34,   94,  138, 2576,   15, 1511,  155,   32,   25,  260,   68,
    #      598,   84,   71,  184,   84,  426,   38,   94,   79,   36, 1168,
    #       32,   36,    5, 1202,   43,  465,   79]
    # print([vocab.id2word[i] for i in t1])
    # print([vocab.id2word[i] for i in t2])
