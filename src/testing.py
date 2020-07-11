import tensorflow as tf
from src.model.Seq2Seq import Seq2Seq
from src.model.pgn import PGN
from src.util.batch_utils import Vocab, batcher
from src.util.test_helper import batch_greedy_decode
from tqdm import tqdm
import  pandas  as pd
from src.util import batch_utils
from src import config

def test():
    # global model, ckpt, checkpoint_dir
    assert config.mode.lower() == "test" #, "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    if config.model == "SequenceToSequence":
        model = Seq2Seq()
    elif config.model == "PGN":
        model = PGN()
    print("Creating the vocab ...")
    vocab = batch_utils.Vocab(config.word2index_path, config.vocab_size)

    print("Creating the batcher ...")
    b = batcher(vocab)

    print("Creating the checkpoint manager")
    if config.model == "SequenceToSequence":
        checkpoint_dir = "{}/checkpoint".format(config.seq2seq_model_dir)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
    elif config.model == "PGN":
        checkpoint_dir = "{}/checkpoint".format(config.pgn_model_dir)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), PointerGeneratorModel=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    for batch in b:
        yield batch_greedy_decode(model, batch, vocab, config.max_dec_len)

def test_and_save():
    # assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test()
    results = []
    with tqdm(total=config.num_to_test, position=0, leave=True) as pbar:
        for i in range(config.num_to_test):
            trial = next(gen)
            trial = list(map(lambda x: x.replace(" ", ""), trial))
            results.append(trial[0])
            pbar.update(1)
    print(results)
    save_predict_result(results)



def save_predict_result(results):
    # 读取结果
    # test_df = pd.read_csv(config.testset_path)
    test_df = pd.read_csv(config.min_t_x_csv_path)
    print(test_df.shape, len(results))
    # 填充结果
    test_df['Prediction'] = results
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(config.results_path, index=None, sep=',')

if __name__ == '__main__':
    test_and_save()