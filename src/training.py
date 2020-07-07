import tensorflow as tf
from src.model.Seq2Seq import Seq2Seq
from src.util.batch_utils import  batcher, Vocab
from src.util.train_helper import train_model
from src import config

def train():
    global checkpoint_dir, ckpt, model
    assert config.mode.lower() == "train"

    vocab = Vocab(config.word2index_path, config.vocab_size)
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    b = batcher(vocab)
    print("Building the model ...")
    if config.model == "SequenceToSequence":
        model = Seq2Seq()
    # elif params["model"] == "PGN":
    #     model = PGN(params)

    print("Creating the checkpoint manager")
    if config.model == "SequenceToSequence":
        checkpoint_dir = "{}/checkpoint".format(config.seq2seq_model_dir)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
    # elif params["model"] == "PGN":
    #     checkpoint_dir = "{}/checkpoint".format(params["pgn_model_dir"])
    #     ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, b, ckpt_manager, vocab)


if __name__ == '__main__':
    pass