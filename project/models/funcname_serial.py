import argparse
import logging

import tensorflow as tf

from project.models.base_model import ExperimentSummary
from project.models.char_baseline import CharSeqBaseline

import project.utils.args as args
import project.utils.logging as log_util
import project.utils.saveload as saveload
import project.utils.tokenize as tokenize


LOGGER = logging.getLogger('')


class FuncAndCharSeqSerial(CharSeqBaseline):
    def __init__(self, embed_tuple, rnn_size=300, batch_size=128, learning_rate=0.001, dropout=0.3, name="FuncCharSerialModel"):
        super().__init__(embed_tuple, rnn_size, batch_size, learning_rate, dropout, name)


@args.log_args
@args.train_args
@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Augment the variable name with the function name, and run through baseline model')
    parser.add_argument('--lstm-size', '-l', dest='lstm_size', action='store',
                        type=int, default=128,
                        help='size of LSTM size')
    return parser


def _run_model(name, logdir, test_freq, test_translate, save_every,
               lstm_size, dropout, lr, batch_size, epochs,
               vocab_size, char_seq, desc_seq, char_embed, desc_embed,
               use_full_dataset, use_split_dataset, **kwargs):
    log_path = log_util.to_log_path(logdir, name)
    log_util.setup_logger(log_path)

    embed_tuple, data_tuple = tokenize.get_embed_tuple_and_data_tuple(
        vocab_size, char_seq, desc_seq, char_embed, desc_embed,
        use_full_dataset, use_split_dataset, "var_funcname")
    
    nn = FuncAndCharSeqSerial(embed_tuple, lstm_size, batch_size, lr, dropout)
    summary = ExperimentSummary(nn, vocab_size, char_seq, desc_seq, char_embed, desc_embed,
                                use_full_dataset, use_split_dataset)

    LOGGER.warning("Printing to {}".format(log_path))
    LOGGER.multiline_info(summary)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4)
    sess = tf.Session(config=session_conf)

    saveload.setup_saver(save_every)
    filewriters = log_util.get_filewriters(log_path, sess)

    sess.run(init)
    # log_util.load(sess, "logdir_0618_204400", "BasicModel.ckpt-1" )
    nn.main(sess, epochs, data_tuple, log_path, filewriters,
            test_check=test_freq, test_translate=test_translate)


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    _run_model(**vars(args))
