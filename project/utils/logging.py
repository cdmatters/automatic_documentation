from datetime import datetime
import logging
import os

import tensorflow as tf

LOGGER = logging.getLogger('')

SUMMARY_STR = u"""EPOCH: {} MINIB: {}, TRAIN_LOSS: {:5f}, VALID_LOSS: {:5f}, TEST_LOSS: {:5f},
TRAIN_BLEU: {}
VALID_BLEU: {}
TEST_BLEU: {}"""


def to_log_path(logdir, name):
    return "{}/{}_{}".format(logdir, name, datetime.strftime(datetime.now(), '%d%m_%H%M%S'))


def scalar_to_summary(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])


def log_tensorboard(filewriter, i, bleu_tuple, av_loss, translations):
    s = scalar_to_summary("av_loss", av_loss)
    filewriter.add_summary(s, i)

    b = scalar_to_summary("bleu score", bleu_tuple[0]*100)
    filewriter.add_summary(b, i)


def build_translation_log_string(prefix, bleu_tuple, loss, translations):
    log = [
        u"{}: Bleu:{}, Av Loss:".format(prefix, bleu_tuple[0] * 100, loss),
        u"\n--{}--\n".format(prefix[:3]).join(str(t) for t in translations),
        u'--------------------'
    ]
    return "\n".join(log)

def run_model_startup_log(summary, log_path):
    LOGGER.warning("\n./log_summary.sh -f {}/main.log # to follow\n".format(log_path))
    LOGGER.multiline_info(summary)
    LOGGER.warning("\n".join([str(v) for v in tf.trainable_variables()]))

def get_filewriters(logpath, session):
    return {
        'train_continuous':  tf.summary.FileWriter('{}/train_continuous'.format(logpath), session.graph),
        'train': tf.summary.FileWriter('{}/train'.format(logpath), session.graph),
        'valid': tf.summary.FileWriter('{}/valid'.format(logpath)),
        # 'test': tf.summary.FileWriter('{}/test'.format(logpath))
    }


def build_summary_log_string(e, i, train_eval_tuple, val_eval_tuple, test_eval_tuple):
    return SUMMARY_STR.format(e, i, train_eval_tuple[1], val_eval_tuple[1], test_eval_tuple[1],
                              train_eval_tuple[0][0] * 100, val_eval_tuple[0][0] * 100, test_eval_tuple[0][0] * 100)


def log_std_out(e, i, eval_tuple, valid_eval_tuple, test_eval_tuple):
    LOGGER.debug("---------------------------------------------")
    train_log = build_translation_log_string("TRAINING", *eval_tuple)
    valid_log = build_translation_log_string("TEST", *valid_eval_tuple)
    summary = build_summary_log_string(
        e, i, eval_tuple, valid_eval_tuple, test_eval_tuple)

    LOGGER.multiline_debug(train_log)
    LOGGER.multiline_debug(valid_log)
    LOGGER.multiline_info(summary)


def setup_logger(log_path):
    if not os.path.exists("{}".format(log_path)):
        os.makedirs("{}".format(log_path))

    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('{}/main.log'.format(log_path, encoding='utf-8'))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)5s - %(message)s', "%m%d_%H:%M")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.multiline_info = lambda x: [logger.info(
        l) for l in str(x).split("\n")]  # closure
    logger.multiline_debug = lambda x: [
        logger.debug(l) for l in str(x).split("\n")]  # closure
    return logger


if __name__ == "__main__":
    print("No __main__ implemented")
