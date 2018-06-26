from datetime import datetime
import os
import sys
import shutil

import tensorflow as tf

SAVER = None

def save(session, logdir, name, iterations, max_saves=5):
    global SAVER
    if SAVER is None:
        SAVER = tf.train.Saver(max_to_keep=5)
    name = "logs/{}/{}.ckpt".format(logdir, name)
    file = SAVER.save(session, name, global_step=iterations)
    print("Saved to {}".format(file))
    return file


def load(session, logdir, name):
    global SAVER
    if SAVER is None:
        SAVER = tf.train.Saver(max_to_keep=5)
    return SAVER.restore(session, "logs/{}/{}".format(logdir, name))

def scalar_to_summary(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])

def log_tensorboard(filewriter, i, bleu_tuple, av_loss, translations):
    s = scalar_to_summary("av_loss", av_loss)
    filewriter.add_summary(s, i)
    
    b = scalar_to_summary("bleu score", bleu_tuple[0]*100)
    filewriter.add_summary(b, i)

def build_translation_log_string(prefix, bleu_tuple, loss, translations):
    log = [
        "{}: Bleu:{}, Av Loss:".format(prefix, bleu_tuple[0] * 100, loss),
        "\n--{}--\n".format(prefix[:3]).join(str(t) for t in translations),
        '--------------------'
    ]
    return "\n".join(log)

def build_summary_log_string(i, train_evaluation_tuple, test_evaluation_tuple):
    return "MINIBATCHES: {}, TRAIN_LOSS: {}, TEST_LOSS: {},\nTRAIN_BLEU: {}\nTEST_BLEU: {}".format(
        i, train_evaluation_tuple[1], test_evaluation_tuple[1],
         train_evaluation_tuple[0][0] * 100 , test_evaluation_tuple[0][0] * 100 )

def log_std_out(i, evaluation_tuple, test_evaluation_tuple):
    print("---------------------------------------------")
    train_log = build_translation_log_string("TRAINING", *evaluation_tuple)
    test_log = build_translation_log_string("TEST", *test_evaluation_tuple)
    summary = build_summary_log_string(i, evaluation_tuple, test_evaluation_tuple)

    print(train_log)
    print(test_log)
    print(summary)

    sys.stdout.flush() # remove when adding a logger

if __name__=="__main__":
    save(None, 'logdir_0618_181434', 'model_1' )
