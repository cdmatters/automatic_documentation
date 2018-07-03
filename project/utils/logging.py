from datetime import datetime
import os
import sys

import tensorflow as tf

SUMMARY_STR="""MINIBATCHES: {}, TRAIN_LOSS: {:5f}, VALID_LOSS: {:5f}, TEST_LOSS: {:5f},
TRAIN_BLEU: {}
VALID_BLEU: {}
TEST_BLEU: {}"""


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

def build_summary_log_string(i, train_eval_tuple, val_eval_tuple, test_eval_tuple):
    return SUMMARY_STR.format(i, train_eval_tuple[1], val_eval_tuple[1], test_eval_tuple[1],
         train_eval_tuple[0][0] * 100, val_eval_tuple[0][0] * 100, test_eval_tuple[0][0] * 100 )

def log_std_out(i, eval_tuple, valid_eval_tuple, test_eval_tuple):
    print("---------------------------------------------")
    train_log = build_translation_log_string("TRAINING", *eval_tuple)
    test_log = build_translation_log_string("TEST", *test_eval_tuple)
    summary = build_summary_log_string(i, eval_tuple, valid_eval_tuple, test_eval_tuple)

    print(train_log.encode("utf-8"))
    print(test_log.encode("utf-8"))
    print(summary.encode("utf-8"))

    sys.stdout.flush() # remove when adding a logger

if __name__=="__main__":
    save(None, 'logdir_0618_181434', 'model_1' )
