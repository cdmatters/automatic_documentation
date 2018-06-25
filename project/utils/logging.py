import sys

import tensorflow as tf

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
