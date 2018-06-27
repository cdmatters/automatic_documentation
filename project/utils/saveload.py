from datetime import datetime

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