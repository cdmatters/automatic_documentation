from datetime import datetime
import logging

import tensorflow as tf

LOGGER = logging.getLogger('')
SAVER = None

class NoSaverException(Exception):
    pass

def save(session, logdir, name, iterations, max_saves=5):
    global SAVER

    if SAVER is None:
        raise NoSaverException("Instatiate saver with saveload.setup_saver(...)")
    elif SAVER == -1:
        return "Saving skipped" 
    
    name = "logs/{}/{}.ckpt".format(logdir, name)
    file = SAVER.save(session, name, global_step=iterations)
    LOGGING.info("Saved to {}".format(file))
    return file

def load(session, logdir, name):
    global SAVER
    if SAVER is None:
        SAVER = tf.train.Saver(max_to_keep=5)
    return SAVER.restore(session, "logs/{}/{}".format(logdir, name))

def setup_saver(max_saves):
    global SAVER

    if max_saves <=0:
        SAVER = -1
    else:
        SAVER = tf.train.Saver(max_to_keep=max_saves)