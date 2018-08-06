import logging

import tensorflow as tf
import pickle
import os
import glob
import shutil
import subprocess

LOGGER = logging.getLogger('')
SAVER = None
GIT_HASH = "git log --pretty=format:'%h' -n 1"

class NoSaverException(Exception):
    pass

def get_githash():
    p = subprocess.Popen(GIT_HASH.split(), stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output

def load_args(log_path):
    with open("{}/args.pkl".format(log_path), 'rb') as f:
        kwargs = pickle.load(f)
    return kwargs

def save_args(log_path, kwargs_dict):
    with open("{}/args.pkl".format(log_path), 'wb') as f:
        pickle.dump(kwargs_dict, f)

def save(session, logpath, name, iterations, max_saves=5):
    global SAVER

    if SAVER is None:
        raise NoSaverException(
            "Instatiate saver with saveload.setup_saver(...)")
    elif SAVER == -1:
        return "Saving skipped"

    name = "{}/{}.ckpt".format(logpath, name)
    file = SAVER.save(session, name, global_step=iterations)
    LOGGER.warning("Saved to {}".format(file))
    return file

def get_latest_checkpoint(logpath):
    ckpts = [f[:-6].split(".ckpt-") for f in os.listdir(logpath) if f.endswith(".index")]
    return sorted(ckpts, key = lambda x:int(x[1]), reverse=True)[0]


def load(session, logpath):
    global SAVER
    model, iteration = get_latest_checkpoint(logpath)
    ckpt = "{}/{}.ckpt-{}".format(logpath, model, iteration)
    if SAVER is None:
        SAVER = tf.train.Saver(max_to_keep=5)
    return SAVER.restore(session, ckpt), int(iteration)

def backup_for_later(logpath, model, directory):
    dest_dir = '{}/{}/'.format(logpath, directory)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    for filename in glob.glob(r'{}*'.format( model)):
        shutil.copy(filename, dest_dir)

def setup_saver(max_saves):
    global SAVER

    if max_saves <= 0:
        SAVER = -1
    else:
        SAVER = tf.train.Saver(max_to_keep=max_saves)
