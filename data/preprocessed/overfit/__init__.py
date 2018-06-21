from yaml import load, CLoader

from collections import namedtuple
import os

Data = namedtuple("Data", ["train","test"])

def load_overfit():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open('{}/train.yaml'.format(dirname), 'r') as f:
        train = load(f, Loader=CLoader)
    with open('{}/test.yaml'.format(dirname), 'r') as f:
        test = load(f, Loader=CLoader)
    return Data(train, test)

data = load_overfit()
