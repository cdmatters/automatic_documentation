from yaml import load

from collections import namedtuple
import os

Data = namedtuple("Data", ["train","test"])

def load_overfit():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open('{}/train.yaml'.format(dirname), 'r') as f:
        train = load(f)
    with open('{}/test.yaml'.format(dirname), 'r') as f:
        test = load(f)
    return Data(train, test)

data = load_overfit()

