from collections import namedtuple
import os

from yaml import load, CLoader
from yaml.constructor import Constructor


## Deal with Yaml 1.2 and 1.1 incompatibilty: Turn off 'on' == True (bool)
def add_bool(self, node):
    return self.construct_scalar(node)
Constructor.add_constructor(u'tag:yaml.org,2002:bool', add_bool)

Data = namedtuple("Data", ["train","test"])

def load_main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open('{}/main_train.yaml'.format(dirname), 'r') as f:
        train = load(f, Loader=CLoader)
    with open('{}/main_test.yaml'.format(dirname), 'r') as f:
        test = load(f, Loader=CLoader)
    return Data(train, test)

data = load_main()

