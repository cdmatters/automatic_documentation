from collections import namedtuple
import os

from yaml import load, CLoader
from yaml.constructor import Constructor


## Deal with Yaml 1.2 and 1.1 incompatibilty: Turn off 'on' == True (bool)
def add_bool(self, node):
    return self.construct_scalar(node)
Constructor.add_constructor(u'tag:yaml.org,2002:bool', add_bool)

DataTuple = namedtuple("DataTuple", ["train","test"])

def load_main(prefix):
    dirname = os.path.dirname(os.path.abspath(__file__))
    train_file = '{}/{}_train.yaml'.format(dirname, prefix)
    test_file = '{}/{}_test.yaml'.format(dirname, prefix)

    if os.path.isfile(train_file) and  os.path.isfile(test_file):
        with open(train_file, 'r') as f:
            train = load(f, Loader=CLoader)
        with open(test_file, 'r') as f:
            test = load(f, Loader=CLoader)
        return DataTuple(train, test)
    else:
        return None

main_data = load_main('main')
main_data_split = load_main('split_repo')

