from collections import namedtuple
import os

from yaml import load, dump, CLoader, CDumper
from yaml.constructor import Constructor



## Deal with Yaml 1.2 and 1.1 incompatibilty: Turn off 'on' == True (bool)
def add_bool(self, node):
    return self.construct_scalar(node)
Constructor.add_constructor(u'tag:yaml.org,2002:bool', add_bool)

DataTuple = namedtuple("DataTuple", ["train","valid","test","name"])
DataTuple.__str__ = lambda s: "Name: {} | Tr: {}, Vd: {}, Te: {}".format(
                       s.name, len(s.train), len(s.valid), len(s.test))

def load_data(prefix, validation=0.3):
    dirname = os.path.dirname(os.path.abspath(__file__))
    train_file = '{}/{}/{}_train.yaml'.format(dirname, prefix, prefix)
    test_file = '{}/{}/{}_test.yaml'.format(dirname, prefix, prefix)

    if os.path.isfile(train_file) and  os.path.isfile(test_file):
        with open(train_file, 'r') as f:
            train = load(f, Loader=CLoader)
        with open(test_file, 'r') as f:
            test_total = load(f, Loader=CLoader)
            val_size = int(len(test_total) * validation)
            valid = test_total[:val_size]
            test =  test_total[val_size:]
        return DataTuple(train, valid, test, prefix)
    else:
        return None

def save_data(train_data, test_data, name):
    dirname = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(dirname+'/'+name):
        os.makedirs(dirname+'/'+name)

    with open(dirname+"/{}/{}_train.yaml".format(name, name), 'w') as f:
        f.write(dump(train_data, Dumper=CDumper))
    with open(dirname+"/{}/{}_test.yaml".format(name, name), 'w') as f:
        f.write(dump(test_data, Dumper=CDumper))
    with open(dirname+"/{}/__init__.py".format(name, name), 'w') as f:
        l = 'from project.data.preprocessed import load_data\n\n{}_data = load_data("{}")'.format(name,name)
        f.write(l)
