from collections import namedtuple
import os

from yaml import load, dump, CLoader, CDumper
from yaml.constructor import Constructor
from pyaml import dump as pdump


# Deal with Yaml 1.2 and 1.1 incompatibilty: Turn off 'on' == True (bool)
def add_bool(self, node):
    return self.construct_scalar(node)


Constructor.add_constructor(u'tag:yaml.org,2002:bool', add_bool)

DataTuple = namedtuple("DataTuple", ["train", "valid", "test", "name"])
DataTuple.__str__ = lambda s: "Name: {} | Tr: {}, Vd: {}, Te: {}".format(
    s.name, len(s.train), len(s.valid), len(s.test))


def load_data(prefix, name, validation=0.3):
    dirname = os.path.dirname(os.path.abspath(__file__))
    train_file = '{}/{}/{}_train.yaml'.format(dirname, prefix, name)
    test_file = '{}/{}/{}_test.yaml'.format(dirname, prefix, name)

    if os.path.isfile(train_file) and os.path.isfile(test_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            train = load(f, Loader=CLoader)
        with open(test_file, 'r', encoding='utf-8') as f:
            test_total = load(f, Loader=CLoader)
            val_size = int(len(test_total) * validation)
            valid = test_total[:val_size]
            test = test_total[val_size:]
        return DataTuple(train, valid, test, prefix)
    else:
        return None

def load_vocab(name, subname=None):
    dirname = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(dirname+'/'+name):
        os.makedirs(dirname+'/'+name)

    if subname == None:
        filename = name
    else:
        filename = name + "_" + subname

    with open(dirname+"/{}/{}.vocab".format(name, filename), 'r', encoding='utf-8') as f:
        vocab_list = load(f, Loader=CLoader)

        voc2idx = {None:0, "<UNK>":1}
        voc2count = {}
        for v,c in vocab_list:
            voc2idx[v] = len(voc2idx)
            voc2count[v] = int(c)
    return voc2idx, voc2count


def save_vocab(counter, name, subname=None):
    dirname = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(dirname+'/'+name):
        os.makedirs(dirname+'/'+name)

    if subname == None:
        filename = name
    else:
        filename = name + "_" + subname

    with open(dirname+"/{}/{}.vocab".format(name, filename), 'w', encoding='utf-8') as f:
        counter = [[v, str(c)] for v,c in counter]
        f.write(dump(counter,  Dumper=CDumper))
        



def save_data(train_data, test_data, name, subname=None):
    dirname = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(dirname+'/'+name):
        os.makedirs(dirname+'/'+name)
    
    import_stmt = '{}_data = lambda : load_data("{}", "{}")'.format(name, name, name)
    if subname == None:
        filename = name
    else:
        filename = name + "_" + subname
        import_stmt += '\n{}_data = lambda : load_data("{}", "{}")'.format(filename, name, filename)


    with open(dirname+"/{}/{}_train.yaml".format(name, filename), 'w', encoding='utf-8') as f:
        f.write(dump(train_data, Dumper=CDumper))
    with open(dirname+"/{}/{}_test.yaml".format(name, filename), 'w', encoding='utf-8') as f:
        f.write(dump(test_data,  Dumper=CDumper))
    with open(dirname+"/{}/__init__.py".format(name), 'w', encoding='utf-8') as f:
        l = 'from project.data.preprocessed import load_data\n\n{}'.format(import_stmt)
        f.write(l)
