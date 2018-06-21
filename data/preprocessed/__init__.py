from ruamel.yaml import YAML
yaml = YAML()

from collections import namedtuple
import os

Data = namedtuple("Data", ["train","test"])

def load_main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open('{}/main_train.yaml'.format(dirname), 'r') as f:
        train = yaml.load(f)
    with open('{}/main_test.yaml'.format(dirname), 'r') as f:
        test = yaml.load(f)
    return Data(train, test)

data = load_main()
