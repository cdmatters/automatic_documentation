from ruamel.yaml import YAML
yaml = YAML()



from collections import namedtuple
import os

Data = namedtuple("Data", ["train","test"])

def load_overfit():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open('{}/train.yaml'.format(dirname), 'r') as f:
        train = yaml.load(f)
    with open('{}/test.yaml'.format(dirname), 'r') as f:
        test = yaml.load(f)
    return Data(train, test)

data = load_overfit()
