import argparse
import os
import random
import re

import pyaml
import yaml
from yaml import CLoader, CDumper
from yaml.constructor import Constructor
from tqdm import tqdm

from project.data import preprocessed, data


## Deal with Yaml 1.2 and 1.1 incompatibilty: Turn off 'on' == True (bool)
def add_bool(self, node):
        return self.construct_scalar(node)
Constructor.add_constructor(u'tag:yaml.org,2002:bool', add_bool)

RAWDATADIR = os.path.dirname(os.path.abspath(data.__file__))
PREPROCESSDATADIR = os.path.dirname(os.path.abspath(preprocessed.__file__))

def _ad_hoc_clean(filename, line):
    '''Cleans data for specific versions of data. These are very adhoc rules,
    often simply replacing single lines'''
    f = filename.split('.')[0]
    if f == 'networkx':
        if '           type: "in+out")' in line:
            line = line.replace('           type: "in+out")',
                                '           type: "in+out"')
    elif f == 'rsa':
        if "            desc: ' The byte to count. Default \x00.'" in line:
            line = line.replace('\x00', '00')

    elif f == 'setuptools':
        if "desc: ' return '''' and not ''\x86'' if architecture is x86.'" in line:
            line = line.replace('\x86', 'x86')
        if "desc: ' return ''d'' and not ''\x07md64'' if architecture is amd64.'" in line:
            line = line.replace('\x07', 'a')

    return line

def assimilate_data():
    '''Clean and assimilate data into big yaml files (not necessarily human readable)'''
    types = ['full', 'short']
    for t in types:
        all_data = {}
        all_files = {}
        tot_f, tot_args = 0, 0
        for i, yaml_file in enumerate(os.listdir(RAWDATADIR + "/" + t + "/")):
            if yaml_file == 'error.yaml':
                continue
            with open(RAWDATADIR + "/{}/{}".format(t, yaml_file), "r", encoding='utf-8') as f:
                string = '\n'.join([_ad_hoc_clean(yaml_file, l) for l in f.readlines()])
                data = yaml.load(string.replace(
                    "            desc: `", "            desc: \\`").replace(
                    "            type: `", "            type: \\`"), Loader=CLoader)

                print("{}: {} To Update: {} ".format(i, yaml_file, len(data.keys())))
                tot = len(all_data.keys())
                all_data.update(data)
                print("{}: {} Updated: {} ".format(i, yaml_file, len(all_data.keys()) - tot))


                args = 0
                for d in data.values():
                    args += len([k for k,v in d["arg_info"].items() if v['desc']])
                    tot_args += len(d["args"])

                tot_f += len(data)
                all_files[yaml_file] = {
                    "funcs": len(data),
                    "args": args
                }

        with open(PREPROCESSDATADIR+"/index.txt", "w") as f:
            all_files["TOTAL__"] = {"funcs":tot_f, "args":tot_args}
            f.write(pyaml.dump(all_files))
        with open(PREPROCESSDATADIR+"/all_{}.yaml".format(t), "w") as f:
            f.write(yaml.dump(all_data, Dumper=CDumper))

    for t in types:
        print("Assimilated, now test loading...")
        with open(PREPROCESSDATADIR+"/all_{}.yaml".format(t), "r", encoding='utf-8') as f:
            data = yaml.load(f, Loader=CLoader)
            print("loaded {}: {} records".format(t, len(data)))
    return all_files

def map_yaml_to_arg_list(yaml_object):
    import copy
    args = []
    id_no = 0
    count = 0
    desc_count = 0
    for k, v in yaml_object.items():
        for a in v['args']:
            count += 1
            if v['arg_info'][a]['desc']:
                desc_count += 1
                data = copy.deepcopy(v)
                data['arg_name'] = a
                data['arg_desc'] = v['arg_info'][a]['desc']
                data['arg_type'] = v['arg_info'][a]['type']

                data['arg_info'].pop(a)
                args.append(data)

    print("Args: ", count, " Args with Desc: ", desc_count)
    return args

def prep_main_set(test_percentage):
    with open(PREPROCESSDATADIR+"/all_full.yaml", "r") as f:
        data = yaml.load(f, Loader=CLoader)

    main_data = map_yaml_to_arg_list(data)
    random.shuffle(main_data)

    n = len(main_data)
    print("Training Data Size: ", n)
    test = main_data[:int(n * test_percentage)]
    train = main_data[int(n * test_percentage):]

    with open(PREPROCESSDATADIR+"/main_train.yaml", 'w') as f:
        f.write(yaml.dump(train, Dumper=CDumper))
    with open(PREPROCESSDATADIR+"/main_test.yaml", 'w') as f:
        f.write(yaml.dump(test, Dumper=CDumper))

def prep_overfit_set(test_percentage):
    '''Prepare a tiny dataset from the raw data, to test overfit.'''
    with open(PREPROCESSDATADIR+"/all_full.yaml", "r") as f:
        data = yaml.load(f, Loader=CLoader)

    overfit_set = {}
    for k,v in data.items():
        if v['filename'].startswith("/numpy/"):
            overfit_set[k] = v

    overfit_data = map_yaml_to_arg_list(overfit_set)
    random.shuffle(overfit_data)

    n = len(overfit_data)
    print("Overfit Size:", n)
    test = overfit_data[:int(n * test_percentage)]
    train = overfit_data[int(n * test_percentage):]

    with open(PREPROCESSDATADIR+"/overfit/train.yaml", 'w') as f:
        f.write(yaml.dump(train, Dumper=CDumper))
    with open(PREPROCESSDATADIR+"/overfit/test.yaml", 'w') as f:
        f.write(yaml.dump(test, Dumper=CDumper))

def _build_argparser():
    parser = argparse.ArgumentParser(description='Preprocess your raw Bonaparte data into formats that can be used')
    parser.add_argument('--assimilate', '-a', dest='assimilate', action='store_true',
                        default=False, help='collect all individual yaml files and assimilate into master yaml (must be fone before prepping data sets)')
    parser.add_argument('--prep_overfit', '-o', dest='overfit_set', action='store_true',
                        default=False, help='prepare an overfit dataset from the assimilated yaml')
    parser.add_argument('--prep_data', '-d', dest='data_set', action='store_true',
                        default=False, help='prep an main dataset from the assimilated yamls')
    parser.add_argument('--run-all', '-r', dest='run_all', action='store_true',
                        default=False, help='assimilate and prep both main and overfit datasets')
    return parser

if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    if args.run_all:
        assimilate_data()
        prep_main_set(0.25)
        prep_overfit_set(0.25)
    else:
        if args.assimilate:
            assimilate_data()
        if args.data_set:
            prep_main_set(0.25)
        if args.overfit_set:
            prep_overfit_set(0.25)
    if not any(vars(args).values()):
        parser.print_help()
