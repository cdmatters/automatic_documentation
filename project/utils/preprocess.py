import argparse
from collections import Counter
import os
import random

import pyaml
import yaml
from yaml import CLoader, CDumper
from yaml.constructor import Constructor

from project.data import preprocessed, data
from project.utils.tokenize import nltk_tok
from project.utils.code_tokenize import return_populated_codepath
random.seed(100)

# Deal with Yaml 1.2 and 1.1 incompatibilty: Turn off 'on' == True (bool)
def add_bool(self, node):
    return self.construct_scalar(node)


Constructor.add_constructor(u'tag:yaml.org,2002:bool', add_bool)

RAWDATADIR = os.path.dirname(os.path.abspath(data.__file__))
PREPROCESSDATADIR = os.path.dirname(os.path.abspath(preprocessed.__file__))

def to_quickload(data):
    new_data = []
    for d in data:
        new_data.append({
            "arg_name": d["arg_name"],
            "arg_desc": d["arg_desc"],
            "path_idx": " ".join(str(n) for n in d["path_idx"]),
            "target_var_idx": " ".join(str(n) for n in d["target_var_idx"]),
            "target_var_mask_idx": " ".join(str(n) for n in d["target_var_mask_idx"]),
            "target_var_mask_names": " ".join(str(n) for n in d["target_var_mask_names"]),
            "name": d["name"],
            "args": d["args"],
            "pkg": d["pkg"]
        })
    return new_data


def gen_code_vocab_files(data, name, subname):
    all_paths = []
    all_target_vars = []
    for d in data:
        all_paths.extend(d["path_strings"])
        all_target_vars.extend(d["target_var_string"])
    count_paths = Counter(all_paths).most_common()
    count_vars = Counter(all_target_vars).most_common()

    preprocessed.save_vocab(count_paths, name, subname+'_paths')
    preprocessed.save_vocab(count_vars, name, subname+'_tvs')

def tok_code_vocab_files(data, name, subname):
    voc2idx_path, voc2count_path = preprocessed.load_vocab(name, subname+'_paths')
    voc2idx_tv, voc2count_tv = preprocessed.load_vocab(name, subname+'_tvs')

    for d in data:
        names = list(set(d["target_var_string"]))
        name_dict = {n:i for i,n in enumerate(names)}
        d['path_idx'] = [voc2idx_path[p] if p in voc2idx_path else 0 for p in d["path_strings"] ]
        d['target_var_idx'] = [voc2idx_tv[p] if p in voc2idx_tv else 0 for p in d["target_var_string"]]
        d['target_var_mask_idx'] = [name_dict[p] for p in d["target_var_string"]]
        d['target_var_mask_names'] = names
    return data

def save_quickload_version(data, name, test_percentage):
    do_split = (len(data) == 1)

    if do_split:
        data = return_populated_codepath(data[0])
        n = len(data)
        test = data[:int(n * test_percentage)]
        train = data[int(n * test_percentage):]
    else:
        assert len(data) == 2
        train = return_populated_codepath(data[0])
        test = return_populated_codepath(data[1])

    gen_code_vocab_files(train, name, "quickload")
    train = to_quickload(tok_code_vocab_files(train, name, "quickload"))
    test = to_quickload(tok_code_vocab_files(test, name, "quickload"))

    preprocessed.save_data(train, test, name, "quickload")





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
                string = ''.join([_ad_hoc_clean(yaml_file, l)
                                    for l in f.readlines()])
                data = yaml.load(string.replace(
                    "            desc: `", "            desc: \\`").replace(
                    "            type: `", "            type: \\`"), Loader=CLoader)

                print("{}: {} To Update: {} ".format(
                    i, yaml_file, len(data.keys())))
                tot = len(all_data.keys())
                all_data.update(data)
                print("{}: {} Updated: {} ".format(
                    i, yaml_file, len(all_data.keys()) - tot))

                args = 0
                for d in data.values():
                    args += len([k for k, v in d["arg_info"].items() if v['desc']])
                    tot_args += len(d["args"])

                tot_f += len(data)
                all_files[yaml_file] = {
                    "funcs": len(data),
                    "args": args
                }

        with open(PREPROCESSDATADIR+"/index.txt", "w", encoding='utf-8') as f:
            all_files["TOTAL__"] = {"funcs": tot_f, "args": tot_args}
            f.write(pyaml.dump(all_files))
        with open(PREPROCESSDATADIR+"/all_{}.yaml".format(t), "w", encoding='utf-8') as f:
            f.write(pyaml.dump(all_data))#, Dumper=CDumper))

    for t in types:
        print("Assimilated, now test loading...")
        with open(PREPROCESSDATADIR+"/all_{}.yaml".format(t), "r", encoding='utf-8') as f:
            data = yaml.load(f, Loader=CLoader)
            print("loaded {}: {} records".format(t, len(data)))
    return all_files


def map_yaml_to_arg_list(yaml_object):
    import copy
    args = []
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
    print("starting: prep_main_set")
    with open(PREPROCESSDATADIR+"/all_full.yaml", "r", encoding='utf-8') as f:
        data = yaml.load(f, Loader=CLoader)

    main_data = map_yaml_to_arg_list(data)
    random.shuffle(main_data)

    n = len(main_data)
    print("Training Data Size: ", n)
    test_data = main_data[:int(n * test_percentage)]
    train_data = main_data[int(n * test_percentage):]

    preprocessed.save_data(train_data, test_data, 'unsplit')
    save_quickload_version([train_data, test_data], 'unsplit', test_percentage)


def prep_no_dups_split(train_data, test_data):
    for dups in [1,2,3,4,5,10]:
        dups_str = str(dups) if dups != 10 else 'X'
        filtered_train_data = []
        filtered_test_data = []
        c = Counter()
        for d in train_data:
            uniq_pair = d['arg_name']+'|'+ " ".join(nltk_tok(d['arg_desc']))
            c.update({uniq_pair: 1})

            if c[uniq_pair] <= dups:
                filtered_train_data.append(d)

        for d in test_data:
            uniq_pair = d['arg_name']+'|'+ " ".join(nltk_tok(d['arg_desc']))
            c.update({uniq_pair: 1})

            if c[uniq_pair] <= dups:
                filtered_test_data.append(d)

        frac = len(filtered_test_data)/(len(filtered_train_data) + len(filtered_test_data))
        print("Split No Dups{}: Train: {} Test and Valid: {},  Fraction:{}".format(
            dups_str, len(filtered_train_data), len(filtered_test_data), frac))
        preprocessed.save_data(filtered_train_data, filtered_test_data, 'no_dups_split_{}'.format(dups_str))
        save_quickload_version([filtered_train_data, filtered_test_data], 'no_dups_split_{}'.format(dups_str), 0.0)


def prep_no_duplicates(test_percentage):
    print("starting: prep_no_duplicates")
    with open(PREPROCESSDATADIR+"/all_full.yaml", "r", encoding='utf-8') as f:
        data = yaml.load(f, Loader=CLoader)

    main_data = map_yaml_to_arg_list(data)
    random.shuffle(main_data)
    for dups in [1,2,3,4,5,10]:
        dups_str = str(dups) if dups != 10 else 'X'
        filtered_data = []
        c = Counter()
        for d in main_data:
            uniq_pair = d['arg_name']+'|'+ " ".join(nltk_tok(d['arg_desc']))
            c.update({uniq_pair: 1})

            if c[uniq_pair] <= dups:
                filtered_data.append(d)

        n = len(filtered_data)
        print("Total Data Size for NoDup{}: {}".format(dups_str, n))
        test_data = filtered_data[:int(n * test_percentage)]
        train_data = filtered_data[int(n * test_percentage):]


        preprocessed.save_data(train_data, test_data, 'no_dups_{}'.format(dups_str))
        save_quickload_version([filtered_data], 'no_dups_{}'.format(dups_str), test_percentage)

def prep_overfit_set(test_percentage):
    '''Prepare a tiny dataset from the raw data, to test overfit.'''
    print("starting: prep_overfit_set")
    with open(PREPROCESSDATADIR+"/all_full.yaml", "r", encoding='utf-8') as f:
        data = yaml.load(f, Loader=CLoader)

    overfit_set = {}
    for k, v in data.items():
        if v['filename'].startswith("/numpy/"):
            overfit_set[k] = v

    overfit_data = map_yaml_to_arg_list(overfit_set)
    random.shuffle(overfit_data)

    n = len(overfit_data)
    print("Overfit Size:", n)
    test = overfit_data[:int(n * test_percentage)]
    train = overfit_data[int(n * test_percentage):]

    preprocessed.save_data(train, test, 'overfit')
    save_quickload_version([overfit_data], 'overfit', test_percentage)

def prep_repo_split_set(test_percentage):
    '''Prepare a data set with training and test data from different repositories'''
    print("starting: prep_repo_split_set")
    with open(PREPROCESSDATADIR + "/index.txt", "r", encoding='utf-8') as f:
        index_data = yaml.load(f, Loader=CLoader)

    total = index_data.pop("TOTAL__")
    test_set_size = total["args"] * test_percentage

    count = 0
    test_repos = []

    for k, v in index_data.items():
        name = k.split('.')[0]
        if name in ['scipy', 'numpy']:
            continue

        test_repos.append(name)
        count += v["args"]

        if count > test_set_size:
            break

    with open(PREPROCESSDATADIR+"/all_full.yaml", "r", encoding='utf-8') as f:
        data = yaml.load(f, Loader=CLoader)

    test_set = {}
    train_set = {}
    for k, v in data.items():
        if v['pkg'] in test_repos:
            test_set[k] = v
        else:
            train_set[k] = v

    train_data = map_yaml_to_arg_list(train_set)
    test_data = map_yaml_to_arg_list(test_set)

    random.shuffle(train_data)
    random.shuffle(test_data)

    print("Test Args: {}".format(len(test_data)))
    print("Train Args: {}".format(len(train_data)))
    print("Test Fraction: {:4f}".format(
        len(test_data)/(len(test_data) + len(train_data))))

    preprocessed.save_data(train_data, test_data, 'split')
    save_quickload_version([train_data, test_data], 'split', test_percentage)

    prep_no_dups_split(train_data, test_data)



def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Preprocess your raw Bonaparte data into formats that can be used')
    parser.add_argument('--assimilate', '-a', dest='assimilate', action='store_true',
                        default=False, help='collect all individual yaml files and assimilate into master yaml (must be fone before prepping data sets)')
    parser.add_argument('--prep_overfit', '-o', dest='overfit_set', action='store_true',
                        default=False, help='prepare an overfit dataset from the assimilated yaml')
    parser.add_argument('--prep_combined_repos', '-c', dest='unsplit_set', action='store_true',
                        default=False, help='prep datasets with train and test from combined repositories')
    parser.add_argument('--prep_separate_repos', '-s', dest='sep_repos', action='store_true',
                        default=False, help='prepare data sets with train and test from different repositories')
    parser.add_argument('--prep_no_duplicates', '-d', dest='no_dups', action='store_true',
                        default=False, help='prepare data sets with train and test without duplicates of (name, desc) pairs')
    parser.add_argument('--run-all', '-r', dest='run_all', action='store_true',
                        default=False, help='assimilate and prep both main and overfit datasets')

    return parser




if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    if args.run_all:
        assimilate_data()
        prep_main_set(0.3)
        prep_overfit_set(0.3)
        prep_repo_split_set(0.25)
        prep_no_duplicates(0.3)
    else:
        if args.assimilate:
            assimilate_data()
        if args.unsplit_set:
            prep_main_set(0.3)
        if args.overfit_set:
            prep_overfit_set(0.3)
        if args.sep_repos:
            prep_repo_split_set(0.25)
        if args.no_dups:
            prep_no_duplicates(0.3)
    if not any(vars(args).values()):
        parser.print_help()
