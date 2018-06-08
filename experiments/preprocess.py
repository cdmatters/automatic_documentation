from data import preprocessed, data

import pyaml
import yaml
from tqdm import tqdm

import os
import random


RAWDATADIR = os.path.dirname(os.path.abspath(data.__file__))
PREPROCESSDATADIR = os.path.dirname(os.path.abspath(preprocessed.__file__))

def assimilate_data():
    '''Clean and assimilate data into big yaml files (not necessarily human readable)'''
    types = ['full', 'short']
    for t in types:
        all_data = {}
        all_files = {}
        tot_f, tot_args = 0, 0
        for i, yaml_file in enumerate(os.listdir(RAWDATADIR + "/" + t + "/")):
            with open(RAWDATADIR + "/{}/{}".format(t, yaml_file), "r", encoding='utf-8') as f:
                string = "\n".join(f.readlines())
                data = yaml.load(string.replace(
                    "            desc: `", "            desc: \\`").replace(
                    "            type: `", "            type: \\`"))
                
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
            f.write(yaml.dump(all_data))

    for t in types:
        print("Assimilated, now test loading...")
        with open(PREPROCESSDATADIR+"/all_{}.yaml".format(t), "r", encoding='utf-8') as f:
            data = yaml.load(f) 
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

    print(count, desc_count)
    return args

def prep_main_set(test_percentage):
    with open(PREPROCESSDATADIR+"/all_full.yaml", "r") as f:
        data = yaml.load(f)

    overfit_data = map_yaml_to_arg_list(data)
    random.shuffle(overfit_data)

    n = len(overfit_data)
    print(n)
    test = overfit_data[:int(n * test_percentage)]
    train = overfit_data[int(n * test_percentage):]

    with open(PREPROCESSDATADIR+"/main_train.yaml", 'w') as f:
        f.write(yaml.dump(train))
    with open(PREPROCESSDATADIR+"/main_test.yaml", 'w') as f:
        f.write(yaml.dump(test))

def prep_overfit_set(test_percentage):
    '''Prepare a tiny dataset from the raw data, to test overfit.'''
    with open(PREPROCESSDATADIR+"/all_full.yaml", "r") as f:
        data = yaml.load(f)

    overfit_set = {}
    for k,v in data.items():
        if v['filename'].startswith("/numpy/"):
            overfit_set[k] = v
    
    overfit_data = map_yaml_to_arg_list(overfit_set)
    random.shuffle(overfit_data)

    n = len(overfit_data)
    print(n)
    test = overfit_data[:int(n * test_percentage)]
    train = overfit_data[int(n * test_percentage):]

    with open(PREPROCESSDATADIR+"/overfit/train.yaml", 'w') as f:
        f.write(yaml.dump(train))
    with open(PREPROCESSDATADIR+"/overfit/test.yaml", 'w') as f:
        f.write(yaml.dump(test))
        

if __name__ == "__main__":
    # assimilate_data()
    # prep_main_set(0.3)
    prep_overfit_set(0.3)
  