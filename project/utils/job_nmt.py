#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os
import json

import sys
from datetime import datetime


COMMAND = '''PYTHONPATH=. anaconda-python3-gpu -m nmt.nmt  {args} '''


def cartesian_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def to_name(configuration):
    kvs = sorted([(k, v)
                  for k, v in configuration.items()], key=lambda e: e[0])
    name = configuration.get("name", "auto_")
    return name + "-" + '-'.join([('{}_{}'.format(k[:1], v)) for (k, v) in kvs if k not in ["name", "base"]])



def to_cmd(**kwargs):
    DIR = os.path.dirname(os.path.abspath(__file__))

    with open("{}/base_nmt.json".format(DIR), 'r') as f:
        default_args = json.load(f)

    default_args.update(kwargs.items())
    default_args.update(to_data_dirs(**kwargs).items())

    arg_list = " ".join(["--{}={}".format(k, v)
                         for k, v in default_args.items()])
    return COMMAND.format(args=arg_list)

def to_data_dirs(base, tokenizer, desc_embed, split, vocab_size, **kwargs):
    sp = 'split' if split else 'unsplit'
    data = "nmt_data/{}_{}_{}"
    return {
        "embed_prefix": base.format("nmt_data/glove_inserted.{}.txt".format(desc_embed)),
        "vocab_prefix": base.format("nmt_data/vocab_{}_{}".format(sp, vocab_size)),
        "train_prefix": base.format(data.format('train', tokenizer, sp)),
        "dev_prefix": base.format(data.format('valid', tokenizer, sp)),
        "test_prefix": base.format(data.format('test', tokenizer, sp)),
    }

def main(_):
    now = datetime.strftime(datetime.now(), '%d%m_%H%M%S')

    log_path = '/home/ehambro/EWEEZ/nmt/logs/'
    qstat_logs = "/home/ehambro/EWEEZ/nmt/qstat_logs/{}".format(now)


    hyperparameters_space = dict(

        # char_seq=[600],
        # char_embed=[100],
        # batch_size=[128],
        # lstm_size=[128],
        # bidirectional=[True],
        vocab_size=[25000],
        num_train_steps= [15000],
        desc_embed=[200],
        split=[True],
        base=['/home/ehambro/EWEEZ/nmt/{}'],
        tokenizer=['var_only', 'var_funcname', 'var_otherargs', 'var_funcname_otherargs'],
        name=['comp_tokenizer_split'],
        # save_every=[-1],
        # logdir=[log_path]
    )

    configurations = cartesian_product(hyperparameters_space)

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/ehambro/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(qstat_logs):
            os.makedirs(qstat_logs)

    configurations = list(configurations)

    command_lines = set()
    for cfg in configurations:
        name = to_name(cfg)

        command_line = '{} --out_dir={} >> {}/{}.log 2>&1'.format(
            to_cmd(**cfg), log_path + "/" + name + "_" + now, qstat_logs, name)
        command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l tmem=11G
#$ -l h_rt=20:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/ehambro/EWEEZ/nmt/
export PYTHONPATH=.

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}\n'.format(job_id, command_line))


if __name__ == '__main__':
    main(sys.argv[1:])
