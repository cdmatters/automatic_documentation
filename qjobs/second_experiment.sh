#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-4
#$ -l tmem=10G
#$ -l h_rt=16:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/ehambro/EWEEZ/project/
export PYTHONPATH=.



# 300 + ATTENTION + BIDIRECTIONAL + DROPOUT
test $SGE_TASK_ID -eq 1 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=120 --vocab-size=40000 --char-embed=200 --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_only  --save-every=5 --logdir=/home/ehambro/EWEEZ/project/logs   --name=second_experiment__var_only >> /home/ehambro/EWEEZ/project/qstat_logs/second_experiment/var_only.log 2>&1

test $SGE_TASK_ID -eq 2 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=120 --vocab-size=40000 --char-embed=200 --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_funcname  --save-every=5 --logdir=/home/ehambro/EWEEZ/project/logs   --name=second_experiment__var_funcname >> /home/ehambro/EWEEZ/project/qstat_logs/second_experiment/var_funcname.log 2>&1

test $SGE_TASK_ID -eq 3 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=120 --vocab-size=40000 --char-embed=200 --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_otherargs  --save-every=5 --logdir=/home/ehambro/EWEEZ/project/logs   --name=second_experiment__var_otherargs >> /home/ehambro/EWEEZ/project/qstat_logs/second_experiment/var_otherargs.log 2>&1

test $SGE_TASK_ID -eq 4 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=120 --vocab-size=40000 --char-embed=200 --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_funcname_otherargs  --save-every=5 --logdir=/home/ehambro/EWEEZ/project/logs   --name=second_experiment__var_funcname_otherargs >> /home/ehambro/EWEEZ/project/qstat_logs/second_experiment/var_funcname_otherargs.log 2>&1


