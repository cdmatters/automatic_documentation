#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-8
#$ -l tmem=10G
#$ -l h_rt=16:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/ehambro/EWEEZ/project/
export PYTHONPATH=.

# CODE2VECS
test $SGE_TASK_ID -eq 1 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F -S --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=150 --dropout=0.1 --tokenizer=var_only  --code-tokenizer=code2vec --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=seven_code2vec_solo >> /home/ehambro/EWEEZ/project/qstat_logs/seven_code2vec_solo.log 2>&1

test $SGE_TASK_ID -eq 2 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F -S --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=150 --dropout=0.1 --tokenizer=var_only  --code-tokenizer=code2vec_mask_args --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=seven_code2vec_solo_mask_args >> /home/ehambro/EWEEZ/project/qstat_logs/seven_code2vec_solo_mask_args.log  2>&1

test $SGE_TASK_ID -eq 3 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F -S --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=150 --dropout=0.1 --tokenizer=var_only  --code-tokenizer=code2vec_mask_all --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=seven_code2vec_solo_mask_all >> /home/ehambro/EWEEZ/project/qstat_logs/seven_code2vec_solo_mask_all.log 2>&1

# TOKENIZERS
test $SGE_TASK_ID -eq 4 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F -S --char-seq=120 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=seven_char_baseline_var_only >> /home/ehambro/EWEEZ/project/qstat_logs/seven_char_baseline_var_only.log 2>&1

test $SGE_TASK_ID -eq 5 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F -S --char-seq=120 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_funcname  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=seven_char_baseline_var_funcname >> /home/ehambro/EWEEZ/project/qstat_logs/seven_char_baseline_var_funcname.log 2>&1

test $SGE_TASK_ID -eq 6 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F -S --char-seq=120 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_otherargs  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=seven_char_baseline_var_otherargss >> /home/ehambro/EWEEZ/project/qstat_logs/seven_char_baseline_var_otherargs.log 2>&1

test $SGE_TASK_ID -eq 7 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F -S --char-seq=120 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=150 --dropout=0.1 --tokenizer=var_funcname_otherargs  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=seven_char_baseline_var_funcname_otherargs >> /home/ehambro/EWEEZ/project/qstat_logs/seven_char_baseline_var_funcname_otherargs.log 2>&1

# CODE2VEC + NAME ONLY
test $SGE_TASK_ID -eq 8 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_encoder -F -S --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=100 --dropout=0.1 --tokenizer=var_only  --code-tokenizer=code2vec --save-every=3 --logdir=logs   --name=seven_code2vec_encoder  >> /home/ehambro/EWEEZ/project/qstat_logs/seven_code2vec_encoder.log 2>&1

