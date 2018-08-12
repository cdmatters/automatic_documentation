#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-6
#$ -l tmem=10G
#$ -l h_rt=12:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/ehambro/EWEEZ/project/
export PYTHONPATH=.

test $SGE_TASK_ID -eq 1 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --no-dup=10 --epochs=150 --dropout=0.1 --tokenizer=var_only  --save-every=3 --bidirectional=1 --desc-seq=120 --logdir=/home/ehambro/EWEEZ/project/logs  --name=fourth_fifth_experiment__name_only >> /home/ehambro/EWEEZ/project/qstat_logs/fourth_fifth_experiment__name_only/

test $SGE_TASK_ID -eq 2 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=100 --dropout=0.0 --tokenizer=var_only  --code-tokenizer=code2vec --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=fourth_fifth_experiment_solo_canon_d0.0 >> /home/ehambro/EWEEZ/project/qstat_logs/fourth_fifth_experiment_solo_canon_d0.0.log 2>&1

test $SGE_TASK_ID -eq 3 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=100 --dropout=0.1 --tokenizer=var_only  --code-tokenizer=code2vec --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=fourth_fifth_experiment_solo_canon_d0.1 >> /home/ehambro/EWEEZ/project/qstat_logs/fourth_fifth_experiment_solo_canon_d0.1.log 2>&1

test $SGE_TASK_ID -eq 4 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=100 --dropout=0.3 --tokenizer=var_only  --code-tokenizer=code2vec --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=fourth_fifth_experiment_solo_canon_d0.3 >> /home/ehambro/EWEEZ/project/qstat_logs/fourth_fifth_experiment_solo_canon_d0.3.log 2>&1

test $SGE_TASK_ID -eq 5 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=100 --dropout=0.1 --tokenizer=var_only  --code-tokenizer=code2vec_mask_args --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=fourth_fifth_experiment_solo_mask_args_d0.1 >> /home/ehambro/EWEEZ/project/qstat_logs/fourth_fifth_experiment_solo_mask_args_d0.1.log 2>&1

test $SGE_TASK_ID -eq 6 && PYTHONPATH=. anaconda-python3-gpu -m project.models.code2vec_solo -F --char-seq=60 --vocab-size=40000 --desc-embed=200 --batch-size=64 --path-embed=300 --path-vocab=15000 --path-seq=5000 --code2vec-size=300 --lstm-size=300 --bidirectional=1 --no-dup=10 --epochs=100 --dropout=0.1 --tokenizer=var_only  --code-tokenizer=code2vec_mask_all --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=fourth_fifth_experiment_solo_mask_all_d0.1 >> /home/ehambro/EWEEZ/project/qstat_logs/fourth_fifth_experiment_solo_mask_all_d0.1.log 2>&1
