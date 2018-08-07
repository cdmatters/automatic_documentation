#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-7
#$ -l tmem=10G
#$ -l h_rt=23:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/ehambro/EWEEZ/project/
export PYTHONPATH=.

#  LSTM SIZE
test $SGE_TASK_ID -eq 1 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=150 --bidirectional=0 --no-dup=1 --epochs=200 --dropout=0 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs --use-no-attention  --name=first_experiment__lstm150__hparam__b_128-b_0-c_70-c_60-d_200-d_0-e_100-l_150-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000 >> /home/ehambro/EWEEZ/project/qstat_logs/first_experiment/lstm-size_150___hparam__b_128-b_0-c_70-c_60-d_200-d_0-e_100-l_150-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000.log 2>&1

test $SGE_TASK_ID -eq 2 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=0 --no-dup=1 --epochs=200 --dropout=0 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs --use-no-attention  --name=first_experiment__lstm300__hparam__b_128-b_0-c_70-c_60-d_200-d_0-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000 >> /home/ehambro/EWEEZ/project/qstat_logs/first_experiment/lstm-size_300___hparam__b_128-b_0-c_70-c_60-d_200-d_0-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000.log 2>&1

# 300 + ATTENTION
test $SGE_TASK_ID -eq 3 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=0 --no-dup=1 --epochs=200 --dropout=0 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs  --name=first_experiment__with_attention___hparam__b_128-b_0-c_70-c_60-d_200-d_0-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000 >> /home/ehambro/EWEEZ/project/qstat_logs/first_experiment/with_attention__hparam__b_128-b_0-c_70-c_60-d_200-d_0-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000.log 2>&1

# 300 + ATTENTION + BIDIRECTIONAL
test $SGE_TASK_ID -eq 4 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=200 --dropout=0 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=first_experiment__bidirectional__hparam__b_128-b_1-c_70-c_60-d_200-d_0-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000 >> /home/ehambro/EWEEZ/project/qstat_logs/first_experiment/bidirectional__hparam__b_128-b_1-c_70-c_60-d_200-d_0-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000.log 2>&1

# 300 + ATTENTION + BIDIRECTIONAL + DROPOUT
test $SGE_TASK_ID -eq 5 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=200 --dropout=0.1 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=first_experiment__dropout0.1__hparam__b_128-b_1-c_70-c_60-d_200-d_0.1-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000 >> /home/ehambro/EWEEZ/project/qstat_logs/first_experiment/dropout0.1__hparam__b_128-b_1-c_70-c_60-d_200-d_0.1-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000.log 2>&1

test $SGE_TASK_ID -eq 6 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=200 --dropout=0.3 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=first_experiment__dropout0.3__hparam__b_128-b_1-c_70-c_60-d_200-d_0.3-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000 >> /home/ehambro/EWEEZ/project/qstat_logs/first_experiment/dropout0.3__hparam__b_128-b_1-c_70-c_60-d_200-d_0.3-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000.log 2>&1

test $SGE_TASK_ID -eq 7 && PYTHONPATH=. anaconda-python3-gpu -m project.models.char_baseline -F --char-seq=60 --vocab-size=40000  --desc-embed=200 --batch-size=128 --lstm-size=300 --bidirectional=1 --no-dup=1 --epochs=200 --dropout=0.5 --tokenizer=var_only  --save-every=3 --logdir=/home/ehambro/EWEEZ/project/logs   --name=first_experiment__dropout0.5__hparam__b_128-b_1-c_70-c_60-d_200-d_0.5-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000 >> /home/ehambro/EWEEZ/project/qstat_logs/first_experiment/dropout0.5__hparam__b_128-b_1-c_70-c_60-d_200-d_0.5-e_100-l_300-n_lstm-size__hparam__-n_1-s_5-t_var_only-v_40000.log 2>&1

