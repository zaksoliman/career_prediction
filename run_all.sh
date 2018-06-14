#!/bin/bash

python run_models.py -ds top550 -m simple_rnn -r bow -t test
python run_models.py -ds top550 -m simple_rnn -r jobid -t test
python run_models.py -ds top550 -m simple_rnn -r fasttext -t test
#python run_skill_model.py -ds top550 -t test
#python  multi_label_models.py top550
#
#python run_models.py -ds reduced7000 -m simple_rnn -r bow -t test
#python run_models.py -ds reduced7000 -m simple_rnn -r jobid -t test
#python run_models.py -ds reduced7000 -m simple_rnn -r fasttext -t test
#python run_skill_model.py -ds reduced7000 -t test
#python multi_label_models.py reduced7k
