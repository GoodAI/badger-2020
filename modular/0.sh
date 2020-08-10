#!/bin/bash

#python coral/baseline_experiment.py with lr=0.00002 policy=FfCommNet matrix=robot p_branching=0.5
#python coral/baseline_experiment.py with lr=0.00002 policy=FfCommNet matrix=None p_branching=0.5
#python coral/baseline_experiment.py with lr=0.0001 policy=RnnCommNet matrix=robot p_branching=0.5
#python coral/baseline_experiment.py with lr=0.0001 policy=RnnCommNet matrix=None p_branching=0.5
#python coral/baseline_experiment.py with lr=0.0001 policy=LstmCommNet matrix=robot num_passes=1 p_branching=0.5
#python coral/baseline_experiment.py with lr=0.0001 policy=LstmCommNet matrix=None num_passes=1 p_branching=0.5
#
python coral/baseline_experiment.py with policy=LstmCommNetCoordinated ep_len=238 loss_last_n=70 p_branching=0.5 use_net=False binary_threshold=False
