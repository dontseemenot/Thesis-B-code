#!/bin/bash

#PBS -l select=1:ncpus=32:ngpus=4:mem=184gb,walltime=12:00:00
#PBS -l walltime=12:00:00
#PBS -M z5165205@unsw.edu.au
#PBS -m ae
#PBS -o /srv/scratch/z5165205/Thesis/job_output/


cd /srv/scratch/z5165205/Thesis
module unload cuda
module load cuda/11.3
source thesis_env/bin/activate 
python train_test_main.py "Katana hyperparameter Berlin filter2" "Berlin" "Berlin_no_overlap_filter2" 2>&1 | tee -a "Katana hyperparameter Berlin filter2.txt"