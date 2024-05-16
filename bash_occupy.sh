#!/bin/bash

source bash_slurm_env.sh

mkdir -p logs

srun --gres=gpu:8 --cpus-per-task=112 --ntasks=1 --pty bash
