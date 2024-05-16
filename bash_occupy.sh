#!/bin/bash

source bash_slurm_env.sh

mkdir -p logs

srun -N${NUM_NODES} --gres=gpu:${GRES} --ntasks-per-node=1 --cpus-per-task=${CPUS} --job-name=$3 --pty bash
