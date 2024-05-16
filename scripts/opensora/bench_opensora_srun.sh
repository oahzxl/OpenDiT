#!/bin/bash

WARMUP=$4
RUNTIME=$5
BATCH_SIZE=$6
NUM_FRAMES=$7
H=$8
W=$9
SP=${10}
SP_SIZE=${11}

source bash_slurm_env.sh

mkdir -p log

echo "run WARMUP=$WARMUP RUNTIME=$RUNTIME BATCH_SIZE=$BATCH_SIZE NUM_FRAMES=$NUM_FRAMES H=$H W=$W SP=$SP SP_SIZE=$SP_SIZE"
srun -N${NUM_NODES} --gres=gpu:${GRES} --ntasks-per-node=1 --cpus-per-task=${CPUS} --job-name=$3 --mem=0 \
python $slurm_launcher --script scripts/opensora/bench_opensora.py --main_process_port 34088 \
    --batch_size $BATCH_SIZE \
    --mixed_precision bf16 \
    --grad_checkpoint \
    --data_path "./videos/demo.csv" \
    --text_speedup \
    --enable_flashattn \
    --enable_layernorm_kernel \
    --warmup $WARMUP \
    --runtime $RUNTIME \
    --num_frames $NUM_FRAMES \
    --image_size $H $W \
    --sp $SP \
    --sequence_parallel_size $SP_SIZE \
    > >(tee log/batch${BATCH_SIZE}_f${NUM_FRAMES}_h${H}_w${W}_sp${SP_SIZE}_${SP}.log) 2> >(tee log/batch${BATCH_SIZE}_f${NUM_FRAMES}_h${H}_w${W}_sp${SP_SIZE}_${SP}-error.log >&2)
