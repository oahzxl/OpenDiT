WARMUP=20
RUNTIME=20
BATCH_SIZE=1

num_frame_list=(16 32 64 128 256 512 1024 2048 4096 8192 16384)
HW_list=(512 1024 2048 4096 8192 16384)
gpu_list=(1 2 4 8)
sp_list=("dsp" "ulysses" "megatron")

for NUM_FRAMES in ${num_frame_list[@]}
do
    for H in ${HW_list[@]}
    do
        W=$H
        for SP in ${sp_list[@]}
        do
            for SP_SIZE in ${gpu_list[@]}
            do
                GPUNUM=$SP_SIZE
                echo "run WARMUP=$WARMUP RUNTIME=$RUNTIME BATCH_SIZE=$BATCH_SIZE NUM_FRAMES=$NUM_FRAMES H=$H W=$W SP=$SP SP_SIZE=$SP_SIZE GPUNUM=$GPUNUM"
                bash scripts/opensora/bench_opensora.sh $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE $GPUNUM
            done
        done
    done
done
