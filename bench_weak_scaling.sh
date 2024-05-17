WARMUP=1
RUNTIME=1

HYPE_LIST=(
    "(8 256 1024 1024 64)"
    "(4 256 1024 1024 32)"
    "(2 256 1024 1024 16)"
    "(1 256 1024 1024 8)" # 2048k
)

sp_list=("dsp" "megatron" "ulysses")

for hype in "${HYPE_LIST[@]}"
do
    read -r BATCH_SIZE NUM_FRAMES H W SP_SIZE <<<"${hype//[()]/}"
    for SP in ${sp_list[@]}
    do
        GPU_NUM=$SP_SIZE
        bash scripts/opensora/bench_opensora_srun.sh xx $GPU_NUM test_ds $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE
    done
done
