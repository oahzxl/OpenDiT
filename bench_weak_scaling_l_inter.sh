WARMUP=1
RUNTIME=1
MODEL_TYPE="3B"

HYPE_LIST=(
    "(2 128 1024 1024 16)"
    "(1 128 1024 1024 8)"
    # "(8 128 1024 1024 64)"
    "(4 128 1024 1024 32)"
)

sp_list=("ring")

for hype in "${HYPE_LIST[@]}"
do
    read -r BATCH_SIZE NUM_FRAMES H W SP_SIZE <<<"${hype//[()]/}"
    for SP in ${sp_list[@]}
    do
        GPU_NUM=$SP_SIZE
        bash scripts/opensora/bench_opensora_srun.sh xx $GPU_NUM test_ds $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE $MODEL_TYPE
    done
done
