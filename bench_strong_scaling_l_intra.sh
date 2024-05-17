WARMUP=4
RUNTIME=4
MODEL_TYPE="3B"

HYPE_LIST=(
    "(1 32 1024 1024 1)"
    "(1 32 1024 1024 2)"
    "(1 32 1024 1024 4)"
    "(1 32 1024 1024 8)"
)

sp_list=("dsp" "megatron" "ulysses")

for hype in "${HYPE_LIST[@]}"
do
    read -r BATCH_SIZE NUM_FRAMES H W SP_SIZE <<<"${hype//[()]/}"
    for SP in ${sp_list[@]}
    do
        GPU_NUM=$SP_SIZE
        bash scripts/opensora/bench_opensora_local.sh xx $GPU_NUM test_ds $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE $MODEL_TYPE
    done
done
