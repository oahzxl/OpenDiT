WARMUP=1
RUNTIME=1
MODEL_TYPE="720M"

HYPE_LIST=(
    "(8 64 1024 1024 8)"
    "(4 64 1024 1024 4)"
    "(2 64 1024 1024 2)"
    "(1 64 1024 1024 1)"
)

sp_list=("dsp" "megatron" "ulysses")

for hype in "${HYPE_LIST[@]}"
do
    read -r BATCH_SIZE NUM_FRAMES H W SP_SIZE <<<"${hype//[()]/}"
    for SP in ${sp_list[@]}
    do
        GPU_NUM=$SP_SIZE
        bash scripts/opensora/bench_opensora_srun.sh xx $GPU_NUM test_ds $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE $MODEL_TYPE
    done
done
