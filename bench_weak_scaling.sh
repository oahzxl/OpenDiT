WARMUP=3
RUNTIME=2

HYPE_LIST=(
    "(1 512 720 1280 8)" # 2048k
    "(2 512 720 1280 16)"
    "(4 512 720 1280 32)"
    "(8 512 720 1280 64)"
)

sp_list=("dsp" "megatron" "ulysses")

for hype in "${HYPE_LIST[@]}"
do
    read -r NUM_FRAMES H W SP_SIZE <<<"${hype//[()]/}"
    for SP in ${sp_list[@]}
    do
        GPU_NUM=$SP_SIZE
        bash scripts/opensora/bench_opensora_srun.sh xx $GPU_NUM test_ds $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE
    done
done
