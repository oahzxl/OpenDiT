WARMUP=1
RUNTIME=1
BATCH_SIZE=1

MODEL_TYPE="3B"

# frame, hw, sp_size
HYPE_LIST=(
    "(128 1024 1024 2)"
    "(128 1024 1024 4)"
    "(256 1024 1024 4)"
    "(256 1024 1024 8)"
    "(256 1024 1024 16)"
    "(512 1024 1024 8)"
    "(512 1024 1024 16)"
    "(512 1024 1024 32)"
    "(1024 1024 1024 16)"
    "(1024 1024 1024 32)"
    "(1024 1024 1024 64)"
)

GPU_NUM=128

sp_list=("dsp" "megatron" "ulysses")

for hype in "${HYPE_LIST[@]}"
do
    read -r NUM_FRAMES H W SP_SIZE <<<"${hype//[()]/}"
    for SP in ${sp_list[@]}
    do
        bash scripts/opensora/bench_opensora_srun.sh xx $GPU_NUM test_ds $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE $MODEL_TYPE
    done
done
