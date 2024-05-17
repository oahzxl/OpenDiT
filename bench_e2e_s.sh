WARMUP=2
RUNTIME=2
BATCH_SIZE=1

MODEL_TYPE="720M"

# frame, hw, sp_size
HYPE_LIST=(
    "(128 1024 1024 2)" # 512K
    "(128 1024 1024 4)" # 512K
    "(256 1024 1024 4)" # 1024k
    "(256 1024 1024 8)" # 1024k
    "(512 1024 1024 8)" # 2048k
    "(512 1024 1024 16)" # 2048k
    "(1024 1024 1024 16)"
    "(1024 1024 1024 32)"
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
