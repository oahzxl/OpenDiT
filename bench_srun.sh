WARMUP=20
RUNTIME=20
BATCH_SIZE=1

# frame, hw, sp_size
HYPE_LIST=(
    "(128 720 1280 2)" # 512K
    "(256 720 1280 4)" # 1024k
    "(512 720 1280 8)" # 2048k
    "(1024 720 1280 16)"
)
GPU_NUM=64
# sp_list=("dsp" "megatron" "ulysses")
sp_list=("dsp")

for hype in "${HYPE_LIST[@]}"
do
    read -r NUM_FRAMES H W SP_SIZE <<<"${hype//[()]/}"
    for SP in ${sp_list[@]}
    do
        bash scripts/opensora/bench_opensora_srun.sh xx $GPU_NUM test_ds $WARMUP $RUNTIME $BATCH_SIZE $NUM_FRAMES $H $W $SP $SP_SIZE
    done
done
