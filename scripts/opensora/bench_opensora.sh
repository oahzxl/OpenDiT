BATCH_SIZE=1
WARMUP=10
RUNTIME=10
NUM_FRAMES=16
H=512
W=512
SP="dsp"
SP_SIZE=2

mkdir -p log

torchrun --standalone --nproc_per_node=8 scripts/opensora/bench_opensora.py \
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
    --sequence_parallel_size $SP_SIZE | tee log/BATCHSIZE-${BATCH_SIZE}_NUMFRAMES-${NUM_FRAMES}_H-${H}_W-${W}_SPSIZE-${SP_SIZE}_METHOD-${SP}.log
