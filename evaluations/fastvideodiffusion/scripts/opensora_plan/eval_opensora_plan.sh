GPU_ID="3"

# origin
# CUDA_VISIBLE_DEVICES=$GPU_ID python evaluations/fastvideodiffusion/scripts/eval.py \
#     --calculate_lpips \
#     --calculate_psnr \
#     --calculate_ssim \
#     --eval_method "videogpt" \
#     --eval_dataset "./evaluations/fastvideodiffusion/datasets/webvid_selected.csv" \
#     --eval_video_dir "./evaluations/fastvideodiffusion/datasets/webvid" \
#     --generated_video_dir "./evaluations/fastvideodiffusion/samples/opensora_plan/sample_65f"

# pab
# CUDA_VISIBLE_DEVICES=$GPU_ID python evaluations/fastvideodiffusion/scripts/eval.py \
#     --calculate_lpips \
#     --calculate_psnr \
#     --calculate_ssim \
#     --eval_method "videogpt" \
#     --eval_dataset "./evaluations/fastvideodiffusion/datasets/webvid_selected.csv" \
#     --eval_video_dir "./evaluations/fastvideodiffusion/datasets/webvid" \
#     --generated_video_dir "./evaluations/fastvideodiffusion/samples/opensora_plan/sample_65f_pab"
