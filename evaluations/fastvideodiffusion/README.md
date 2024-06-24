## Dataset
download wevvid dataset for evaluation
```shell
bash evaluations/fastvideodiffusion/eval_webvid.sh
```

## Latte
You can edit `eval` config in `evaluations/fastvideodiffusion/configs/latte/sample_skip.yaml`

1. generate videos
```shell
bash evaluations/fastvideodiffusion/scripts/latte/generate_eval_latte_dataset.sh
```
The generated videos will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/latte/sample_skip.yaml`


2. eval
```shell
bash evaluations/fastvideodiffusion/scripts/latte/eval_latte.sh
```
The evaluation results will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/latte/sample_skip.yaml`





## OpenSore
You can edit `eval` config in `evaluations/fastvideodiffusion/configs/opensore/sample_skip.yaml`

1. generate videos
```shell
bash evaluations/fastvideodiffusion/scripts/opensore/generate_eval_opensore_dataset.sh
```
The generated videos will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensore/sample_skip.yaml`


2. eval
```shell
bash evaluations/fastvideodiffusion/scripts/opensore/eval_opensore.sh
```
The evaluation results will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensore/sample_skip.yaml`




## opensore_plan
You can edit `eval` config in `evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f_skip.yaml`

1. generate videos
```shell
bash evaluations/fastvideodiffusion/scripts/opensore_plan/generate_eval_opensore_plan_dataset.sh
```
The generated videos will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f_skip.yaml`


2. eval
```shell
bash evaluations/fastvideodiffusion/scripts/opensore_plan/eval_opensore_plan.sh
```
The evaluation results will be saved to `save_img_path` in `evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f_skip.yaml`



# TODO
1. eval code claim
2. how to edit config
