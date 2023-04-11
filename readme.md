# Instruct-finetuning Flan models with LoRA
**F**lan finetuned by **LoRA**: This repo is based on [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), and served as the minimum codebase to instruct finetune [FLAN](https://huggingface.co/docs/transformers/model_doc/flan-t5) models using [LoRA](https://arxiv.org/pdf/2106.09685.pdf) and Alpaca's instruction datasets.

- `alpaca_data.json`: Downloaded from https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json.
- `requirements.txt`: Python depdendencies
- `train.py`: training script.
- `utils.py`: utility script.

Example training script:
```
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
    --model_name_or_path google/flan-t5-small \
    --data_path ./alpaca_data.json \
    --bf16 False \
    --output_dir ./output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False
```

# Notes
https://github.com/huggingface/peft#caveats
