#!/bin/bash
# qwen_belle/resume_train.sh

cd /root/qwen_belle || exit 1

LAST_CHECKPOINT=$(ls -td ./output/qwen2-0.5b-lora/checkpoint-* | head -1)

if [ -z "$LAST_CHECKPOINT" ]; then
    echo "❌ 未找到 checkpoint"
    exit 1
fi

echo "从 $LAST_CHECKPOINT 恢复训练"

python train_sft.py \
  --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \
  --torch_dtype "bfloat16" \
  --use_lora true \
  --use_q_lora true \
  --q_lora_4bit_compute_dtype "bfloat16" \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.1 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --train_file "data/train_conversations.jsonl" \
  --max_length 1024 \
  --min_assistant_tokens 8 \
  --output_dir "./output/qwen2-0.5b-lora" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 5 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --gradient_checkpointing true \
  --bf16 true \
  --remove_unused_columns false \
  --report_to "none" \
  --ddp_find_unused_parameters false \
  --do_train \
  --resume_from_checkpoint "$LAST_CHECKPOINT"