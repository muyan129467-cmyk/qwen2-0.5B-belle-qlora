#!/bin/bash
# qwen_belle/run_all.sh

cd /root/qwen_belle || exit 1

echo "=== Qwen2-0.5B QLoRA 全流程 ==="

chmod +x install.sh
./install.sh

echo "GPU 状态:"
nvidia-smi

echo "数据检查..."
python3 - << 'EOF'
import json, os
for name in ["train_conversations.jsonl", "eval_conversations.jsonl"]:
    path = f"data/{name}"
    if not os.path.exists(path):
        raise SystemExit(f"❌ 缺少 {path}")
    with open(path) as f:
        json.loads(f.readline())
    print(f"✓ {name} OK")
EOF

read -p "是否开始训练？(y/n): " ans
ans=${ans:-n}
[[ "$ans" != "y" ]] && exit 0


START=$(date +%s)

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
  --do_train 2>&1 | tee train_output.log

END=$(date +%s)
echo "训练完成，用时 $(( (END-START)/60 )) 分钟"

echo "开始自动验证..."
python eval_sft.py \
  --model_path "./output/qwen2-0.5b-lora" \
  --eval_file "data/eval_conversations.jsonl" \
  --max_length 1024 \
  --batch_size 4 \
  --use_qlora 2>&1 | tee eval_output.log
