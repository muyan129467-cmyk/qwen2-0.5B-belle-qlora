#!/bin/bash
# qwen_belle/eval.sh

cd /root/qwen_belle || exit 1

echo "=== 验证 QLoRA 模型 ==="

if [ ! -f "./output/qwen2-0.5b-lora/adapter_config.json" ]; then
    echo "❌ 未找到 adapter_config.json，请先训练"
    exit 1
fi

if [ ! -f "data/eval_conversations.jsonl" ]; then
    echo "❌ 验证数据不存在"
    exit 1
fi

python eval_sft.py \
  --model_path "./output/qwen2-0.5b-lora" \
  --eval_file "data/eval_conversations.jsonl" \
  --max_length 1024 \
  --batch_size 4 \
  --use_qlora

echo "验证完成"