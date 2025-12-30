#!/bin/bash
# qwen_belle/check.sh - 环境检查脚本

cd /root/qwen_belle

echo "========================================"
echo "Qwen2-0.5B QLoRA 训练环境检查"
echo "========================================"

echo "1. GPU 检查"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA驱动: 已安装"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv
else
    echo "✗ NVIDIA驱动: 未找到"
fi

echo ""
echo "2. Python 环境检查"
echo "----------------------------------------"
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        reserved_gb = torch.cuda.memory_reserved(i) / 1024**3
        allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
        free_gb = total_gb - reserved_gb
        print(f'GPU {i}: {device_name}, 总内存: {total_gb:.1f}GB, 已用: {allocated_gb:.1f}GB, 预留: {reserved_gb:.1f}GB, 剩余: {free_gb:.1f}GB')
"

echo ""
echo "3. 依赖包检查"
echo "----------------------------------------"
python3 -c "
packages = {
    'transformers': '4.38.0',
    'peft': '0.9.0',
    'accelerate': '0.27.0',
    'bitsandbytes': '0.42.0',
    'torch': '2.1.2',
}

for pkg, expected in packages.items():
    try:
        import importlib.metadata as metadata
        version = metadata.version(pkg)
        if version.startswith(expected.split('.')[0]):
            print(f'✓ {pkg}: {version}')
        else:
            print(f'⚠ {pkg}: {version} (预期: {expected})')
    except:
        print(f'✗ {pkg}: 未安装')
"

echo ""
echo "4. 数据文件检查"
echo "----------------------------------------"
for file in "train_conversations.jsonl" "eval_conversations.jsonl"; do
    if [ -f "data/${file}" ]; then
        lines=$(wc -l < "data/${file}")
        size=$(du -h "data/${file}" | cut -f1)
        echo "✓ ${file}: ${lines} 行, ${size}"
        
        # 检查JSON格式
        python3 -c "
import json
try:
    with open('data/${file}', 'r') as f:
        first = json.loads(f.readline())
    if 'conversations' in first:
        print('  格式正确: 包含 conversations 字段')
    else:
        print('  格式错误: 缺少关键字段')
except Exception as e:
    print(f'  JSON错误: {str(e)[:50]}')
"
    else
        echo "✗ ${file}: 文件不存在"
    fi
done

echo ""
echo "5. 磁盘空间检查"
echo "----------------------------------------"
df -h /workspace | tail -1

echo ""
echo "6. 模型下载检查"
echo "----------------------------------------"
python3 -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2-0.5B-Instruct',  # 统一模型名称
        trust_remote_code=True,
        local_files_only=True
    )
    print('✓ 模型已缓存')
except:
    print('⚠ 模型未缓存，训练时会自动下载')
"

echo ""
echo "========================================"
echo "检查完成！"
echo "运行 ./train.sh 开始训练"
echo "========================================"
