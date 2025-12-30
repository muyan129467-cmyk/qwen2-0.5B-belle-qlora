#!/bin/bash
# qwen_belle/install.sh

echo "=== 安装依赖 ==="

# 更新系统
apt-get update
apt-get install -y python3-pip git gcc g++

# 升级 pip
python3 -m pip install --upgrade pip

# 检查CUDA版本
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
else
    CUDA_VERSION="11.8"  # 默认值
fi
echo "CUDA Version detected: $CUDA_VERSION"

# 安装torch及相关库（修复：兼容 RTX 4090 / CUDA 12.x）
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" || "$CUDA_VERSION" == "12.2" ]]; then
    # PyTorch 2.1.2 只支持 cu121 官方 whl，12.2 fallback 到 cu121
    echo "Using cu121 wheels for CUDA $CUDA_VERSION"
    pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu121
else
    echo "Warning: Unsupported CUDA version ($CUDA_VERSION), using CUDA 11.8"
    pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu118
fi

# 安装bitsandbytes（修复：使用预编译 CUDA 12+ wheel，兼容 RTX 4090）
pip3 install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.42.0-py3-none-linux_x86_64.whl

# 安装其他依赖
pip3 install transformers==4.38.0 accelerate==0.27.0 peft==0.9.0 \
  datasets==2.17.0 sentencepiece==0.2.0 protobuf==4.25.3 \
  einops==0.7.0 tqdm==4.66.1 scipy==1.11.4 numpy==1.24.4 \
  huggingface-hub==0.20.3 safetensors==0.4.2

# 默认 accelerate 配置（可选）
accelerate config default

echo "=== 验证安装 ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
python3 -c "import bitsandbytes as bnb; print(f'BitsAndBytes: {bnb.__version__}')"
