#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Industrial-grade SFT training script
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', message='.*Unable to register.*')

# =====================================================
# Logging
# =====================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =====================================================
# Arguments
# =====================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HuggingFace 模型名或路径"})
    torch_dtype: Optional[str] = field(default="bfloat16")
    # 新增LoRA参数
    use_lora: bool = field(default=False, metadata={"help": "是否使用LoRA微调"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(default=None, metadata={"help": "LoRA目标模块，逗号分隔"})
    use_q_lora: bool = field(default=False, metadata={"help": "是否使用QLoRA (4-bit量化)"})
    q_lora_4bit_compute_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "QLoRA计算dtype"})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "训练数据 jsonl 路径"})
    max_length: int = field(default=2048)
    min_assistant_tokens: int = field(default=8)


# =====================================================
# Dataset
# =====================================================

class BelleSFTDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_length: int,
            min_assistant_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_assistant_tokens = min_assistant_tokens

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        logger.info(f"[Dataset] Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def _build_messages(self, conversations) -> List[Dict[str, str]]:
        messages = []
        for turn in conversations:
            content = turn.get("value", "").strip()
            if not content:
                continue
            if turn.get("from") == "human":
                messages.append({"role": "user", "content": content})
            elif turn.get("from") == "assistant":
                messages.append({"role": "assistant", "content": content})
        return messages

    def __getitem__(self, idx) -> Optional[Dict[str, List[int]]]:
        """返回列表格式，让collator处理张量转换"""
        try:
            example = self.data[idx]
            messages = self._build_messages(example.get("conversations", []))

            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                return None

            # 1. 分别获取prefix和full的文本
            prefix_messages = messages[:-1]
            prefix_text = self.tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # 2. Tokenize完整对话（允许截断）
            enc_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=False,
            )
            input_ids = enc_full["input_ids"]
            attention_mask = enc_full["attention_mask"]

            # 3. Tokenize prefix（不截断，用于计算位置）
            enc_prefix = self.tokenizer(
                prefix_text,
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )
            prefix_ids = enc_prefix["input_ids"]

            # 4. 安全计算assistant起始位置
            # 如果prefix本身已经超过max_length，直接跳过
            if len(prefix_ids) >= self.max_length:
                return None

            # assistant起始位置是prefix的长度
            assistant_start = len(prefix_ids)

            # 5. 对齐到截断后的序列
            assistant_start = min(assistant_start, len(input_ids))

            # 6. 验证assistant部分
            assistant_len = len(input_ids) - assistant_start
            if assistant_len < self.min_assistant_tokens:
                return None

            # 构建labels：排除assistant回复中的<|im_end|>等特殊token
            labels = input_ids.copy()
            labels[:assistant_start] = [-100] * assistant_start

            # 找到assistant部分中的<|im_end|>（eos_token）位置
            eos_token_id = self.tokenizer.eos_token_id

            # 在assistant部分中查找eos_token
            for i in range(assistant_start, len(input_ids)):
                if input_ids[i] == eos_token_id:
                    # 从eos_token开始，后面的token都设为-100（不训练）
                    # 这样模型只学习生成assistant的实际内容，不学习生成特殊token
                    for j in range(i, len(labels)):
                        labels[j] = -100
                    break

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {str(e)[:120]}")
            return None


# =====================================================
# Data Collator - 手动处理padding
# =====================================================

class SFTDataCollator:
    def __init__(self, tokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_label_id = -100  # padding位置的label值

    def __call__(self, features: List[Optional[Dict[str, List[int]]]]):
        # 过滤None
        features = [f for f in features if f is not None]

        if len(features) == 0:
            # 记录警告，但返回有效的小batch
            logger.warning("Empty batch detected, returning minimal batch")
            return {
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
            }

            # 确定batch的最大长度
        max_len_in_batch = max(len(f["input_ids"]) for f in features)
        if self.max_length:
            max_len_in_batch = min(max_len_in_batch, self.max_length)

        batch_size = len(features)

        # 初始化batch张量
        input_ids_batch = torch.full((batch_size, max_len_in_batch), self.pad_token_id, dtype=torch.long)
        attention_mask_batch = torch.zeros((batch_size, max_len_in_batch), dtype=torch.long)
        labels_batch = torch.full((batch_size, max_len_in_batch), self.pad_label_id, dtype=torch.long)

        # 填充每个样本
        for i, feature in enumerate(features):
            seq_len = min(len(feature["input_ids"]), max_len_in_batch)

            # 填充input_ids
            input_ids_batch[i, :seq_len] = torch.tensor(feature["input_ids"][:seq_len], dtype=torch.long)

            # 填充attention_mask
            attention_mask_batch[i, :seq_len] = 1

            # 填充labels
            labels_batch[i, :seq_len] = torch.tensor(feature["labels"][:seq_len], dtype=torch.long)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }


# =====================================================
# Model / Tokenizer
# =====================================================

def load_model_and_tokenizer(model_name: str, torch_dtype: Optional[str], **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # 转换torch_dtype
    dtype = None
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32

    # 从kwargs获取LoRA相关参数
    use_lora = kwargs.get('use_lora', False)
    use_q_lora = kwargs.get('use_q_lora', False)
    q_lora_4bit_compute_dtype = kwargs.get('q_lora_4bit_compute_dtype', 'bfloat16')

    # 处理QLoRA的计算dtype
    compute_dtype = None
    if q_lora_4bit_compute_dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    elif q_lora_4bit_compute_dtype == "float16":
        compute_dtype = torch.float16
    elif q_lora_4bit_compute_dtype == "float32":
        compute_dtype = torch.float32

    # QLoRA处理
    if use_q_lora:
        logger.info("Using QLoRA (4-bit quantization)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": 0},  # 单卡A10明确指定
            torch_dtype=compute_dtype,
        )

        # 为k-bit训练准备模型
        model = prepare_model_for_kbit_training(model)

    else:
        # 原有逻辑保持不变
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )

    # LoRA处理
    if use_lora or use_q_lora:
        # 获取LoRA参数
        lora_r = kwargs.get('lora_r', 16)
        lora_alpha = kwargs.get('lora_alpha', 32)
        lora_dropout = kwargs.get('lora_dropout', 0.1)
        lora_target_modules = kwargs.get('lora_target_modules', None)

        # 解析目标模块
        if lora_target_modules:
            target_modules = [m.strip() for m in lora_target_modules.split(",")]
        else:
            # 默认目标模块（适用于大多数LLaMA架构）
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        logger.info(f"Applying LoRA with target modules: {target_modules}")

        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 验证LoRA是否正确应用
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        if trainable_params == 0:
            logger.error("❌ ERROR: No trainable parameters found after applying LoRA!")
            logger.error("This means LoRA was not applied correctly.")

            # 手动检查并修复
            logger.info("Manually checking and fixing LoRA layers...")
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                    logger.info(f"  Set {name} as trainable")

        # 确保模型在训练模式
        model.train()
        # logger.info(f"Model set to training mode: {model.training}")

    return model, tokenizer


# =====================================================
# Main
# =====================================================

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    # 传递LoRA相关参数
    model, tokenizer = load_model_and_tokenizer(
        model_args.model_name_or_path,
        model_args.torch_dtype or "bfloat16",
        use_lora=model_args.use_lora,
        use_q_lora=model_args.use_q_lora,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_target_modules=model_args.lora_target_modules,
        q_lora_4bit_compute_dtype=model_args.q_lora_4bit_compute_dtype,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    train_dataset = BelleSFTDataset(
        data_path=data_args.train_file,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        min_assistant_tokens=data_args.min_assistant_tokens,
    )

    data_collator = SFTDataCollator(tokenizer, max_length=data_args.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    
    )

    logger.info("Starting training...")

    # 添加一些调试信息
    logger.info(f"Dataset size: {len(train_dataset)}")

    # 测试几个样本
    logger.info("Testing first few samples...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        if sample is not None:
            logger.info(
                f"Sample {i}: length={len(sample['input_ids'])}, assistant_tokens={(sum(1 for x in sample['labels'] if x != -100))}")

    train_result = trainer.train()

    # LoRA模型保存
    if model_args.use_lora or model_args.use_q_lora:
        # 保存LoRA权重
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"LoRA weights saved to {training_args.output_dir}")
    else:
        pass
        # trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"Training completed! Final loss: {metrics.get('train_loss', 'N/A')}")


if __name__ == "__main__":
    main()
