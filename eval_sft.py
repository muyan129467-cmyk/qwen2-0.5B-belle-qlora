#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone evaluation script for SFT LoRA / QLoRA models
(Belle-style conversations)
"""

import json
import math
import argparse
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm


# =====================================================
# Dataset（保持不变）
# =====================================================

class BelleEvalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int,
        min_assistant_tokens: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_assistant_tokens = min_assistant_tokens

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"[EvalDataset] Loaded {len(self.data)} samples from {data_path}")

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
        example = self.data[idx]
        messages = self._build_messages(example.get("conversations", []))

        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            return None

        prefix_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        enc_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        enc_prefix = self.tokenizer(
            prefix_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )
        prefix_len = len(enc_prefix["input_ids"])

        if prefix_len >= self.max_length:
            return None

        assistant_start = min(prefix_len, len(input_ids))
        assistant_len = len(input_ids) - assistant_start
        if assistant_len < self.min_assistant_tokens:
            return None

        labels = input_ids.copy()
        labels[:assistant_start] = [-100] * assistant_start

        eos_token_id = self.tokenizer.eos_token_id
        for i in range(assistant_start, len(input_ids)):
            if input_ids[i] == eos_token_id:
                for j in range(i, len(labels)):
                    labels[j] = -100
                break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =====================================================
# Collator（修改：增加空 batch 安全处理）
# =====================================================

class EvalCollator:
    def __init__(self, tokenizer, max_length: int):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, features):
        # 过滤 None
        features = [f for f in features if f is not None]

        if len(features) == 0:
            # 返回最小占位 batch，避免 DataLoader 报错
            return {
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
            }

        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = min(len(f["input_ids"]), max_len)
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"][:seq_len])
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(f["labels"][:seq_len])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =====================================================
# Evaluation（保持不变）
# =====================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if batch is None:
            continue

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        valid_tokens = (batch["labels"] != -100).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


# =====================================================
# Main —— 修复QLoRA加载逻辑
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_qlora", action="store_true", help="是否为QLoRA模型")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = os.path.join(args.model_path, "adapter_config.json")

    if os.path.exists(adapter_config):
        print(f"[INFO] Loading LoRA / QLoRA model from {args.model_path}")

        peft_config = PeftConfig.from_pretrained(args.model_path)

        # ========== 修复后的QLoRA加载逻辑 ==========
        quantization_config = None
        if args.use_qlora:
            print(f"[INFO] Loading QLoRA (4-bit) model")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )

        model = PeftModel.from_pretrained(base_model, args.model_path)

    else:
        print(f"[INFO] Loading full model from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )

    dataset = BelleEvalDataset(
        data_path=args.eval_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    collator = EvalCollator(tokenizer, args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = model.device
    loss, ppl = evaluate(model, dataloader, device)

    print("\n========== Evaluation Result ==========")
    print(f"Eval loss       : {loss:.4f}")
    print(f"Perplexity (PPL): {ppl:.2f}")
    print("======================================\n")


if __name__ == "__main__":
    main()
