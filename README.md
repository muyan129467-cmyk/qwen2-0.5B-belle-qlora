**qwen2-0.5B-belle-qlora**

本项目使用belle group的10万数据集对qwen2-0.5B模型进行QLoRA微调，并使用2k数据作为验证集评价微调前后模型在验证集上的表现（使用Eval loss 和 PPL指标）


**数据集**

包含训练数据集（train_conversations.jsonl）和验证数据集（eval_conversations.jsonl）

data 链接: https://pan.baidu.com/s/1hKOlCd21DJgDbQ2QAM-GOQ?pwd=2f4v 提取码: 2f4v

**项目文件介绍**

实验环境文件：install.sh

微调训练代码：train_sft.py

验证评估代码：eval_sft.py

训练日志：trian_output.log

验证日志：final_eval_result.log

训练输出内容：output/qwen2-0.5b-lora
