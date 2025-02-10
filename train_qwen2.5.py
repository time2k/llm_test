import json
import pandas as pd
import torch
from datasets import Dataset
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import os
import swanlab

# Check that CUDA/MPS is available
x_device = ""
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    x_device = "cuda"
else:
    if torch.mps.is_available():
        print("Apple MPS is available. Using MPS.")
        x_device = "mps"
    else:
        x_device = "cpu"


def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(x_device)

    #model.generation_config.do_sample=False
    #model.generation_config.temperature=0.6
    #model.generation_config.top_k=None
    #nemodel.generation_config.top_p=None

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    return response


# 在modelscope上下载Qwen模型到本地目录下
#model_dir = snapshot_download(
#    "Qwen/Qwen2.5-0.5B-Instruct", cache_dir="./", revision="master"
#)

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(
    "./models/Qwen/Qwen2.5-0.5B-Instruct/", use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen/Qwen2.5-0.5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

train_dataset = Dataset.load_from_disk("./dataset")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)
#model = model.bfloat16()

args = TrainingArguments(
    output_dir="./output/Qwen2.5",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,           # 梯度裁剪 防止梯度爆炸
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-fintune",
    experiment_name="Qwen2.5-0.5B-Instruct",
    description="使用通义千问Qwen2.5-0.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
    config={
        "model": "qwen/Qwen2.5-0.5B-Instruct",
        "dataset": "huangjintao/zh_cls_fudan-news",
    },
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

# 用测试集的前10条，测试模型
test_df = pd.read_json("./raw_dataset/new_test.jsonl", lines=True)[:10]

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row["instruction"]
    input_value = row["input"]

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"},
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()
