from transformers import AutoModelForSequenceClassification,AutoTokenizer
import os
 
# 需要保存的lora路径
lora_path= "./output_lora/Qwen2.5/"
# 模型路径
model_path = './models/Qwen/Qwen2.5-0.5B-Instruct/'
# 检查点路径
checkpoint_dir = './output/Qwen2.5'
checkpoint = [file for file in os.listdir(checkpoint_dir) if 'checkpoint-' in file][-1] #选择更新日期最新的检查点
print('checkpoint_dir is '+checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(f'{checkpoint_dir}/{checkpoint}')
# 保存模型
model.save_pretrained(lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# 保存tokenizer
tokenizer.save_pretrained(lora_path)
