import json

# 定义输入文件和输出文件的路径
input_file_path = 'train.jsonl'
output_file_path = 'output.json'

# 读取输入文件并逐行解析
sharegpt_data = []
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        # 解析每行的JSON字符串
        data = json.loads(line)
        # 转换为ShareGPT格式
        for conversation in data['conversation']:
            sharegpt_conversation = {
                "system": data['system'],
                "tools":"",
                "conversations": [
                    {"from": "human", "value": conversation['human']},
                    {"from": "assistant", "value": conversation['assistant']}
                ]
            }
            sharegpt_data.append(sharegpt_conversation)

# 将转换后的数据写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(sharegpt_data, output_file, ensure_ascii=False, indent=2)

print(f"转换完成，结果已保存到 {output_file_path}")
