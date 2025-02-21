import json

# 定义输入文件和输出文件的路径
input_file_path = 'train.jsonl'
output_file_path = 'output.jsonl'

# 读取输入文件并逐行解析
chartml_data = []
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        # 解析每行的JSON字符串
        data = json.loads(line)
        # 转换为ShareGPT格式
        for conversation in data['conversation']:
            chartml_conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": data['system'],
                    },
                    {
                        "role": "user",
                        "content": conversation['human'],
                    },
                    {
                        "role": "assistant",
                        "content": conversation['assistant'],
                    }
                ]
            }
            chartml_data.append(chartml_conversation)

# 将转换后的数据写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for item in chartml_data:
        json.dump(item, output_file, ensure_ascii=False)
        output_file.write('\n')  # 每个JSON对象后添加换行符

print(f"转换完成，结果已保存到 {output_file_path}")
