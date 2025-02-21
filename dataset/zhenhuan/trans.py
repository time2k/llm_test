import json

def extract_character_conversations(input_file_path, output_file_path, character_name):
    """
    从剧本文件中提取指定角色的所有对话，以及上一句其他角色的对话，并保存为JSON格式的文件。
    
    :param input_file_path: 输入文件路径
    :param output_file_path: 输出文件路径
    :param character_name: 要提取对话的角色名称
    """
    try:
        # 打开输入文件并读取内容
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()

        # 准备存储对话的列表
        conversations = []
        previous_line = ""  # 用于存储上一行的内容
        for line in lines:
            # 去除行首和行尾的空白字符
            line = line.strip()
            # 检查当前行是否是指定角色的对话
            if line.startswith(character_name + "："):
                # 如果上一行不是空的，提取上一行的角色和对话
                if previous_line:
                    previous_parts = previous_line.split("：", 1)
                    if len(previous_parts) == 2:
                        previous_character, previous_dialogue = previous_parts
                        # 创建JSON对象
                        conversation = {
                            "instruction": f"({previous_character}) {previous_dialogue.strip()}",
                            "input":"",
                            "output": line[len(character_name) + 1:].strip()
                        }
                        conversations.append(conversation)
            # 更新上一行的内容
            previous_line = line

        # 打开输出文件并写入JSON格式的数据
        with open(output_file_path, 'a', encoding='utf-8') as output_file:
            #for conversation in conversations:
            
                output_file.write(json.dumps(conversations, indent=4, ensure_ascii=False))  # 每个JSON对象后添加换行符

        print(f"成功提取角色 {character_name} 的对话及其上一句对话，结果已保存到 {output_file_path}")
    except FileNotFoundError:
        print(f"错误：文件 {input_file_path} 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

# 示例使用
input_file_path = [] # 剧本文件路径
input_file_path.append('甄嬛传剧本01-10.txt')
input_file_path.append('甄嬛传剧本11-20.txt')
input_file_path.append('甄嬛传剧本21-30.txt')
input_file_path.append('甄嬛传剧本31-40.txt')
input_file_path.append('甄嬛传剧本41-50.txt')
input_file_path.append('甄嬛传剧本51-60.txt')
input_file_path.append('甄嬛传剧本61-70.txt')
input_file_path.append('甄嬛传剧本71-76.txt')

output_file_path = 'output.json'  # 输出文件路径
character_name = '甄嬛'  # 要提取对话的角色名称

for input_file in input_file_path:
    extract_character_conversations(input_file, output_file_path, character_name)