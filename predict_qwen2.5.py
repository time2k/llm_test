import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(x_device)

    #model.generation_config.do_sample=False
    #model.generation_config.temperature=None
    #model.generation_config.top_k=None
    #model.generation_config.top_p=None

    #generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, max_new_tokens=512)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 加载原下载路径的tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen/Qwen2.5-0.5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./models/Qwen/Qwen2.5-0.5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)

# 加载训练好的Lora模型，将下面的checkpoint-[XXX]替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="./output/Qwen2.5/checkpoint-500")

test_texts = {
    'instruction': "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
    'input': "文本:航空动力学报JOURNAL OF AEROSPACE POWER1998年 第4期 No.4 1998科技期刊管路系统敷设的并行工程模型研究*陈志英*　*　马　枚北京航空航天大学【摘要】　提出了一种应用于并行工程模型转换研究的标号法，该法是将现行串行设计过程(As-is)转换为并行设计过程(To-be)。本文应用该法将发动机外部管路系统敷设过程模型进行了串并行转换，应用并行工程过程重构的手段，得到了管路敷设并行过程模型，文中对该模型进行了详细分析。最后对转换前后的模型进行了时间效益分析，得到了对敷管工程有指导意义的结论。\t主题词：　航空发动机　管路系统　计算机辅助设计　建立模型　　　自由词：　并行工程分类号：　V2331　管路系统现行设计过程发动机管路设计的传统过程是串行的，即根据发动机结构总体设计要求，首先设计发动机主体及附件，之后确定各附件位置及相互连接关系。传统的管路敷设方法通常先制作一个与原机比例为1∶1的金属样机，安装上附件模型或实际的附件，敷管工程师用4mm铅丝进行附件间导管连接的实地测量打样，产生导管走向的三维空间形状，进行间隙检查、导管最小弯曲半径检查、导管强度振动分析、可维修性、可达性及热膨胀约束分析等等［1］。历经多次反复敷管并满足所有这些要求后交弯管车间按此铅丝的三维空间形状进行弯管，之后安装导管、管接头、卡箍等。这样每敷设一根管时均需经上述过程，直到敷完所有连接导管。最后装飞机检验图1　管路系统传统设计过程模型整个发动机外廓与飞机短舱间的间隙是否合适，检验发动机各导管与飞机连接是否合理。管路系统传统研制过程模型参见图1。由该过程模型可见，传统的管路系统设计是经验设计，其管路敷设过程涉及总体、附件、材料、工艺、加工、飞机部门等众多部门和环节，而管路设计本身尚不规范，设计研制过程中必然要有大量反复工作存在，从而会大大延长发动机研制周期。面临现代社会越来越激烈的市场竞争，使得产品开发者力争在最短时间内（T），消耗最少的资金（C），生产出满足市场需求（Q）的产品，必须改变产品开发模式，通过过程重构，把传统的串行过程转换为并行设计过程，提高敷管过程的并行度［2］。2　过程重构及标号法过程重构是研究如何将传统的开发过程转换为并行的开发过程，从而使产品研制周期大大缩短，并取得“一次成功”。过程模型的转换可以通过专家经验法实现，然而对于一个复杂过程，如果不借助数学工具，仅靠人工观察是不够的。一个复杂产品的开发需要多部门、多学科的共同参与，因此会有大量活动，而且活动间的约束关系错综复杂［3］。本文提出了进行过程重构的“标号法”算法。思路如下：2．1　概念与定义定义1设有n个活动V1，V2，…，Vn，之间存在有序关系，定义关联矩阵：A＝［aij］n×n若Vj是Vi的前序，aij＝1；若Vj不是Vi的前序，aij＝0；并规定aii＝1。定义2　若ai1i2＝ai2i3＝…＝ais-1is＝aisi1＝1，则称C＝｛Vi1，Vi2，…，Vis｝为1个环。定义3　若C为一个环，且C与其它Vj不构成环，称C为饱和环。定义4　L(1)，L(2)，…，L(k)称为层集。L(i)是Vj或C的集合，且L(1)∪L(2)∪…∪L(k)＝｛1，2，…，n｝，即L(1)，…，L(k)互不相交。2．2　基本原理（1）Vi没有前序，即 Ai*e＝1，则Vi为初始层OA；（2）｛V1，V2，…，Vn｝中的初始层有且只有环存在，则A*e＞e。2．3　迭代(1)Ae＝　XJ＞eJ可对层进行标号L(s)；(2)A＝令A＝AJ，S＝S＋1，进行步骤(1)；(3)AJ*e＞eJ ，搜索法找到一个饱和环C(k)；(4)对C(k)进行标号L(k)，并判断其前序性。若J＝Φ（空）则结束。若J≠Φ，令A＝AJ并返回到步骤(1)。3　并行过程模型的建立根据前述标号法方法对图1包含20项任务的管路系统传统设计过程模型建立关联矩阵A，如图2所示（图中数字含义与图1相同）。对此关联矩阵用本文提出的标号法进行过程重构，得到变换后的关联矩阵A′，见图3。从而把原20项任务确定出7个活动层，其中第2层含2个并行活动，第3层包含3个并行活动，第5层包含4个并行活动，第6层是一个饱和环。通过对此实例的过程变换与分析，可以看出“标号法”具有以下特点：(1)总结出此类算法的2条基本原理，证实了算法的可行性；(2)在算法上用单位向量与活动关联阵相乘，求初始层OA，使标号更加明确；(3)寻找层的同时进行了标号，对活动项排序的同时也进行了标号；(4)定义了饱和环概念，消除了嵌环的问题，从而可找到最小环，消除原过程的大循环；(5)用数学的概念进行了算法表达，可对任何过程通过计算进行模型转换。图2　关联矩阵A　　　　　　　　　　图3　转换后的关联矩阵A′由于工程性问题通常存在任务反馈，反馈对于生产研制过程是必不可少的而又非常重要的。为提高并行度，有的专家提出无条件消除反馈［4］，但这势必带来产品设计过程中的不合理性，大大降低了其工程实用性。正是因为反馈参数的存在，才使产品开发过程出现循环反复，延长了研制周期。因此解决反馈问题是工程性课题的重点研究内容。本文提出把“标号法”应用于解决工程实际问题时，可采用修正方法，即：把既是后续任务又具有约束条件的任务项，变换为约束控制类任务提前到被分析任务项前进行，使其成为前序约束任务项来制约过程设计。由此可知任务项11（约束分析）、12（工艺性分析）和任务19（装飞机检验）符合上述原则，可考虑把这三项任务提前到管路敷设任务9前进行。转换后的并行过程模型见图4（图中数字含义与图1相同）。从图中可看出这7个层次，即结构总体设计、外形设计、建模、样机装配、材料准备及约束、敷管和质量检验。图4　管路系统并行设计过程模型(图中数字说明见图1)由此可见，经过重构后的敷管过程具有以下几个特点：(1)敷管过程大循环多反馈变换为小循环少反馈，一些任务项转换为约束控制类任务，使需要多次调整管路走向的敷管任务环内项目数大大减少，缩短了开发周期；(2)对管路敷设有反馈的所有任务项集中在敷管任务项前后进行，突出体现了并行工程强调协作的重要作用意义；例如把装飞机检验提到敷管同时进行，可完全避免大返工现象发生；(3)过程管理流程的层次更加分明，有利于量化管理过程，有利于产品管理、组织管理、资源管理等做相应调整。4　效益分析经分段分析计算，传统的管路敷设过程若一次敷管成功要持续14个月左右，但事实上没有一台新机研制过程中是一次敷管成功的。由上述过程重构分析可见，传统的串行敷管过程基本上一次敷管只偏重满足一种约束要求，而敷管过程是一个多约束控制过程，因此必然造成多次敷管方能满足所有要求。例如，首次敷管偏重于考虑不干涉问题，但当管路敷设完之后，才发现已敷管路不能满足其它敷管要求时，这时调整某些导管，会发生“牵一动百”效应，势必需要再花一段时间（2个月左右）调整管路走向。在实际工程中，由于管路设计任务滞后发动机其它零部件研制阶段，使得留给管路设计任务的周期短而又短，为满足发动机试车任务的急需，可能先敷设一个不考虑任何约束的管路系统。因此在传统敷管方式下多次敷管过程是必然的。当然实际工程中做到同时满足所有约束要求的一次敷管成功，不仅要建立1个并行过程模型，还需要一些技术和管理条件的支持，如到一定阶段加强各部门协调。还有采用CAD技术，金属表1　效益分析传统敷管成功次数传统时间节省时间节省效益一次二次三次…十次14个月16个月18个月32个月5个月7个月9个月23个月36.710%43.755%50.000%71.870%* 节省效益＝节省时间/传统时间样机的制作也可相应废除掉，改做电子模型样机，至少可节省数万元的制作费［5］。另外把原每敷设完一根导管就要进行弯管和装管的过程，改为在计算机上敷设完所有管路模型之后进行弯管及装配，可避免大量无效弯管的浪费，也大大节省了开发周期，同时降低开发费用，使一次敷管成功成为可能。管路敷设过程重构后，由于一些任务可同时进行，因此并行敷管过程只需9个月左右时间。效益分析及结果见表1。结束语　改变现行开发模式，建立以CAD为技术支持的管路系统敷设开发数据工具，形成以过程管理为基础的管路敷设系统，将大大缩短开发周期，也必将提高敷管质量，同时也将降低开发费用。参　考　文　献1　马晓锐．航空发动机外部管路敷设的专家系统：［学位论文］．航空航天工业部六o六研究所，19932　王甫君．支持并行工程集团群工作的通信、协调、控制系统：［学位论文］．北京航空航天大学，19973　Biren Prased.Concurrent Engineering Fundamentals. Volume I,Prentice Hall PTR,USA,19964　A Kusiak, T N Larson.Reengineering of Design and Mannufacturing Processes.Computer and Engingeering,1994,26(3):521-5365　James C B,Henry A. J.Electronic Engine‘Mockup’Shortens Design Time. AEROSPACE AMERICA,PP98－100,January 1985（责任编辑　杨再荣）1998年5月收稿；1998年7月收到修改稿。　*潮疚南倒家自然科学基金资助项目，编号：95404F100A(69584001)*　*男　38岁　副研究员　北京航空航天大学405教研室　100083,类型选型:['Military', 'Space']'"
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages, model, tokenizer)
print(response)
