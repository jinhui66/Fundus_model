import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import pandas as pd
device = torch.device("cuda:0")
# 配置路径
MODEL_PATH = "/data3/wangchangmiao/jinhui/deepseek_fundus"
CSV_FILE = "/data3/wangchangmiao/jinhui/haha/data/validation_.csv"
BASE_IMAGE_PATH = "/data3/wangchangmiao/jinhui/DATA/fundus_test/Enhanced"
OUTPUT_CSV = os.path.join(os.path.dirname(CSV_FILE), "test_right_fundus_results.csv")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

text = "这张眼底图有什么异常，20字以内，避免重复提及之前提到的内容。"
# text = "这张眼底图像的杯盘比是多少？"
data = pd.read_csv(CSV_FILE)
# print(data[0])
messages = []
system_prompt = """你是一个专业的眼科医生，拥有丰富的眼科疾病诊断知识和经验。你需要帮助用户诊断眼底图像，回答问题。"""

for index, row in data.iterrows():
    # print(row)
    left_path = os.path.join(BASE_IMAGE_PATH, row.iloc[3])
    right_path =  os.path.join(BASE_IMAGE_PATH, row.iloc[4])
    # print(left_path)
    message = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
        "role": "user",
        "content": [
            # {
            #     "type": "image", 
            #     "image": f"file://{left_path}"
            # },
            {
                "type": "image", 
                "image": f"file://{right_path}"
            },
            {
                "type": "text",
                "text": text
            }
        ]
    }]
    messages.append(message)
results = []  # 用于存储结果
BSZ = 1
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]
    
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    

    # 保存结果
    for j, output_text in enumerate(batch_output_text):
        idx = i + j
        results.append({
            "right_image": data.iloc[idx, 4],
            "analysis_result": output_text.strip()
        })
    
    print('输入:', batch_messages)
    print('输出:', batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

# 保存到CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to {OUTPUT_CSV}")