import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import pandas as pd
import numpy as np
import os
from config import *

print("\n=====================================")
print("    开始使用 LoRA 模型进行预测      ")
print("=====================================")

print("加载微调后的模型进行预测...")

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base_model, output_dir_for_saving)
    print("微调模型加载成功！")
except Exception as e:
    print(f"加载微调模型时发生错误: {e}")
    exit(1)

print("\n--- 加载数据集 ---")

test_df = None

try:
    print(f"尝试加载: {test_file}")
    test_df = pd.read_csv(test_file, encoding='utf-8', engine='python', encoding_errors='replace')
    print("test.csv 加载成功！")
except FileNotFoundError as e:
    print(f"错误：文件未找到 - {e}")
    print(f"请确保 'data_dir' 路径 '{data_dir}' 正确，并且包含 train.csv, test.csv 文件。")
    exit(1)
except Exception as e:
    print(f"加载数据集时发生错误: {e}")
    print("请检查文件的内容和编码。")
    exit(1)

try:
    print("检查并转换为 Hugging Face Dataset 格式...")
    test_dataset = Dataset.from_pandas(test_df[['Id', 'Sentence']])
    print("数据集已转换为 Hugging Face Dataset 格式。")
except KeyError as e:
    print(f"错误：在转换到 Dataset 时，找不到预期的列 - {e}")
    print("\n请检查 CSV 文件中的列名是否正确：")
    print("test.csv 应包含 'Id', 'Sentence' 列。")
    if test_df is not None: print(f"test.csv 实际列名: {test_df.columns.tolist()}")
    exit(1)

print("开始生成预测结果...")
preds_list = []

model.eval()
with torch.no_grad():
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        sentence = example["Sentence"]

        prompt_text = (
            "<|user|>\n"
            "Classify the following sentence into exactly one of these categories: {categories_string}. "
            "You must respond with only the category name, nothing else.\n\n"
            "Sentence: {sentence}\n\n"
            "Category: "
        ).format(
            categories_string=categories_string,
            sentence=sentence
        )

        inputs = tokenizer(prompt_text, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

        generated_token_ids = outputs[0][prompt_length:]

        predicted_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        cleaned_prediction = predicted_text.split('\n')[0]
        cleaned_prediction = cleaned_prediction.split('.')[0]
        cleaned_prediction = cleaned_prediction.strip()

        print(f"样本ID {example['Id']} 原始生成: '{predicted_text}' -> 清理后: '{cleaned_prediction}'")

        final_category = "Unknown"

        for category in unique_categories:
            if cleaned_prediction.lower().startswith(category.lower()):
                final_category = category
                break

        if final_category == "Unknown":
            for category in unique_categories:
                if category.lower() in cleaned_prediction.lower():
                    final_category = category
                    break

        if final_category == "Unknown":
            print(
                f"  -> 识别为 'Unknown', 尝试特殊处理 (ID: {example['Id']}, Sentence: '{sentence}', Raw Gen: '{predicted_text}')")

            lower_pred = cleaned_prediction.lower()

            keywords = {
                "PlayMusic": ["play", "music", "song", "album", "track", "hear", "listen", "on deezer", "on spotify",
                              "on pandora", "on groove shark", "on last fm", "by artist", "singing", "playing"],
                "RateBook": ["rate", "rating", "stars", "out of", "points", "give", "star", "book", "novel", "textbook",
                             "essay", "series", "chronicle"],
                "SearchCreativeWork": ["find", "search", "show", "locate", "see", "watch", "game", "painting",
                                       "photograph", "novel", "tv series", "trailer", "discography", "movie", "film",
                                       "picture", "journal"],
                "GetWeather": ["weather", "forecast", "temperature", "humid", "cloudy", "sunny", "rainy", "snowy",
                               "freezing", "warm", "cold", "chillier", "hotter", "degrees", "humidity", "snowfall",
                               "windy"],
                "BookRestaurant": ["book", "reservation", "table", "spot", "restaurant", "cafe", "bar", "gastropub",
                                   "reserve", "seating", "dinner", "lunch", "breakfast", "eat", "food", "cuisine"],
                "AddToPlaylist": ["add", "to my playlist", "include", "put", "incorporate", "playlist", "mix",
                                  "collection", "tracks", "songs"],
                "SearchScreeningEvent": ["movie", "theatre", "theater", "showing", "showtimes", "schedule", "time",
                                         "playing", "films", "cinema", "animated", "screening", "matinee", "showtime"]
            }

            matched = False
            for category, keys in keywords.items():
                for key in keys:
                    if key in lower_pred:
                        final_category = category
                        matched = True
                        break
                if matched:
                    break

            if final_category == "Unknown":
                print(f"  -> 仍然 Unknown, 默认分类为 'SearchCreativeWork' (ID: {example['Id']})")
                final_category = "SearchCreativeWork"

        preds_list.append({"Id": example["Id"], "Category": final_category})

        if i % 50 == 0:
            print(f"已处理 {i + 1}/{len(test_dataset)} 个样本。当前最佳猜测: {final_category}")

print("预测完成！")

print("\n开始生成提交文件...")

submission_df = pd.DataFrame(preds_list)
submission_df['Id'] = submission_df['Id'].astype(int)
submission_df = submission_df.sort_values(by='Id')

submission_output_path = os.path.join(data_dir, "submission.csv")
submission_df.to_csv(submission_output_path, index=False)

print(f"提交文件已生成并保存到: {submission_output_path}")
print("\n--- 提交文件预览 (前5行) ---")
print(submission_df.head())

print("\n--- 预测结果统计 ---")
print(submission_df['Category'].value_counts())

print("\n===================================================")
print("    项目四（文本分类 LoRA + Phi-3）完整代码执行完毕！   ")
print("===================================================")
