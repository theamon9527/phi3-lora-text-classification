print("--- 0. 导入必要的库 ---")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import pandas as pd
import numpy as np
import os
from transformers.cache_utils import DynamicCache
from typing import Optional

print("--- 应用 DynamicCache 兼容性修复 ---")
if not hasattr(DynamicCache, "get_seq_length"):
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            layer_cache = self.key_cache[layer_idx]
            if layer_cache is not None:
                return layer_cache.shape[-2]
        return 0
    DynamicCache.get_seq_length = get_seq_length
    print("✓ 已手动添加 get_seq_length 方法到 DynamicCache")

if not hasattr(DynamicCache, "get_max_length"):
    def get_max_length(self) -> Optional[int]:
        return None
    DynamicCache.get_max_length = get_max_length
    print("✓ 已手动添加 get_max_length 方法到 DynamicCache")

if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length
    DynamicCache.get_usable_length = get_usable_length
    print("✓ 已手动添加 get_usable_length 方法到 DynamicCache")

print("--- 0. 库导入完成 ---")

print("\n--- 1. 定义文件和模型路径 ---")
data_dir = r"D:\pycharm\PythonProjectllm\nlp-2025-experiment-task4"
train_file = os.path.join(data_dir, "train.csv")
test_file = os.path.join(data_dir, "test.csv")
local_model_path = r"D:\HFLLM\hub\models--microsoft--Phi-3-mini-4k-instruct\snapshots\0a67737cc96d2554230f90338b163bc6380a2a85"
print(f"数据集目录: {data_dir}")
print(f"本地模型路径: {local_model_path}")
print(f"训练文件: {train_file}")
print(f"测试文件: {test_file}")

print("\n--- 2. 加载数据集 ---")
train_df = None
test_df = None
try:
    print(f"尝试加载: {train_file}")
    train_df = pd.read_csv(train_file, encoding='utf-8', engine='python', encoding_errors='replace')
    print("train.csv 加载成功！")
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
    train_dataset = Dataset.from_pandas(train_df[['Id', 'Sentence', 'Category']])
    test_dataset = Dataset.from_pandas(test_df[['Id', 'Sentence']])
    print("数据集已转换为 Hugging Face Dataset 格式。")
except KeyError as e:
    print(f"错误：在转换到 Dataset 时，找不到预期的列 - {e}")
    print("\n请检查 CSV 文件中的列名是否正确：")
    print("train.csv 应包含 'Id', 'Sentence', 'Category' 列。")
    print("test.csv 应包含 'Id', 'Sentence' 列。")
    if train_df is not None: print(f"train.csv 实际列名: {train_df.columns.tolist()}")
    if test_df is not None: print(f"test.csv 实际列名: {test_df.columns.tolist()}")
    exit(1)

print("\n--- 2.1. 分割数据集 ---")
if len(train_dataset) > 0:
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset_split = split_dataset['train']
    eval_dataset_split = split_dataset['test']
    print(f"原训练集分割完成：")
    print(f"  训练集大小: {len(train_dataset_split)}")
    print(f"  验证集大小: {len(eval_dataset_split)}")
else:
    print("错误：训练数据集为空，无法进行分割。")
    exit(1)

print("\n--- 3. 加载模型和分词器 (从本地) ---")
print(f"正在尝试从本地加载模型和 Tokenizer: {local_model_path}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("4-bit 量化配置已设置。")

try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token 已设置为 eos_token。")
    print("Tokenizer 加载完成！")
except Exception as e:
    print(f"从本地加载 Tokenizer 时发生错误: {e}")
    print(f"请检查本地模型路径 '{local_model_path}' 是否包含正确的 tokenizer 文件。")
    exit(1)

try:
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print(f"模型已在 {model.device} 上加载 (4-bit 量化)。")
except Exception as e:
    print(f"从本地加载模型时发生错误: {e}")
    print(f"请检查本地模型路径 '{local_model_path}' 是否包含正确的模型文件。")
    exit(1)

print("\n--- 4. 数据预处理 (分词) ---")
unique_categories = ['PlayMusic', 'RateBook', 'SearchCreativeWork', 'GetWeather', 'BookRestaurant', 'AddToPlaylist', 'SearchScreeningEvent']
categories_string = ", ".join(unique_categories)
category_to_id = {category: i for i, category in enumerate(unique_categories)}
id_to_category = {i: category for category, i in category_to_id.items()}
print(f"类别映射: {category_to_id}")

def create_full_prompt(example):
    sentence = example["Sentence"]
    category = example["Category"]
    instruction_template = (
        "<|user|>\n"
        "Classify the following sentence into exactly one of these categories: {categories_string}. "
        "You must respond with only the category name, nothing else.\n\n"
        "Sentence: {sentence}\n\n"
        "Category: "
    )
    full_prompt = instruction_template.format(
        categories_string=categories_string,
        sentence=sentence
    )
    target_output = category
    full_text = full_prompt + target_output
    return full_text

test_example = {"Sentence": "What's the weather like today?", "Category": "GetWeather"}
test_prompt = create_full_prompt(test_example)
print("Prompt示例:")
print(test_prompt)

def tokenize_function(examples):
    texts = []
    labels_list = []
    for i in range(len(examples["Sentence"])):
        full_text = create_full_prompt({
            "Sentence": examples["Sentence"][i],
            "Category": examples["Category"][i]
        })
        texts.append(full_text)
        tokenized_text = tokenizer(
            full_text,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        prompt_only = (
            "<|user|>\n"
            "Classify the following sentence into exactly one of these categories: {categories_string}. "
            "You must respond with only the category name, nothing else.\n\n"
            "Sentence: {sentence}\n\n"
            "Category: "
        ).format(
            categories_string=categories_string,
            sentence=examples["Sentence"][i]
        )
        prompt_tokens = tokenizer(prompt_only, return_tensors="pt", max_length=256, truncation=True)
        prompt_length = prompt_tokens["input_ids"].shape[1]
        input_ids = tokenized_text["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels_list.append(labels)
    tokenized_output = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100
    )
    tokenized_output["labels"] = padded_labels
    return tokenized_output

print("应用优化后的分词函数到训练集...")
tokenized_train_dataset = train_dataset_split.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset_split.column_names
)
print("训练集分词完成。")
print("应用分词函数到验证集...")
tokenized_eval_dataset = eval_dataset_split.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset_split.column_names
)
print("验证集分词完成。")

def tokenize_test_function(examples):
    prompt_texts = []
    for i in range(len(examples["Sentence"])):
        prompt_only = (
            "<|user|>\n"
            "Classify the following sentence into exactly one of these categories: {categories_string}. "
            "You must respond with only the category name, nothing else.\n\n"
            "Sentence: {sentence}\n\n"
            "Category: "
        ).format(
            categories_string=categories_string,
            sentence=examples["Sentence"][i]
        )
        prompt_texts.append(prompt_only)
    tokenized_output = tokenizer(
        prompt_texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    return tokenized_output

print("应用分词函数到测试集...")
tokenized_test_dataset = test_dataset.map(
    tokenize_test_function,
    batched=True,
    remove_columns=test_dataset.column_names
)
print("测试集分词完成。")

print("\n--- 5. 配置 LoRA ---")
print("正在分析模型结构以找到正确的目标模块...")
target_modules = []
for name, module in model.named_modules():
    if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
        target_modules.append(name.split('.')[-1])
TARGET_MODULES = list(set(target_modules))
print(f"可用的目标模块: {TARGET_MODULES}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=TARGET_MODULES,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
print("LoRA 配置完成。")
print("准备模型进行 4-bit 训练...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
print("将 LoRA 适配器应用到模型...")
model = get_peft_model(model, lora_config)
print("\n--- LoRA 微调模型参数信息 ---")
model.print_trainable_parameters()

print("\n--- 6. 定义训练参数并创建 Trainer ---")
output_dir = "./phi3_text_classification_lora_finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=50,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)
print("训练参数定义完成。")
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("DataCollator 准备完成。")
print("\n创建 Trainer 对象...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Trainer 对象创建成功。")

print("\n=====================================")
print("        开始 LoRA 微调训练         ")
print("=====================================")
try:
    train_result = trainer.train()
    print("\n=====================================")
    print("        LoRA 微调训练完成！        ")
    print("=====================================")
    metrics = train_result.metrics
    print("训练指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"\n训练过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

output_dir_for_saving = os.path.join(output_dir, "final_lora_adapters")
print(f"\n正在将最终的 LoRA 适配器保存到: {output_dir_for_saving}")
trainer.save_model(output_dir_for_saving)
print("最终 LoRA 适配器已成功保存！")
