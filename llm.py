# =============================================
# NLP 2025 实验任务4: 大模型微调文本分类 (LoRA + Phi-3)
# =============================================
# LLM 在微调（Fine-tuning）时，默认的训练目标是序列生成，而不是多分类预测。
# --- 0. 导入必要的库 ---

print("--- 0. 导入必要的库 ---")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import pandas as pd
import numpy as np
import os

# --- 关键修复：手动添加 DynamicCache 缺失的方法 ---
print("--- 应用 DynamicCache 兼容性修复 ---")
from transformers.cache_utils import DynamicCache
from typing import Optional  # 添加 Optional 类型支持

# 1. 首先确保添加获取序列长度的方法，这是其他方法的基础
if not hasattr(DynamicCache, "get_seq_length"):
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """返回指定层的缓存序列长度。"""
        # 检查缓存是否存在且该层有内容
        if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            layer_cache = self.key_cache[layer_idx]
            if layer_cache is not None:
                # 形状: [batch_size, num_heads, seq_len, head_dim]
                return layer_cache.shape[-2]  # 返回序列长度
        return 0  # 如果缓存不存在或为空，返回0


    DynamicCache.get_seq_length = get_seq_length
    print("✓ 已手动添加 get_seq_length 方法到 DynamicCache")

# 2. 修复 get_max_length 方法 - 正确的类型注解
if not hasattr(DynamicCache, "get_max_length"):
    def get_max_length(self) -> Optional[int]:  # 使用 Optional[int] 而不是 int
        """返回缓存状态的最大序列长度。DynamicCache 没有最大长度限制。"""
        return None  # 对于 DynamicCache，返回 None 表示无限制


    DynamicCache.get_max_length = get_max_length
    print("✓ 已手动添加 get_max_length 方法到 DynamicCache")

# 3. 修复 get_usable_length 方法，正确使用参数
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """
        给定新输入的序列长度，返回缓存的可用长度。
        这是 transformers 库中 Cache 基类的标准实现。
        """
        # 获取缓存的最大长度（DynamicCache 理论上无限制）
        max_length = self.get_max_length()
        # 获取当前已缓存的序列长度
        previous_seq_length = self.get_seq_length(layer_idx)

        # 如果缓存有大小限制，且总长度将超过限制，则计算可用长度
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        # 缓存无大小限制 -> 所有缓存都可用
        return previous_seq_length


    DynamicCache.get_usable_length = get_usable_length
    print("✓ 已手动添加 get_usable_length 方法到 DynamicCache")

print("--- 0. 库导入完成 ---")

# --- 1. 定义文件和模型路径 ---
print("\n--- 1. 定义文件和模型路径 ---")

# 数据集目录
data_dir = r"D:\pycharm\PythonProjectllm\nlp-2025-experiment-task4"
train_file = os.path.join(data_dir, "train.csv")
test_file = os.path.join(data_dir, "test.csv")

# !!! !!! 指定你本地 Phi-3 模型文件夹的完整路径 !!! !!!
local_model_path = r"D:\HFLLM\hub\models--microsoft--Phi-3-mini-4k-instruct\snapshots\0a67737cc96d2554230f90338b163bc6380a2a85"

print(f"数据集目录: {data_dir}")
print(f"本地模型路径: {local_model_path}")
print(f"训练文件: {train_file}")
print(f"测试文件: {test_file}")

# --- 2. 加载数据集 ---
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

# --- 2.1. 分割数据集 ---
print("\n--- 2.1. 分割数据集 ---")
if len(train_dataset) > 0:
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)  # 10% 作为验证集
    train_dataset_split = split_dataset['train']
    eval_dataset_split = split_dataset['test']
    print(f"原训练集分割完成：")
    print(f"  训练集大小: {len(train_dataset_split)}")
    print(f"  验证集大小: {len(eval_dataset_split)}")
else:
    print("错误：训练数据集为空，无法进行分割。")
    exit(1)

# --- 3. 加载模型和分词器 (从本地) ---
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
        attn_implementation="eager"  # 强制使用eager注意力实现
    )
    print(f"模型已在 {model.device} 上加载 (4-bit 量化)。")
except Exception as e:
    print(f"从本地加载模型时发生错误: {e}")
    print(f"请检查本地模型路径 '{local_model_path}' 是否包含正确的模型文件。")
    exit(1)

# --- 4. 数据预处理 (分词) ---
print("\n--- 4. 数据预处理 (分词) ---")

# 定义所有可能的类别字符串
unique_categories = ['PlayMusic', 'RateBook', 'SearchCreativeWork', 'GetWeather',
                     'BookRestaurant', 'AddToPlaylist', 'SearchScreeningEvent']
categories_string = ", ".join(unique_categories)

# 创建类别映射字典
category_to_id = {category: i for i, category in enumerate(unique_categories)}
id_to_category = {i: category for category, i in category_to_id.items()}

print(f"类别映射: {category_to_id}")



def create_full_prompt(example):
    sentence = example["Sentence"]
    category = example["Category"]

    # 严格的指令遵循Prompt模板
    instruction_template = (
        "<|user|>\n"
        "Classify the following sentence into exactly one of these categories: {categories_string}. "
        "You must respond with only the category name, nothing else.\n\n"
        "Sentence: {sentence}\n\n"
        "Category: "
    )

    # 构建完整的Prompt（不含结束符）
    full_prompt = instruction_template.format(
        categories_string=categories_string,
        sentence=sentence
    )

    # 目标输出：仅类别名称（干净标签）
    target_output = category

    # 完整训练文本：Prompt + 干净类别
    full_text = full_prompt + target_output

    return full_text


# 测试Prompt生成
test_example = {"Sentence": "What's the weather like today?", "Category": "GetWeather"}
test_prompt = create_full_prompt(test_example)
print("Prompt示例:")
print(test_prompt)


def tokenize_function(examples):
    texts = []
    labels_list = []

    for i in range(len(examples["Sentence"])):
        # 为每个样本创建完整的Prompt训练文本
        full_text = create_full_prompt({
            "Sentence": examples["Sentence"][i],
            "Category": examples["Category"][i]
        })
        texts.append(full_text)

        # 对完整文本进行分词
        tokenized_text = tokenizer(
            full_text,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # 计算Prompt部分的长度（需要设置为-100）
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

        # 创建labels：Prompt部分为-100，类别部分保留原始ID
        input_ids = tokenized_text["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # 忽略Prompt部分的损失计算

        labels_list.append(labels)

    # 批量处理文本
    tokenized_output = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    # 手动处理labels的padding
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100  # 使用-100进行padding
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
    """测试集分词：只包含Prompt，不包含标签"""
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

# --- 5. 配置 LoRA ---
print("\n--- 5. 配置 LoRA ---")

# 关键修改：使用 Phi-3 模型的实际模块名称
print("正在分析模型结构以找到正确的目标模块...")
target_modules = []
for name, module in model.named_modules():
    if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
        target_modules.append(name.split('.')[-1])  # 只取最后一层名称

# 去重并确保模块存在
target_modules = list(set(target_modules))
print(f"可用的目标模块: {target_modules}")

# 使用找到的模块名称
TARGET_MODULES = target_modules

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
# 关键修改：禁用梯度检查点以避免DynamicCache问题
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

print("将 LoRA 适配器应用到模型...")
model = get_peft_model(model, lora_config)

print("\n--- LoRA 微调模型参数信息 ---")
model.print_trainable_parameters()

# --- 6. 定义训练参数并创建 Trainer ---
print("\n--- 6. 定义训练参数并创建 Trainer ---")
output_dir = "./phi3_text_classification_lora_finetuned"

# 修复训练参数，禁用缓存以避免DynamicCache问题
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
    eval_strategy="steps",  # 启用评估
    save_strategy="steps",
    load_best_model_at_end=True,  # 加载最佳模型
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

print("训练参数定义完成。")

# 准备 DataCollator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("DataCollator 准备完成。")

print("\n创建 Trainer 对象...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # 提供验证集
    tokenizer=tokenizer,
    data_collator=data_collator,  # 使用DataCollator
)
print("Trainer 对象创建成功。")

# --- 7. 开始训练 ---
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

# --- 8. 保存模型 ---
output_dir_for_saving = os.path.join(output_dir, "final_lora_adapters")
print(f"\n正在将最终的 LoRA 适配器保存到: {output_dir_for_saving}")
trainer.save_model(output_dir_for_saving)
print("最终 LoRA 适配器已成功保存！")

# --- 9. 修复预测问题 ---
print("\n=====================================")
print("    开始使用 LoRA 模型进行预测      ")
print("=====================================")

# 关键修复：使用正确的预测方法
print("加载微调后的模型进行预测...")

# 重新加载基础模型和LoRA适配器
try:
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  # 强制使用eager注意力实现
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, output_dir_for_saving)
    print("微调模型加载成功！")
except Exception as e:
    print(f"加载微调模型时发生错误: {e}")
    exit(1)

print("开始生成预测结果...")
preds_list = []

model.eval()
with torch.no_grad():
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        sentence = example["Sentence"]

        # 构建测试Prompt（与训练时一致）
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

        # Tokenize输入
        inputs = tokenizer(prompt_text, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 计算Prompt长度
        prompt_length = inputs["input_ids"].shape[1]

        # 生成预测（使用贪婪搜索确保确定性输出）
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,  # 类别名称通常很短
            do_sample=False,  # 关闭采样，使用贪婪搜索
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,  # 启用缓存以加速推理
        )

        # 提取生成的部分（去除Prompt）
        generated_token_ids = outputs[0][prompt_length:]

        # 解码生成的文本
        predicted_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        # 清理生成的文本：移除可能的多余内容
        cleaned_prediction = predicted_text.split('\n')[0]  # 只取第一行
        cleaned_prediction = cleaned_prediction.split('.')[0]  # 移除句点后的内容
        cleaned_prediction = cleaned_prediction.strip()

        print(f"样本ID {example['Id']} 原始生成: '{predicted_text}' -> 清理后: '{cleaned_prediction}'")

        # 匹配已知类别 - 优化后的匹配逻辑
        final_category = "Unknown"

        # 1. 严格开头匹配 (优先尝试)
        for category in unique_categories:
            if cleaned_prediction.lower().startswith(category.lower()):
                final_category = category
                break

        # 2. 宽松匹配 (如果严格匹配失败)
        if final_category == "Unknown":
            for category in unique_categories:
                # 检查类别名称是否出现在生成文本中
                if category.lower() in cleaned_prediction.lower():
                    final_category = category
                    break

        # 3. 针对 Unknown 的特殊处理（基于训练集的关键词匹配）
        if final_category == "Unknown":
            print(
                f"  -> 识别为 'Unknown', 尝试特殊处理 (ID: {example['Id']}, Sentence: '{sentence}', Raw Gen: '{predicted_text}')")

            # 基于训练集的关键词匹配
            lower_pred = cleaned_prediction.lower()

            # 为每个类别定义关键词列表（从训练集中提取）
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

            # 检查每个类别的关键词
            matched = False
            for category, keys in keywords.items():
                for key in keys:
                    if key in lower_pred:
                        final_category = category
                        matched = True
                        break
                if matched:
                    break

            # 如果仍然没有匹配，则默认为 SearchCreativeWork
            if final_category == "Unknown":
                print(f"  -> 仍然 Unknown, 默认分类为 'SearchCreativeWork' (ID: {example['Id']})")
                final_category = "SearchCreativeWork"

        preds_list.append({"Id": example["Id"], "Category": final_category})

        if i % 50 == 0:
            print(f"已处理 {i + 1}/{len(test_dataset)} 个样本。当前最佳猜测: {final_category}")

print("预测完成！")

# --- 10. 生成提交文件 ---
print("\n开始生成提交文件...")

submission_df = pd.DataFrame(preds_list)
submission_df['Id'] = submission_df['Id'].astype(int)
submission_df = submission_df.sort_values(by='Id')

submission_output_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_output_path, index=False)

print(f"提交文件已生成并保存到: {submission_output_path}")
print("\n--- 提交文件预览 (前5行) ---")
print(submission_df.head())

print("\n--- 预测结果统计 ---")
print(submission_df['Category'].value_counts())

print("\n===================================================")
print("    项目四（文本分类 LoRA + Phi-3）完整代码执行完毕！   ")
print("===================================================")