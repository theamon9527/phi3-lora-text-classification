# config.py
import os

from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

local_model_path = r"D:\HFLLM\hub\models--microsoft--Phi-3-mini-4k-instruct\snapshots\0a67737cc96d2554230f90338b163bc6380a2a85"
output_dir = "./phi3_text_classification_lora_finetuned"
output_dir_for_saving = os.path.join(output_dir, "final_lora_adapters")
data_dir = r"D:\pycharm\PythonProjectllm\nlp-2025-experiment-task4"
test_file = os.path.join(data_dir, "test.csv")
train_file = os.path.join(data_dir, "train.csv")

unique_categories = [
    'PlayMusic', 'RateBook', 'SearchCreativeWork', 'GetWeather',
    'BookRestaurant', 'AddToPlaylist', 'SearchScreeningEvent'
]

categories_string = ", ".join(unique_categories)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
