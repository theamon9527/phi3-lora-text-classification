# api_service.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import *

app = FastAPI()

try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    print("Tokenizer 加载成功！")
except Exception as e:
    print(f"加载 tokenizer 时出错: {e}")
    exit(1)

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base_model, output_dir_for_saving)
    model.eval()
    print("微调模型加载成功！")
except Exception as e:
    print(f"加载微调模型时出错: {e}")
    exit(1)

unique_categories = [
    'PlayMusic', 'RateBook', 'SearchCreativeWork', 'GetWeather',
    'BookRestaurant', 'AddToPlaylist', 'SearchScreeningEvent'
]

class SentenceInput(BaseModel):
    sentence: str

@app.post("/predict")
def predict_category(input: SentenceInput):
    prompt = (
        f"<|user|>\nClassify the following sentence into exactly one of these categories: "
        f"{', '.join(unique_categories)}. You must respond with only the category name, nothing else.\n\n"
        f"Sentence: {input.sentence}\n\nCategory: "
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            use_cache=False 
        )

    pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    final_category = next((c for c in unique_categories if c.lower() in pred.lower()), "Unknown")

    return {"category": final_category}

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=False)
