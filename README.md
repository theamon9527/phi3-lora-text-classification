# Kaggle 基于 Phi-3 大模型的 LoRA 微调文本分类项目

## 项目简介
针对口语对话的意图检测任务，利用 **LoRA** 技术高效微调 **Phi-3-mini-4k-instruct** 大模型，以准确识别用户话语的意图。  
项目独立构建了端到端的微调流程，并实现了模型推理服务 API，支持输入数据的实时处理与结果显示。  

**贡献：**

- 端到端微调流程：数据清洗 → 特征工程 → 模型加载 → LoRA 微调 → 推理验证 → 结果后处理
- 数据管理：通过 **MySQL** 管理训练数据，实现训练/验证集自动读取
- 模型优化：结合 **Prompt 工程**，解决 LLM 序列生成与多分类目标不匹配问题
- 高效训练：4-bit 量化 + GPU 加速，显存优化与训练速度提升
- 部署能力：使用 **FastAPI** 提供 API 服务，并封装调用逻辑
- 竞赛表现：在 Kaggle 竞赛中取得 **0.95452 高分**
---
## 技术栈 & 
- **大模型与微调**：Phi-3-mini-4k-instruct, Transformers, PEFT (LoRA)  
- **量化加速**：BitsAndBytes 4-bit  
- **数据处理**：pandas, numpy  
- **数据库**：MySQL, SQLAlchemy, pymysql  
- **API 服务**：FastAPI, uvicorn  
- **客户端交互**：requests  
- **训练与推理**：PyTorch, Datasets
---
## 项目架构

PythonProjectllm/  
│  
├─ nlp-2025-experiment-task4/ # 训练和测试数据及检测结果  
├─ phi3_text_classification_lora_finetuned/ # LoRA 适配器模型  
│  
├─ data_to_mysql.py # 数据导入 MySQL  
├─ mysql_to_csv.py # MySQL 导出数据为 CSV  
├─ llm_train.py # LoRA 微调训练脚本  
├─ llm_predict.py # 批量预测脚本  
├─ api_service.py # FastAPI 模型服务  
├─ client.py # Python API 客户端封装  
├─ config.py # 文件路径及配置  
├─ requirements.txt # Python 依赖  
└─ Dockerfile # 容器化部署  

注：大模型文件（Phi-3-mini-4k-instruct）较大，不上传 GitHub，需要自行下载，并在 `config.py` 中修改 `local_model_path` 指向本地路径。
