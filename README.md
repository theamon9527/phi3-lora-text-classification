# Kaggle 基于 Phi-3 大模型的 LoRA 微调文本分类项目

## 项目简介
针对口语对话的意图检测任务，利用 **LoRA** 技术高效微调 **Phi-3-mini-4k-instruct** 大模型，以准确识别用户话语的意图。  
项目独立构建了端到端的微调流程，并实现了模型推理服务 API，支持输入数据的实时处理与结果显示。  

**核心亮点：**

- 端到端微调流程：数据清洗 → 特征工程 → 模型加载 → LoRA 微调 → 推理验证 → 结果后处理
- 数据管理：通过 **MySQL** 管理训练数据，实现训练/验证集自动读取
- 模型优化：结合 **Prompt 工程**，解决 LLM 序列生成与多分类目标不匹配问题
- 高效训练：4-bit 量化 + GPU 加速，显存优化与训练速度提升
- 部署能力：使用 **FastAPI** 提供 API 服务，并封装 Python 客户端调用逻辑
- 竞赛表现：在 Kaggle 竞赛中取得 **0.95452 高分**

---

## 项目架构

PythonProjectllm/
│
├─ nlp-2025-experiment-task4/ # 训练和测试数据及检测结果
├─ phi3_text_classification_lora_finetuned/ # LoRA 适配器模型
│
├─ data_to_mysql.py # CSV 导入 MySQL
├─ mysql_to_csv.py # MySQL 导出 CSV
├─ llm_train.py # LoRA 微调训练脚本
├─ llm_predict.py # 批量预测脚本
├─ api_service.py # FastAPI 模型服务
├─ client.py # Python API 客户端封装
├─ config.py # 文件路径及配置
├─ requirements.txt # Python 依赖
└─ Dockerfile # 容器化部署



---

## 技术栈
- **大模型与微调**：Phi-3-mini-4k-instruct, Transformers, PEFT (LoRA)
- **量化加速**：BitsAndBytes 4-bit
- **数据处理**：pandas, numpy
- **数据库**：MySQL, SQLAlchemy, pymysql
- **API 服务**：FastAPI, uvicorn
- **客户端交互**：requests
- **训练与推理**：PyTorch, Datasets

---


### 1. 克隆项目
```bash
git clone <你的仓库地址>
cd <项目文件夹>
2. 安装依赖
bash
复制代码
pip install -r requirements.txt
3. 启动 MySQL 并导入训练数据
bash
复制代码
python data_to_mysql.py
4. 训练 LoRA 模型（可选）
bash
复制代码
python llm_train.py
5. 批量预测
bash
复制代码
python llm_predict.py
6. 启动 API 服务
bash
复制代码
python api_service.py
7. 使用 Python 客户端交互
bash
复制代码
python client.py
Docker 部署示例
bash
复制代码
# 构建镜像
docker build -t phi3-service .

# 运行容器
docker run -p 8000:8000 phi3-service

# 访问 API
http://localhost:8000/predict
⚠️ 注意：大模型文件（Phi-3-mini-4k-instruct）较大，不上传 GitHub，需要自行下载，并在 config.py 中修改 local_model_path 指向本地路径。

贡献
独立完成端到端 LoRA 微调流程

实现 MySQL 数据管理

构建 FastAPI 推理服务和 Python 客户端

提升模型对口语意图理解能力，解决多分类训练问题

输出高性能 LoRA 适配器，结合 4-bit 量化实现显存优化

成果
Kaggle 得分：0.95452

可直接部署为 API 服务

支持批量预测和实时输入预测