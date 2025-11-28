# mysql_to_csv.py
import pandas as pd
from sqlalchemy import create_engine
from config import *

DB_USER = "root"
DB_PASSWORD = "mysql123"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "nlp_project"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

query = "SELECT Id, Sentence, Category FROM intents"
df = pd.read_sql(query, engine)

df.to_csv("train_for_model.csv", index=False)
print("MySQL 数据已导出为 CSV 成功！")
