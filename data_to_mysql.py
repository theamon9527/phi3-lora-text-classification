# data_to_mysql.py
import pandas as pd
from sqlalchemy import create_engine, text
from config import train_file  
import pymysql

DB_USER = "root"
DB_PASSWORD = "mysql123"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "nlp_project"


engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

df = pd.read_csv(train_file)

create_table_sql = """
CREATE TABLE IF NOT EXISTS intents (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    Sentence VARCHAR(1024),
    Category VARCHAR(128)
);
"""

with engine.begin() as conn:  
    conn.execute(text("DROP TABLE IF EXISTS intents"))
    conn.execute(text(create_table_sql))

df.to_sql("intents", engine, if_exists="replace", index=False)

print("CSV 已成功导入 MySQL 表 intents！")
