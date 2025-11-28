# data_to_mysql.py
import pandas as pd
from sqlalchemy import create_engine, text
from config import train_file  # 假设你在 config.py 里定义了 train_file 路径
import pymysql
# 数据库配置
DB_USER = "root"
DB_PASSWORD = "mysql123"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "nlp_project"

# 创建数据库连接
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

# 读取 CSV 数据
df = pd.read_csv(train_file)


# 建表（如果不存在）
create_table_sql = """
CREATE TABLE IF NOT EXISTS intents (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    Sentence VARCHAR(1024),
    Category VARCHAR(128)
);
"""

with engine.begin() as conn:  # engine.begin() 会自动 commit/rollback
    conn.execute(text("DROP TABLE IF EXISTS intents"))
    conn.execute(text(create_table_sql))

# 导入数据到 MySQL 表
df.to_sql("intents", engine, if_exists="replace", index=False)

print("CSV 已成功导入 MySQL 表 intents！")
