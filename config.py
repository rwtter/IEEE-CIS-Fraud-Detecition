# config.py
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 各类子目录
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")      # 原始 csv
DATA_STAGE_DIR = os.path.join(BASE_DIR, "data_stage")  # 中间处理结果
FEATURE_DIR = os.path.join(BASE_DIR, "features")       # 特征矩阵 & 元信息
MODELS_DIR = os.path.join(BASE_DIR, "models")          # 模型文件
RESULTS_DIR = os.path.join(BASE_DIR, "results")        # 指标 & 图
LOGS_DIR = os.path.join(BASE_DIR, "logs")              # 文本日志

# 目标列
TARGET_COL = "isFraud"

# 随机种子 & 验证集比例（后面会用到）
RANDOM_STATE = 42
VALID_SIZE = 0.2

# 清洗阈值
HIGH_MISSING_THRESHOLD = 0.9   # 缺失率超过这个就删列
VARIANCE_THRESHOLD = 1e-5      # 方差过滤阈值（后面脚本用）

# 确保目录存在
for d in [DATA_STAGE_DIR, FEATURE_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
