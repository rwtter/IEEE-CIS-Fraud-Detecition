# 25_leakage_detection.py
# 功能：
#   基于 train_crossed.csv，对数值特征与目标列 isFraud 做相关性分析，
#   输出与目标高度相关（|corr| 较大）的特征列表，辅助判断特征泄露风险。

import os
import pandas as pd
import numpy as np
from config import DATA_STAGE_DIR, LOGS_DIR, TARGET_COL

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_crossed.csv")
    log_path = os.path.join(LOGS_DIR, "leakage_check.txt")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    if TARGET_COL not in df.columns:
        print(f"未找到目标列 {TARGET_COL}，退出。")
        return

    # 只对数值列做相关性分析
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL not in num_cols:
        num_cols.append(TARGET_COL)

    corr = df[num_cols].corr()[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)

    top_pos = corr.head(30)
    top_neg = corr.tail(30)

    # 标记“可疑高相关特征”
    suspicious = corr[abs(corr) > 0.9]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== 与目标正相关最高的前 30 个特征 ===\n")
        f.write(str(top_pos) + "\n\n")

        f.write("=== 与目标负相关最高的前 30 个特征 ===\n")
        f.write(str(top_neg) + "\n\n")

        f.write("=== 绝对相关系数大于 0.9 的潜在泄露特征（如有） ===\n")
        if suspicious.empty:
            f.write("无 |corr| > 0.9 的特征。\n")
        else:
            f.write(str(suspicious) + "\n")

    print("特征泄露检查结果已写入:", log_path)

if __name__ == "__main__":
    main()
