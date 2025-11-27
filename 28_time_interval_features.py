# 28_time_interval_features.py
# 功能：
#   基于 train_with_behavior_1.csv 构造时间间隔特征（按 card1 分组）：
#     - time_since_last_tx_card1        距离上一次交易的时间差（秒）
#     - time_since_last_tx_card1_hours  距离上一次交易的时间差（小时）
#     - time_since_first_tx_card1       距离该用户第一笔交易的时间差（秒）
#
# 输入： data_stage/train_with_behavior_1.csv
# 输出： data_stage/train_with_behavior_2.csv

import os
import numpy as np
import pandas as pd
from config import DATA_STAGE_DIR

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_with_behavior_1.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_with_behavior_2.csv")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    required_cols = ["TransactionDT", "card1"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("警告：缺少必要列:", missing, "将直接拷贝输入为输出。")
        df.to_csv(out_path, index=False)
        print("已保存到:", out_path)
        return

    # 按 card1 + TransactionDT 排序，便于 diff
    df = df.sort_values(["card1", "TransactionDT"])
    print("按 card1, TransactionDT 排序完成。")

    grp = df.groupby("card1")["TransactionDT"]

    # 距离上一笔交易的时间差（秒）
    df["time_since_last_tx_card1"] = grp.diff().fillna(-1).astype("float32")

    # 换算为小时
    df["time_since_last_tx_card1_hours"] = (df["time_since_last_tx_card1"] / 3600.0).astype("float32")

    # 距离该用户第一笔交易的时间差（秒）
    first_dt = grp.transform("min")
    df["time_since_first_tx_card1"] = (df["TransactionDT"] - first_dt).astype("float32")

    # 恢复原始顺序
    df = df.sort_index()
    print("添加时间间隔特征后形状:", df.shape)

    df.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
