# 27_user_behavior_features.py
# 功能：
#   基于 train_with_email.csv 构造用户行为特征（按 card1 聚合）：
#     - card1_txn_cnt           用户总交易次数
#     - card1_amt_mean          用户历史交易金额均值
#     - card1_amt_std           用户历史交易金额标准差
#     - amt_div_card1_mean      当前金额 / 历史均值
#     - card1_recent3_amt_mean  最近 3 笔交易金额均值（按时间排序）
#     - card1_recent3_amt_std   最近 3 笔交易金额标准差
#   可选：如存在 addr1，则构造 card1_addr1_txn_cnt、card1_addr1_amt_mean 等特征。
#
# 输入： data_stage/train_with_email.csv
# 输出： data_stage/train_with_behavior_1.csv

import os
import numpy as np
import pandas as pd
from config import DATA_STAGE_DIR

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_with_email.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_with_behavior_1.csv")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    required_cols = ["TransactionAmt", "TransactionDT", "card1"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("警告：缺少必要列:", missing, "将直接拷贝输入为输出。")
        df.to_csv(out_path, index=False)
        print("已保存到:", out_path)
        return

    # 先按 card1 + TransactionDT 排序，便于构造“最近 N 笔”特征
    df = df.sort_values(["card1", "TransactionDT"])
    print("按 card1, TransactionDT 排序完成。")

    # 1) 按 card1 聚合统计
    grp_card1 = df.groupby("card1")

    df["card1_txn_cnt"] = grp_card1["TransactionAmt"].transform("count").astype("int32")
    df["card1_amt_mean"] = grp_card1["TransactionAmt"].transform("mean").astype("float32")
    df["card1_amt_std"] = grp_card1["TransactionAmt"].transform("std").fillna(0).astype("float32")

    # 金额 / 历史均值
    eps = 1e-6
    df["amt_div_card1_mean"] = (df["TransactionAmt"] / (df["card1_amt_mean"] + eps)).astype("float32")

    # 2) 最近 3 笔交易的金额统计（滚动窗口）
    rolling_mean = grp_card1["TransactionAmt"].rolling(window=3, min_periods=1).mean()
    rolling_std = grp_card1["TransactionAmt"].rolling(window=3, min_periods=1).std()

    df["card1_recent3_amt_mean"] = rolling_mean.reset_index(level=0, drop=True).astype("float32")
    df["card1_recent3_amt_std"] = rolling_std.reset_index(level=0, drop=True).fillna(0).astype("float32")

    # 3) 如存在 addr1，再构造 (card1, addr1) 聚合统计
    if "addr1" in df.columns:
        key_col = "card1_addr1_key"
        df[key_col] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)

        grp_c1_a1 = df.groupby(key_col)
        df["card1_addr1_txn_cnt"] = grp_c1_a1["TransactionAmt"].transform("count").astype("int32")
        df["card1_addr1_amt_mean"] = grp_c1_a1["TransactionAmt"].transform("mean").astype("float32")
    else:
        print("未找到 addr1 列，跳过 (card1, addr1) 行为统计特征。")

    # 恢复原始行顺序（按 index 排序）
    df = df.sort_index()
    print("添加行为特征后形状:", df.shape)

    df.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
