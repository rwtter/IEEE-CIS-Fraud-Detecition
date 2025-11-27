# 29_group_fraud_rate_features.py
# 功能：
#   基于 train_with_behavior_2.csv 构造“组合 fraud rate”特征：
#     - fr_card1_addr1                   (card1, addr1) 组合的欺诈率
#     - fr_card1_Pemailprov              (card1, P_emaildomain_provider) 组合的欺诈率
#
# 输入： data_stage/train_with_behavior_2.csv
# 输出： data_stage/train_with_behavior_full.csv

import os
import pandas as pd
from config import DATA_STAGE_DIR, TARGET_COL

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_with_behavior_2.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_with_behavior_full.csv")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    if TARGET_COL not in df.columns:
        print(f"警告：未找到目标列 {TARGET_COL}，无法计算 fraud rate，直接拷贝输入为输出。")
        df.to_csv(out_path, index=False)
        print("已保存到:", out_path)
        return

    # 1) (card1, addr1) 组合的欺诈率
    if "card1" in df.columns and "addr1" in df.columns:
        grp_cols = ["card1", "addr1"]
        print("计算 (card1, addr1) 组合 fraud rate ...")
        fr_c1_a1 = df.groupby(grp_cols)[TARGET_COL].transform("mean")
        df["fr_card1_addr1"] = fr_c1_a1.astype("float32")
    else:
        print("未找到 card1 或 addr1，跳过 fr_card1_addr1。")

    # 2) (card1, P_emaildomain_provider) 组合的欺诈率
    if "card1" in df.columns and "P_emaildomain_provider" in df.columns:
        grp_cols2 = ["card1", "P_emaildomain_provider"]
        print("计算 (card1, P_emaildomain_provider) 组合 fraud rate ...")
        fr_c1_ep = df.groupby(grp_cols2)[TARGET_COL].transform("mean")
        df["fr_card1_Pemailprov"] = fr_c1_ep.astype("float32")
    else:
        print("未找到 card1 或 P_emaildomain_provider，跳过 fr_card1_Pemailprov。")

    print("添加 fraud rate 特征后形状:", df.shape)
    df.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
