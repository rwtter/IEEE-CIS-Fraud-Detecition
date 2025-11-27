# 14_cross_features.py
# 功能：在 train_encoded_full 基础上构造少量人工交叉特征，输出 train_crossed.csv

import os
import pandas as pd
from config import DATA_STAGE_DIR

def make_cross_col(df, col_a, col_b, new_col):
    if col_a in df.columns and col_b in df.columns:
        df[new_col] = df[col_a].astype(str) + "_" + df[col_b].astype(str)
        print(f"已生成交叉特征: {new_col}")
    else:
        print(f"跳过交叉特征 {new_col}，因为 {col_a} 或 {col_b} 不存在。")

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_encoded_full.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_crossed.csv")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    # 典型风控交叉特征
    make_cross_col(df, "card1", "addr1", "card1_addr1")
    make_cross_col(df, "ProductCD", "card4", "ProductCD_card4")
    make_cross_col(df, "DeviceType_norm", "ProductCD", "DeviceType_ProductCD")

    print("添加交叉特征后形状:", df.shape)
    df.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
