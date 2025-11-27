# 03_missing_analysis.py
import os
import pandas as pd
from config import DATA_STAGE_DIR, LOGS_DIR

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_merged.csv")
    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("train_merged 形状:", df.shape)

    miss_rate = df.isna().mean().sort_values(ascending=False)
    out_csv = os.path.join(LOGS_DIR, "missing_rate.csv")
    miss_rate.to_csv(out_csv, header=["missing_rate"])

    print("缺失率统计已保存到:", out_csv)
    print("缺失率最高的前 20 列：")
    print(miss_rate.head(20))

if __name__ == "__main__":
    main()
