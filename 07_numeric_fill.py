# 07_numeric_fill.py
import os
import pandas as pd
import numpy as np
import joblib
from config import DATA_STAGE_DIR, FEATURE_DIR

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_basic_clean.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_num_filled.csv")
    med_path = os.path.join(FEATURE_DIR, "numeric_median.pkl")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("数值特征数量:", len(num_cols))

    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    joblib.dump(medians.to_dict(), med_path)
    print("数值特征中位数已保存到:", med_path)

    df.to_csv(out_path, index=False)
    print("填充数值缺失后保存到:", out_path)

if __name__ == "__main__":
    main()
