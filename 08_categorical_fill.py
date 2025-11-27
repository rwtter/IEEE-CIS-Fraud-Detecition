# 08_categorical_fill.py
import os
import pandas as pd
from config import DATA_STAGE_DIR

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_num_filled.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_filled_all.csv")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("类别特征数量:", len(cat_cols))

    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna("missing").astype(str)

    df.to_csv(out_path, index=False)
    print("填充类别缺失后保存到:", out_path)

if __name__ == "__main__":
    main()
