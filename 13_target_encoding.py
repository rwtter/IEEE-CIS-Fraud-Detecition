# 13_target_encoding.py
# 功能：对低/中基数类别特征做 target mean encoding，新增 <col>__te 列
#       映射关系与全局均值保存到 features/target_encoding_maps.pkl

import os
import pandas as pd
import numpy as np
import joblib
from config import DATA_STAGE_DIR, FEATURE_DIR, TARGET_COL

MAX_TE_CARD = 100  # 只对基数不太大的类别列做 target encoding

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_freqenc.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_encoded_full.csv")
    map_path = os.path.join(FEATURE_DIR, "target_encoding_maps.pkl")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"找不到目标列 {TARGET_COL}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("类别特征数量:", len(cat_cols))

    te_maps = {}
    global_mean = df[TARGET_COL].mean()

    for col in cat_cols:
        nunique = df[col].nunique()
        if nunique < 2 or nunique > MAX_TE_CARD:
            continue

        grp = df.groupby(col)[TARGET_COL].mean()
        mapping = grp.to_dict()
        te_maps[col] = mapping

        new_col = col + "__te"
        df[new_col] = df[col].map(mapping).fillna(global_mean).astype("float32")

    joblib.dump({"global_mean": global_mean, "maps": te_maps}, map_path)
    print("target encoding 映射已保存到:", map_path)

    df.to_csv(out_path, index=False)
    print("添加 target encoding 特征后保存到:", out_path)

if __name__ == "__main__":
    main()
