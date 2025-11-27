# 12_freq_encoding.py  （新版）
# 功能：
#   对 train_with_behavior_full.csv 中所有类别特征做频次编码，新增 <col>__freq 列，
#   映射关系保存到 features/freq_encoding_maps.pkl

import os
import pandas as pd
import numpy as np
import joblib
from config import DATA_STAGE_DIR, FEATURE_DIR

def main():
    # 改动点：这里改成 train_with_behavior_full.csv
    in_path = os.path.join(DATA_STAGE_DIR, "train_with_behavior_full.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_freqenc.csv")
    map_path = os.path.join(FEATURE_DIR, "freq_encoding_maps.pkl")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    # 自动识别类别特征：object / category
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("将进行频次编码的类别特征数量:", len(cat_cols))

    n = len(df)
    freq_maps = {}

    for col in cat_cols:
        vc = df[col].value_counts()
        freq = (vc / n).astype("float32")
        mapping = freq.to_dict()
        freq_maps[col] = mapping

        new_col = col + "__freq"
        df[new_col] = df[col].map(mapping).astype("float32")

    joblib.dump(freq_maps, map_path)
    print("频次编码映射已保存到:", map_path)

    df.to_csv(out_path, index=False)
    print("添加频次特征后保存到:", out_path)

if __name__ == "__main__":
    main()
