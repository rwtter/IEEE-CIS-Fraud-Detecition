# 15_assemble_features.py
# 功能：从 train_crossed 中拼装最终特征矩阵 X_all 和标签 y_all，
#       只保留数值型特征（包括 __freq/__te），保存到 features/

import os
import pandas as pd
import numpy as np
import joblib
from config import DATA_STAGE_DIR, FEATURE_DIR, TARGET_COL

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_crossed.csv")
    X_path = os.path.join(FEATURE_DIR, "X_all.npy")
    y_path = os.path.join(FEATURE_DIR, "y_all.npy")
    feat_path = os.path.join(FEATURE_DIR, "feature_names_all.pkl")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"找不到目标列 {TARGET_COL}")

    y = df[TARGET_COL].values.astype("int8")
    df_feats = df.drop(columns=[TARGET_COL])

    # 数值特征（包括原始数值、__freq、__te 等）
    num_cols = df_feats.select_dtypes(include=[np.number]).columns.tolist()
    print("数值特征数量:", len(num_cols))

    X = df_feats[num_cols].values.astype("float32")

    np.save(X_path, X)
    np.save(y_path, y)
    joblib.dump(num_cols, feat_path)

    print("X_all 形状:", X.shape, "y_all 长度:", len(y))
    print("特征名数量:", len(num_cols))
    print("X_all 保存到:", X_path)
    print("y_all 保存到:", y_path)
    print("feature_names_all 保存到:", feat_path)

if __name__ == "__main__":
    main()
