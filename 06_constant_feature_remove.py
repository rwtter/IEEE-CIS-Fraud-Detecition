# 06_constant_feature_remove.py
import os
import pandas as pd
from config import DATA_STAGE_DIR, LOGS_DIR

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_drop_missing.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_basic_clean.csv")
    log_path = os.path.join(LOGS_DIR, "dropped_constant_cols.txt")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    nunique = df.nunique()
    cols_drop = nunique[nunique <= 1].index.tolist()
    print("常数列数量:", len(cols_drop))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("被删除的常数列:\n")
        for c in cols_drop:
            f.write(c + "\n")

    df2 = df.drop(columns=cols_drop)
    print("删除常数列后形状:", df2.shape)
    df2.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
