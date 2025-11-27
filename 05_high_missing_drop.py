# 05_high_missing_drop.py
import os
import pandas as pd
from config import DATA_STAGE_DIR, LOGS_DIR, HIGH_MISSING_THRESHOLD

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_merged.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_drop_missing.csv")
    log_path = os.path.join(LOGS_DIR, "dropped_high_missing.txt")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("原始形状:", df.shape)

    miss_rate = df.isna().mean()
    cols_drop = miss_rate[miss_rate > HIGH_MISSING_THRESHOLD].index.tolist()
    print(f"缺失率>{HIGH_MISSING_THRESHOLD} 将删除 {len(cols_drop)} 列")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"缺失率>{HIGH_MISSING_THRESHOLD} 被删除的列:\n")
        for c in cols_drop:
            f.write(c + "\n")

    df2 = df.drop(columns=cols_drop)
    print("删除高缺失列后形状:", df2.shape)
    df2.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
