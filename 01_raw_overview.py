# 01_raw_overview.py
import os
import pandas as pd
from config import DATA_RAW_DIR, LOGS_DIR, TARGET_COL

def main():
    trans_path = os.path.join(DATA_RAW_DIR, "train_transaction.csv")
    id_path = os.path.join(DATA_RAW_DIR, "train_identity.csv")

    print("读取 train_transaction.csv ...")
    trans = pd.read_csv(trans_path)
    print("train_transaction 维度:", trans.shape)

    print("读取 train_identity.csv ...")
    identity = pd.read_csv(id_path)
    print("train_identity 维度:", identity.shape)

    overview_path = os.path.join(LOGS_DIR, "raw_overview.txt")
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write(f"train_transaction shape: {trans.shape}\n")
        f.write(f"train_identity shape: {identity.shape}\n\n")

        f.write("train_transaction columns:\n")
        f.write(", ".join(trans.columns) + "\n\n")

        if TARGET_COL in trans.columns:
            f.write(f"Target column: {TARGET_COL}\n")
            f.write("Target distribution:\n")
            f.write(str(trans[TARGET_COL].value_counts()) + "\n")

    print("概览信息已写入:", overview_path)

if __name__ == "__main__":
    main()
