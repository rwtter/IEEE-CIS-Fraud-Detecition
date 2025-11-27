# 02_merge_raw_data.py
import os
import pandas as pd
from config import DATA_RAW_DIR, DATA_STAGE_DIR

def main():
    trans_path = os.path.join(DATA_RAW_DIR, "train_transaction.csv")
    id_path = os.path.join(DATA_RAW_DIR, "train_identity.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_merged.csv")

    print("读取 train_transaction.csv ...")
    trans = pd.read_csv(trans_path)
    print("train_transaction:", trans.shape)

    print("读取 train_identity.csv ...")
    identity = pd.read_csv(id_path)
    print("train_identity:", identity.shape)

    print("\n按 TransactionID 左连接合并 ...")
    merged = trans.merge(identity, on="TransactionID", how="left")
    print("合并后 train_merged:", merged.shape)

    merged.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
