# 09_time_features.py
import os
import pandas as pd
from config import DATA_STAGE_DIR

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_filled_all.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_with_time.csv")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    if "TransactionDT" not in df.columns:
        print("Warning: 未找到 TransactionDT 列，直接复制保存。")
        df.to_csv(out_path, index=False)
        print("已保存到:", out_path)
        return

    # TransactionDT 是一个秒级时间戳（相对起始点），这里做一些衍生特征
    dt = df["TransactionDT"]

    # 相对天数 / 周数 / 小时
    df["DT_day"] = (dt // (24 * 60 * 60)).astype("int32")
    df["DT_week"] = (df["DT_day"] // 7).astype("int32")
    df["DT_hour"] = ((dt // (60 * 60)) % 24).astype("int16")

    # 可选：一天中的时间段（早中晚夜）
    df["DT_day_part"] = pd.cut(
        df["DT_hour"],
        bins=[-1, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"]
    ).astype(str)

    print("添加时间特征后形状:", df.shape)
    df.to_csv(out_path, index=False)
    print("已保存到:", out_path)

if __name__ == "__main__":
    main()
