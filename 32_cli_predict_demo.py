# 32_cli_predict_demo.py
# 功能：
#   提供一个命令行小工具：
#     - 选择验证集中的某个样本（或随机样本）
#     - 展示该交易的关键业务字段
#     - 输出 LightGBM_tuned 模型对该样本的欺诈概率和预测标签

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from config import (
    FEATURE_DIR,
    DATA_STAGE_DIR,
    MODELS_DIR,
    TARGET_COL,
    RANDOM_STATE,
    VALID_SIZE,
)

def load_data_and_model():
    # 特征与标签（最终特征）
    X_fs = np.load(os.path.join(FEATURE_DIR, "X_fs.npy"))
    y_all = np.load(os.path.join(FEATURE_DIR, "y_all.npy"))

    # 复现 train/val 划分，得到验证集索引
    idx_all = np.arange(len(y_all))
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    X_val = X_fs[idx_val]
    y_val = y_all[idx_val]

    # 原始带业务特征的数据（行为 + 时间 + 设备 + 邮箱）
    df_full = pd.read_csv(os.path.join(DATA_STAGE_DIR, "train_with_behavior_full.csv"))

    # 尝试优先加载 LightGBM_cost_sensitive，否则退回 LightGBM_tuned
    model_path_cost = os.path.join(MODELS_DIR, "LightGBM_cost_sensitive.pkl")
    model_path_tuned = os.path.join(MODELS_DIR, "LightGBM_tuned.pkl")

    if os.path.exists(model_path_cost):
        model_path = model_path_cost
        model_name = "LightGBM_cost_sensitive"
    elif os.path.exists(model_path_tuned):
        model_path = model_path_tuned
        model_name = "LightGBM_tuned"
    else:
        raise FileNotFoundError(
            "未找到 LightGBM_cost_sensitive.pkl 或 LightGBM_tuned.pkl，请先运行 21 / 31。"
        )

    model = joblib.load(model_path)
    print(f"已加载模型: {model_name} ({model_path})")

    return X_val, y_val, df_full, idx_val, model_name, model

def show_sample_info(sample_id, X_val, y_val, df_full, idx_val, model_name, model):
    n_val = X_val.shape[0]
    if sample_id < 0 or sample_id >= n_val:
        print(f"样本编号超出范围，应该在 [0, {n_val-1}] 之间。")
        return

    ori_idx = idx_val[sample_id]
    row = df_full.iloc[ori_idx]

    print("\n========== 样本信息 ==========")
    print(f"验证集样本编号: {sample_id}")
    print(f"原始行索引: {ori_idx}")

    # 尝试输出一些常见字段，如果不存在就跳过
    for col in [
        "TransactionID",
        "TransactionAmt",
        "DT_week",
        "DT_day",
        "DT_hour",
        "card1",
        "addr1",
        "DeviceType_norm",
        "DeviceInfo_brand",
        "P_emaildomain_provider",
        TARGET_COL,
    ]:
        if col in row.index:
            print(f"{col}: {row[col]}")

    # 模型预测
    x = X_val[sample_id : sample_id + 1]
    prob = model.predict_proba(x)[0, 1]
    pred = int(prob >= 0.5)

    print("\n========== 模型预测 ==========")
    print(f"模型: {model_name}")
    print(f"预测欺诈概率: {prob:.4f}")
    print(f"预测标签 (threshold=0.5): {pred}  (1 = fraud, 0 = normal)")
    print("==============================\n")

def main():
    X_val, y_val, df_full, idx_val, model_name, model = load_data_and_model()
    n_val = X_val.shape[0]
    print(f"验证集中共有 {n_val} 个样本。")

    while True:
        s = input(f"请输入验证集样本编号（0 ~ {n_val-1}），或输入 'r' 随机选择，'q' 退出: ").strip()
        if s.lower() == "q":
            print("退出。")
            break
        if s.lower() == "r":
            sample_id = int(np.random.randint(0, n_val))
        else:
            try:
                sample_id = int(s)
            except ValueError:
                print("输入无效，请重新输入。")
                continue

        show_sample_info(sample_id, X_val, y_val, df_full, idx_val, model_name, model)

if __name__ == "__main__":
    main()
