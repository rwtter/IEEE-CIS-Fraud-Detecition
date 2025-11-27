# 33_streamlit_app.py
# 功能：
#   使用 Streamlit 构建一个简单的本地风控评分 Demo：
#     - 从验证集中选择样本编号
#     - 展示该交易的关键字段
#     - 显示模型预测的欺诈概率和预测标签

import os
import numpy as np
import pandas as pd
import joblib

import streamlit as st
from sklearn.model_selection import train_test_split

from config import (
    FEATURE_DIR,
    DATA_STAGE_DIR,
    MODELS_DIR,
    TARGET_COL,
    RANDOM_STATE,
    VALID_SIZE,
)

@st.cache_resource
def load_model_and_data():
    X_fs = np.load(os.path.join(FEATURE_DIR, "X_fs.npy"))
    y_all = np.load(os.path.join(FEATURE_DIR, "y_all.npy"))

    idx_all = np.arange(len(y_all))
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    X_val = X_fs[idx_val]
    y_val = y_all[idx_val]

    df_full = pd.read_csv(os.path.join(DATA_STAGE_DIR, "train_with_behavior_full.csv"))

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
    return X_val, y_val, df_full, idx_val, model_name, model

def main():
    st.title("信用卡欺诈检测 Demo（验证集样本评分）")

    X_val, y_val, df_full, idx_val, model_name, model = load_model_and_data()
    n_val = X_val.shape[0]

    st.sidebar.header("样本选择")
    sample_id = st.sidebar.slider("验证集样本编号", min_value=0, max_value=n_val - 1, value=0, step=1)

    ori_idx = idx_val[sample_id]
    row = df_full.iloc[ori_idx]

    st.subheader("交易信息")
    show_cols = [
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
    ]
    info = {}
    for col in show_cols:
        if col in row.index:
            info[col] = row[col]
    st.table(pd.DataFrame([info]).T.rename(columns={0: "value"}))

    x = X_val[sample_id : sample_id + 1]
    prob = model.predict_proba(x)[0, 1]
    pred = int(prob >= 0.5)

    st.subheader("模型预测结果")
    st.write(f"模型：{model_name}")
    st.metric("预测欺诈概率", f"{prob:.4f}")
    st.write(f"预测标签（threshold = 0.5）：**{pred}**  （1 = fraud，0 = normal）")

if __name__ == "__main__":
    main()
