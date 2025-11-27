# 30_xgb_catboost_models.py
# 功能：
#   基于 X_train / X_val 训练两个额外模型：
#     - XGBoost_baseline
#     - CatBoost_baseline
#   输出各自指标到 results/metrics_more_models.csv
#   模型保存到 models/ 目录

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

MAX_TRAIN_SAMPLES = 300000  # 为了训练时间可控，对训练集做子样本

def eval_model(name, model, X_tr, y_tr, X_val, y_val):
    print(f"\n=== 训练模型：{name} ===")
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)

    print(f"{name} - AUC: {auc:.4f}, ACC: {acc:.4f}, "
          f"PREC: {prec:.4f}, REC: {rec:.4f}, F1: {f1:.4f}")

    return {
        "model": name,
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

def main():
    X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURE_DIR, "y_val.npy"))

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    # 子样本抽样（为了控制训练时间）
    n_samples = X_train.shape[0]
    sample_n = min(MAX_TRAIN_SAMPLES, n_samples)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(n_samples, size=sample_n, replace=False)
    X_tr_sub = X_train[idx]
    y_tr_sub = y_train[idx]
    print(f"使用子样本训练，大小: {sample_n} / {n_samples}")

    # 计算正负样本比例，用于不平衡设置
    pos = (y_tr_sub == 1).sum()
    neg = (y_tr_sub == 0).sum()
    if pos == 0:
        pos = 1
    scale_pos_weight = neg / pos
    print(f"子样本中正样本: {pos}, 负样本: {neg}, scale_pos_weight: {scale_pos_weight:.2f}")

    metrics = []

    # 1) XGBoost_baseline
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method="hist",   # 如果你的环境支持 GPU，可以改为 "gpu_hist"
    )
    m_xgb = eval_model("XGBoost_baseline", xgb, X_tr_sub, y_tr_sub, X_val, y_val)
    joblib.dump(xgb, os.path.join(MODELS_DIR, "XGBoost_baseline.pkl"))
    metrics.append(m_xgb)

    # 2) CatBoost_baseline
    # 注意：CatBoost 内部可以自动处理不平衡，我们这里通过 class_weights 显式指定
    # 为减少输出，verbose 设为 100
    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=100,
        class_weights=[1.0, float(scale_pos_weight)],
    )
    m_cat = eval_model("CatBoost_baseline", cat, X_tr_sub, y_tr_sub, X_val, y_val)
    joblib.dump(cat, os.path.join(MODELS_DIR, "CatBoost_baseline.pkl"))
    metrics.append(m_cat)

    # 保存指标
    out_csv = os.path.join(RESULTS_DIR, "metrics_more_models.csv")
    pd.DataFrame(metrics).to_csv(out_csv, index=False)
    print("\nXGBoost / CatBoost 指标已保存到:", out_csv)

if __name__ == "__main__":
    main()
