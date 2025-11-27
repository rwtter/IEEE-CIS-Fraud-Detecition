# 19_baseline_models.py
# 功能：基于 X_train/X_val 训练两个基线模型：
#       LogisticRegression_baseline & LightGBM_baseline
#       输出 metrics_baseline.csv 和两个模型 pkl

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier

from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

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
    }, y_prob

def main():
    X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURE_DIR, "y_val.npy"))

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    metrics = []

    # 1) Logistic Regression（用子样本训练，避免过慢）
    n_samples = X_train.shape[0]
    sample_n = min(200000, n_samples)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(n_samples, size=sample_n, replace=False)
    X_tr_lr = X_train[idx]
    y_tr_lr = y_train[idx]
    print(f"\nLogistic 使用子样本大小: {sample_n}")

    log_reg = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        n_jobs=-1,
        class_weight="balanced",
    )
    m_lr, _ = eval_model("LogisticRegression_baseline", log_reg, X_tr_lr, y_tr_lr, X_val, y_val)
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "LogisticRegression_baseline.pkl"))
    metrics.append(m_lr)

    # 2) LightGBM baseline
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    m_lgbm, _ = eval_model("LightGBM_baseline", lgbm, X_train, y_train, X_val, y_val)
    joblib.dump(lgbm, os.path.join(MODELS_DIR, "LightGBM_baseline.pkl"))
    metrics.append(m_lgbm)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_baseline.csv"), index=False)
    print("\n基线指标已保存到 results/metrics_baseline.csv")

if __name__ == "__main__":
    main()
