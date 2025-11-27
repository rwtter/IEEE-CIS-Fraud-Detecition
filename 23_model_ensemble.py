# 23_model_ensemble.py  （新版）
# 功能：
#   在验证集上加载多个模型：
#       LogisticRegression_baseline
#       LightGBM_baseline
#       LightGBM_tuned
#       XGBoost_baseline
#       CatBoost_baseline
#   计算各自指标，并额外计算一个简单平均集成模型 Ensemble_avg。

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR

def eval_from_proba(name, y_val, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)
    return {
        "model": name,
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

def main():
    X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURE_DIR, "y_val.npy"))

    # 所有可能的模型
    paths = {
        "LogisticRegression_baseline": os.path.join(MODELS_DIR, "LogisticRegression_baseline.pkl"),
        "LightGBM_baseline":           os.path.join(MODELS_DIR, "LightGBM_baseline.pkl"),
        "LightGBM_tuned":              os.path.join(MODELS_DIR, "LightGBM_tuned.pkl"),
        "XGBoost_baseline":            os.path.join(MODELS_DIR, "XGBoost_baseline.pkl"),
        "CatBoost_baseline":           os.path.join(MODELS_DIR, "CatBoost_baseline.pkl"),
    }

    model_probs = {}
    metrics = []

    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"警告: 未找到模型 {name} 对应文件 {p}，跳过。")
            continue
        print(f"加载模型: {name} ({p})")
        model = joblib.load(p)
        y_prob = model.predict_proba(X_val)[:, 1]
        model_probs[name] = y_prob
        m = eval_from_proba(name, y_val, y_prob)
        print(f"{name} - AUC: {m['auc']:.4f}, F1: {m['f1']:.4f}")
        metrics.append(m)

    # 简单平均集成（只在至少有两个模型时才做）
    if len(model_probs) >= 2:
        print("\n计算简单平均集成 Ensemble_avg ...")
        probs_stack = np.vstack(list(model_probs.values()))
        y_prob_ens = probs_stack.mean(axis=0)
        m_ens = eval_from_proba("Ensemble_avg", y_val, y_prob_ens)
        print(f"Ensemble_avg - AUC: {m_ens['auc']:.4f}, F1: {m_ens['f1']:.4f}")
        metrics.append(m_ens)
    else:
        print("可用模型不足 2 个，跳过集成。")

    out_csv = os.path.join(RESULTS_DIR, "metrics_ensemble.csv")
    pd.DataFrame(metrics).to_csv(out_csv, index=False)
    print("\n集成对比结果已保存到:", out_csv)

if __name__ == "__main__":
    main()
