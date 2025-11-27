# 31_advanced_hparam_search.py
# 功能：
#   对 LightGBM 进行随机超参数搜索（成本敏感指标）：
#     - 每轮计算 AUC、ACC、PREC、REC、F1、COST
#     - COST = FN_cost * FN + FP_cost * FP（越小越好）
#   输出：
#     - 最佳模型: models/LightGBM_cost_sensitive.pkl
#     - 最佳指标: results/metrics_cost_sensitive.csv
#     - 搜索日志: results/hparam_search_log.csv

import os
import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

N_ITER = 20      # 随机搜索轮数，可根据时间调整
FN_COST = 5.0    # 漏判欺诈的成本权重
FP_COST = 1.0    # 误判正常的成本权重

def sample_params(rng: np.random.Generator):
    """随机采样一组 LightGBM 超参数"""
    num_leaves = int(rng.integers(32, 128))
    min_child_samples = int(rng.integers(20, 200))
    max_depth = int(rng.integers(4, 12))
    subsample = float(rng.uniform(0.6, 1.0))
    colsample_bytree = float(rng.uniform(0.6, 1.0))
    reg_alpha = float(rng.uniform(0.0, 1.0))
    reg_lambda = float(rng.uniform(0.0, 1.0))
    min_split_gain = float(rng.uniform(0.0, 0.5))
    n_estimators = int(rng.integers(200, 600))

    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_split_gain=min_split_gain,
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return params

def compute_cost(y_true, y_pred):
    """根据混淆矩阵计算成本：FN_COST * FN + FP_COST * FP"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = FN_COST * fn + FP_COST * fp
    return cost, tn, fp, fn, tp

def main():
    X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURE_DIR, "y_val.npy"))

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("X_train:", X_train.shape, "X_val:", X_val.shape)
    print(f"随机搜索轮数: {N_ITER}, FN_COST={FN_COST}, FP_COST={FP_COST}")

    rng = np.random.default_rng(RANDOM_STATE + 123)

    best_cost = float("inf")
    best_auc = -1.0
    best_model = None
    best_params = None
    best_metrics = None

    log_rows = []

    for i in range(1, N_ITER + 1):
        print(f"\n=== Random Search Iteration {i}/{N_ITER} ===")
        params = sample_params(rng)
        print("采样参数:", params)

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)
        cost, tn, fp, fn, tp = compute_cost(y_val, y_pred)

        print(f"AUC: {auc:.4f}, ACC: {acc:.4f}, PREC: {prec:.4f}, "
              f"REC: {rec:.4f}, F1: {f1:.4f}, COST: {cost:.2f}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        row = {
            "iter": i,
            "auc": auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cost": cost,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "params": str(params),
        }
        log_rows.append(row)

        # 以成本为主，AUC 为辅：
        #   先看 cost 更小，
        #   若 cost 差不多，可用 AUC 来打平
        if (cost < best_cost) or (cost == best_cost and auc > best_auc):
            best_cost = cost
            best_auc = auc
            best_model = model
            best_params = params
            best_metrics = row

    # 保存最佳模型和指标
    best_model_path = os.path.join(MODELS_DIR, "LightGBM_cost_sensitive.pkl")
    joblib.dump(best_model, best_model_path)

    best_metrics_path = os.path.join(RESULTS_DIR, "metrics_cost_sensitive.csv")
    pd.DataFrame([best_metrics]).to_csv(best_metrics_path, index=False)

    log_path = os.path.join(RESULTS_DIR, "hparam_search_log.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print("\n=== 搜索结束 ===")
    print("最佳参数:", best_params)
    print("最佳 COST:", best_cost)
    print("对应 AUC:", best_auc)
    print("最佳模型已保存到:", best_model_path)
    print("最佳指标已保存到:", best_metrics_path)
    print("完整搜索日志已保存到:", log_path)

if __name__ == "__main__":
    main()
