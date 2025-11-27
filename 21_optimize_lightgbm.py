# 21_optimize_lightgbm.py
# 功能：
#   基于 X_train / X_val 对 LightGBM 进行小范围超参搜索，
#   得到调优后的 LightGBM_tuned 模型和对应指标。

import os
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

def main():
    X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURE_DIR, "y_val.npy"))

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    param_grid = {
        "num_leaves": [32, 64],
        "min_child_samples": [20, 60],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    best_auc = -1.0
    best_model = None
    best_params = None
    best_metrics = None

    for num_leaves in param_grid["num_leaves"]:
        for min_child_samples in param_grid["min_child_samples"]:
            for subsample in param_grid["subsample"]:
                for colsample_bytree in param_grid["colsample_bytree"]:
                    params = dict(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=num_leaves,
                        min_child_samples=min_child_samples,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        objective="binary",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )
                    print("\n尝试参数:", params)

                    model = LGBMClassifier(**params)
                    model.fit(X_train, y_train)

                    y_prob = model.predict_proba(X_val)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)

                    auc = roc_auc_score(y_val, y_prob)
                    acc = accuracy_score(y_val, y_pred)
                    prec = precision_score(y_val, y_pred, zero_division=0)
                    rec = recall_score(y_val, y_pred, zero_division=0)
                    f1 = f1_score(y_val, y_pred, zero_division=0)

                    print(f"AUC: {auc:.4f}, ACC: {acc:.4f}, "
                          f"PREC: {prec:.4f}, REC: {rec:.4f}, F1: {f1:.4f}")

                    if auc > best_auc:
                        best_auc = auc
                        best_model = model
                        best_params = params
                        best_metrics = {
                            "model": "LightGBM_tuned",
                            "auc": auc,
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "params": str(params),
                        }

    tuned_path = os.path.join(MODELS_DIR, "LightGBM_tuned.pkl")
    joblib.dump(best_model, tuned_path)

    metrics_path = os.path.join(RESULTS_DIR, "metrics_tuned.csv")
    pd.DataFrame([best_metrics]).to_csv(metrics_path, index=False)

    print("\n最优参数:", best_params)
    print("最优 AUC:", best_auc)
    print("调参模型已保存到:", tuned_path)
    print("调参指标已保存到:", metrics_path)

if __name__ == "__main__":
    main()
