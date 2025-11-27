# 22_kfold_training.py
# 功能：
#   使用 StratifiedKFold 在 X_fs / y_all 上做 K 折交叉验证，
#   针对 LightGBM 计算每折 AUC，并输出平均性能。

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from config import FEATURE_DIR, RESULTS_DIR, RANDOM_STATE

def main():
    X_path = os.path.join(FEATURE_DIR, "X_fs.npy")
    y_path = os.path.join(FEATURE_DIR, "y_all.npy")
    out_csv = os.path.join(RESULTS_DIR, "kfold_cv_results.csv")

    X = np.load(X_path)
    y = np.load(y_path)
    print("X_fs 形状:", X.shape, "y_all 长度:", len(y))

    n_splits = 5
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    rows = []
    fold_idx = 1

    for train_idx, val_idx in skf.split(X, y):
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE + fold_idx,
        )
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        print(f"Fold {fold_idx} AUC: {auc:.4f}")

        rows.append({
            "fold": fold_idx,
            "auc": auc,
            "n_train": len(y_tr),
            "n_val": len(y_val),
        })

        fold_idx += 1

    df = pd.DataFrame(rows)
    df.loc["mean"] = ["mean", df["auc"].mean(), df["n_train"].mean(), df["n_val"].mean()]
    df.to_csv(out_csv, index=False)
    print("\nK 折结果已保存到:", out_csv)
    print("平均 AUC:", df.loc[df["fold"] != "mean", "auc"].mean())

if __name__ == "__main__":
    main()
