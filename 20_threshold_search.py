# 20_threshold_search.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid")

def main():
    X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURE_DIR, "y_val.npy"))
    model_path = os.path.join(MODELS_DIR, "LightGBM_baseline.pkl")
    out_csv = os.path.join(RESULTS_DIR, "threshold_metrics.csv")
    out_png = os.path.join(RESULTS_DIR, "threshold_prf_curve.png")

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 40) # 增加采样点，使曲线更圆滑
    rows = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # 找到最佳 F1 点
    best_row = df.loc[df["f1"].idxmax()]
    best_thr = best_row["threshold"]
    best_f1 = best_row["f1"]

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["precision"], label="Precision", color="#3498db", lw=2)
    plt.plot(df["threshold"], df["recall"], label="Recall", color="#f1c40f", lw=2)
    plt.plot(df["threshold"], df["f1"], label="F1-score", color="#e74c3c", lw=2.5)

    # 标出最佳阈值竖线
    plt.axvline(x=best_thr, color="#2c3e50", linestyle="--", alpha=0.6, label=f"Best F1 Thr={best_thr:.2f}")
    plt.scatter([best_thr], [best_f1], color="#e74c3c", s=50, zorder=5) # 标记最佳点

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Precision, Recall & F1 vs Threshold", fontsize=15, fontweight='bold')
    plt.legend(loc="lower left", frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"最佳阈值图已保存 (Best F1: {best_f1:.4f} @ {best_thr:.2f})")

if __name__ == "__main__":
    main()