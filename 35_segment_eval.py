# 35_segment_eval.py
# 功能：
#   对验证集上的模型表现进行“分段评估”：
#     - 按 TransactionAmt 金额分位数分段
#     - 按 DT_week 分段
#     - 按 DeviceType_norm 分组（若存在）
#   对每个 segment 计算：AUC, ACC, PREC, REC, F1, 支持数
#   结果保存到 results/segment_metrics.csv，并绘制美化后的柱状图

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns  # 新增：用于美化绘图

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from config import (
    FEATURE_DIR,
    DATA_STAGE_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TARGET_COL,
    RANDOM_STATE,
    VALID_SIZE,
)

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="notebook")

def compute_metrics(y_true, y_prob, y_pred):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # AUC 在只有单一类别时不可用，这里设为 NaN
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return auc, acc, prec, rec, f1

def plot_bar_refined(df, segment_col, value_col, title, out_path, palette="viridis"):
    """
    绘制美化后的分段指标柱状图
    """
    plt.figure(figsize=(8, 5))
    
    # 确保 X 轴数据是字符串，避免自动排序混乱
    df[segment_col] = df[segment_col].astype(str)
    
    # 绘制柱状图，使用 hue 避免 FutureWarning
    ax = sns.barplot(
        x=segment_col, 
        y=value_col, 
        data=df, 
        hue=segment_col, 
        palette=palette, 
        legend=False
    )
    
    # 标注数值
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold', color='black')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(value_col.upper(), fontsize=12)
    plt.xlabel("Segment", fontsize=12)
    
    # 旋转标签以防重叠
    plt.xticks(rotation=30, ha="right")
    
    # 调整 Y 轴范围留出顶部空间
    if not df[value_col].empty:
        plt.ylim(0, df[value_col].max() * 1.15)

    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"已保存美化图像: {out_path}")

def main():
    # 1. 加载特征 + 标签
    X_fs = np.load(os.path.join(FEATURE_DIR, "X_fs.npy"))
    y_all = np.load(os.path.join(FEATURE_DIR, "y_all.npy"))

    # 2. 复现 train/val 划分，得到验证集索引
    idx_all = np.arange(len(y_all))
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    X_val = X_fs[idx_val]
    y_val = y_all[idx_val]

    # 3. 原始带业务特征的数据 (用于提取 amt, week, device 等)
    df_full = pd.read_csv(os.path.join(DATA_STAGE_DIR, "train_with_behavior_full.csv"))
    df_val = df_full.iloc[idx_val].reset_index(drop=True)

    # 4. 加载模型（优先 cost_sensitive）
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
    print(f"使用模型: {model_name}")

    # 5. 预测
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    # === 分段 1: 按金额分段 (TransactionAmt) ===
    if "TransactionAmt" in df_val.columns:
        amt = df_val["TransactionAmt"].values
        # 使用分位数切分
        qs = np.quantile(amt, [0.0, 0.25, 0.5, 0.75, 1.0])
        qs = np.unique(qs)
        if len(qs) > 2:
            # duplicates='drop' 防止相同分位数报错
            df_val["amt_bin"] = pd.qcut(amt, q=min(4, len(qs) - 1), duplicates="drop")
            
            for b in df_val["amt_bin"].cat.categories:
                mask = df_val["amt_bin"] == b
                idx = np.where(mask.values)[0]
                if len(idx) == 0: continue
                
                auc, acc, prec, rec, f1 = compute_metrics(y_val[idx], y_prob[idx], y_pred[idx])
                rows.append({
                    "segment_type": "amount",
                    "segment": str(b),
                    "n": len(idx),
                    "auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                })
        else:
            print("TransactionAmt 分布过于集中，无法有效分段，跳过。")
    else:
        print("df_val 中不存在 TransactionAmt 列，跳过。")

    # === 分段 2: 按时间周分段 (DT_week) ===
    if "DT_week" in df_val.columns:
        weeks = df_val["DT_week"].values
        qs = np.quantile(weeks, [0.0, 0.25, 0.5, 0.75, 1.0])
        qs = np.unique(qs)
        if len(qs) > 2:
            df_val["week_bin"] = pd.qcut(weeks, q=min(4, len(qs) - 1), duplicates="drop")
            for b in df_val["week_bin"].cat.categories:
                mask = df_val["week_bin"] == b
                idx = np.where(mask.values)[0]
                if len(idx) == 0: continue
                
                auc, acc, prec, rec, f1 = compute_metrics(y_val[idx], y_prob[idx], y_pred[idx])
                rows.append({
                    "segment_type": "week",
                    "segment": str(b),
                    "n": len(idx),
                    "auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                })
        else:
            print("DT_week 分布过于集中，跳过。")
    else:
        print("df_val 中不存在 DT_week 列，跳过。")

    # === 分段 3: 按设备分段 (DeviceType_norm) ===
    if "DeviceType_norm" in df_val.columns:
        for dev, g in df_val.groupby("DeviceType_norm"):
            idx = g.index.values
            if len(idx) == 0: continue
            auc, acc, prec, rec, f1 = compute_metrics(y_val[idx], y_prob[idx], y_pred[idx])
            rows.append({
                "segment_type": "device",
                "segment": str(dev),
                "n": len(idx),
                "auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            })
    else:
        print("df_val 中不存在 DeviceType_norm 列，跳过。")

    # 6. 汇总与保存
    df_seg = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "segment_metrics.csv")
    df_seg.to_csv(out_csv, index=False)
    print("分段评估结果已保存到:", out_csv)

    # 7. 绘图 (使用美化后的 plot_bar_refined)
    
    # 金额分段图 (使用蓝色渐变 Blues_d)
    df_amt = df_seg[df_seg["segment_type"] == "amount"].copy()
    if not df_amt.empty:
        plot_bar_refined(
            df_amt,
            segment_col="segment",
            value_col="f1",
            title="Model F1-Score by Transaction Amount",
            out_path=os.path.join(RESULTS_DIR, "segment_amt_f1.png"),
            palette="Blues_d"
        )

    # 时间分段图 (使用紫色渐变 Purples_d)
    df_week = df_seg[df_seg["segment_type"] == "week"].copy()
    if not df_week.empty:
        plot_bar_refined(
            df_week,
            segment_col="segment",
            value_col="f1",
            title="Model F1-Score by Week",
            out_path=os.path.join(RESULTS_DIR, "segment_week_f1.png"),
            palette="Purples_d"
        )

    # 设备分段图 (使用多彩色 viridis)
    df_dev = df_seg[df_seg["segment_type"] == "device"].copy()
    if not df_dev.empty:
        plot_bar_refined(
            df_dev,
            segment_col="segment",
            value_col="f1",
            title="Model F1-Score by Device Type",
            out_path=os.path.join(RESULTS_DIR, "segment_device_f1.png"),
            palette="viridis"
        )

if __name__ == "__main__":
    main()