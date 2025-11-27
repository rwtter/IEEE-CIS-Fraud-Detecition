# 26_visualize_all.py
# 功能：
#   1. 汇总 baseline / tuned / ensemble / more_models / cost_sensitive 的指标，
#      绘制 AUC & F1 对比柱状图
#   2. 基于 LightGBM_tuned 在验证集上绘制 ROC 曲线和 PR 曲线

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns  # 新增：用于美化绘图
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="notebook")

def plot_model_comparison():
    # 定义所有需要读取的指标文件路径
    files_map = {
        "baseline": os.path.join(RESULTS_DIR, "metrics_baseline.csv"),
        "tuned": os.path.join(RESULTS_DIR, "metrics_tuned.csv"),
        "ensemble": os.path.join(RESULTS_DIR, "metrics_ensemble.csv"),
        "more_models": os.path.join(RESULTS_DIR, "metrics_more_models.csv"),
        "cost_sensitive": os.path.join(RESULTS_DIR, "metrics_cost_sensitive.csv"),
    }

    dfs = []
    for tag, p in files_map.items():
        if os.path.exists(p):
            try:
                df_tmp = pd.read_csv(p)
                dfs.append(df_tmp)
            except Exception as e:
                print(f"读取 {p} 失败: {e}")

    if not dfs:
        print("未找到模型指标文件，跳过模型对比可视化。")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # 确保存在 model 列
    if "model" not in df_all.columns:
        print("指标文件中缺少 'model' 列，无法绘制模型对比图。")
        return

    # 去重（按 model 名，保留最后一次出现的记录）
    df_all = df_all.drop_duplicates(subset=["model"], keep="last")

    # 定义内部绘图函数，避免重复代码
    def _plot_bar(data, metric_col, title, filename):
        plt.figure(figsize=(10, 6))
        
        # 使用 seaborn barplot，自动配色
        ax = sns.barplot(
            x="model", 
            y=metric_col, 
            data=data, 
            hue="model", 
            palette="viridis", 
            legend=False
        )
        
        # 设置标题和标签
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.ylabel(metric_col.upper(), fontsize=12)
        plt.xlabel("Model", fontsize=12)
        
        # 防止 x 轴标签重叠，旋转 30 度
        plt.xticks(rotation=30, ha="right")
        
        # 调整 y 轴上限，留出空间写数字
        if not data[metric_col].empty:
            plt.ylim(0, data[metric_col].max() * 1.15)

        # 在柱子上标注数值
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}", 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', 
                            xytext=(0, 5), 
                            textcoords='offset points',
                            fontsize=10, fontweight='bold', color='black')
        
        # 去掉上右边框
        sns.despine()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"已保存: {filename}")

    # 1. 绘制 AUC 对比
    out_auc = os.path.join(RESULTS_DIR, "models_auc_comparison.png")
    _plot_bar(df_all, "auc", "Model AUC Comparison", out_auc)

    # 2. 绘制 F1 对比
    if "f1" in df_all.columns:
        out_f1 = os.path.join(RESULTS_DIR, "models_f1_comparison.png")
        _plot_bar(df_all, "f1", "Model F1 Score Comparison", out_f1)
    else:
        print("缺少 f1 列，跳过 F1 对比图。")

def plot_roc_pr_for_tuned():
    X_val_path = os.path.join(FEATURE_DIR, "X_val.npy")
    y_val_path = os.path.join(FEATURE_DIR, "y_val.npy")
    model_path = os.path.join(MODELS_DIR, "LightGBM_tuned.pkl")

    if not (os.path.exists(X_val_path) and os.path.exists(y_val_path) and os.path.exists(model_path)):
        print("缺少 X_val / y_val / LightGBM_tuned，跳过 ROC/PR 绘制。")
        return

    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    model = joblib.load(model_path)

    y_prob = model.predict_proba(X_val)[:, 1]

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_score = roc_auc_score(y_val, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2c3e50', lw=2.5, label=f"LightGBM Tuned (AUC={auc_score:.3f})")
    plt.fill_between(fpr, tpr, alpha=0.1, color='#2c3e50')  # 填充阴影
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label="Random Chance")
    
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - LightGBM Tuned", fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_roc = os.path.join(RESULTS_DIR, "roc_curve_tuned.png")
    plt.savefig(out_roc, dpi=300)
    plt.close()
    print("ROC 曲线已保存到:", out_roc)

    # --- PR Curve ---
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='#e74c3c', lw=2.5, label=f"AP={ap:.3f}")
    plt.fill_between(recall, precision, alpha=0.1, color='#e74c3c')  # 填充阴影
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve - LightGBM Tuned", fontsize=15, fontweight='bold')
    plt.legend(loc="best", frameon=True)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_pr = os.path.join(RESULTS_DIR, "pr_curve_tuned.png")
    plt.savefig(out_pr, dpi=300)
    plt.close()
    print("PR 曲线已保存到:", out_pr)

def main():
    plot_model_comparison()
    plot_roc_pr_for_tuned()
    print("综合可视化完成。")

if __name__ == "__main__":
    main()