# 24_drift_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_STAGE_DIR, RESULTS_DIR, TARGET_COL

sns.set_theme(style="white", context="talk") # 使用简洁白底

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_with_time.csv")
    out_csv = os.path.join(RESULTS_DIR, "drift_by_week.csv")
    out_png = os.path.join(RESULTS_DIR, "drift_fraud_rate_by_week.png")

    print("读取:", in_path)
    df = pd.read_csv(in_path)

    if "DT_week" not in df.columns or TARGET_COL not in df.columns:
        print("缺少必要列，退出。")
        return

    tmp = df[["DT_week", TARGET_COL, "TransactionAmt"]].copy()
    grouped = tmp.groupby("DT_week").agg(
        n_samples=("DT_week", "size"),
        fraud_rate=(TARGET_COL, "mean"),
    ).reset_index().sort_values("DT_week")
    
    grouped.to_csv(out_csv, index=False)

    # 绘图：双轴图 (左轴 Fraud Rate, 右轴 Sample Count)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制样本量 (柱状背景)
    color_bar = 'lightgray'
    ax2 = ax1.twinx()
    ax2.bar(grouped["DT_week"], grouped["n_samples"], color=color_bar, alpha=0.5, label='Sample Count', width=0.8)
    ax2.set_ylabel('Transaction Count', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.grid(False) # 右轴不要网格，免得乱

    # 绘制欺诈率 (折线)
    color_line = '#c0392b' # 深红色
    sns.lineplot(x=grouped["DT_week"], y=grouped["fraud_rate"], ax=ax1, 
                 color=color_line, marker='o', linewidth=2.5, label='Fraud Rate')
    
    ax1.set_xlabel('Week (DT_week)', fontsize=12)
    ax1.set_ylabel('Fraud Rate', color=color_line, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_line)
    
    plt.title("Fraud Rate Trend by Week\n(with Transaction Volume)", fontsize=16, fontweight='bold', pad=20)
    
    # 强制显示图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # 实际上 bar 没有 line handle，手动处理一下图例也行，或者简单的 title 即可
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Fraud Rate 漂移趋势图已保存:", out_png)

if __name__ == "__main__":
    main()