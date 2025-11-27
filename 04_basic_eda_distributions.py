# 04_basic_eda_distributions.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_STAGE_DIR, RESULTS_DIR, TARGET_COL

sns.set_theme(style="whitegrid", context="notebook")

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_merged.csv")
    print("读取:", in_path)
    df = pd.read_csv(in_path)

    # 保存统计 (逻辑不变)
    stats_path = os.path.join(RESULTS_DIR, "basic_eda_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        if TARGET_COL in df.columns:
            f.write(f"Target Distribution:\n{df[TARGET_COL].value_counts()}\n\n")
        for col in ["TransactionAmt", "TransactionDT", "card1", "addr1"]:
            if col in df.columns:
                f.write(f"=== {col} describe ===\n{df[col].describe()}\n\n")

    # 绘图优化
    plot_cols = ["TransactionAmt", "TransactionDT"]
    for col in plot_cols:
        if col not in df.columns: continue
        
        plt.figure(figsize=(8, 5))
        
        # 特殊处理金额：通常金额是长尾的，画图时为了好看可以截断或者用log
        # 这里用 log_scale=True (如果是 Amount)，或者简单加上 kde
        use_log = True if col == "TransactionAmt" else False
        
        sns.histplot(df[col], bins=50, kde=True, log_scale=use_log, 
                     color="#3498db", edgecolor="white", alpha=0.7)
        
        title_str = f"Distribution of {col}" + (" (Log Scale)" if use_log else "")
        plt.title(title_str, fontsize=14, fontweight='bold')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
        sns.despine()
        plt.tight_layout()
        out_png = os.path.join(RESULTS_DIR, f"eda_hist_{col}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()