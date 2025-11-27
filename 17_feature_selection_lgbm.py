# 17_feature_selection_lgbm.py
# 功能：基于 LightGBM 的特征重要性再做一轮特征选择，得到 X_fs 和 feature_names_fs，
#       同时输出 feature_importance_lgbm.csv 和美化后的图像

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns  # 新增：用于美化绘图
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import FEATURE_DIR, RESULTS_DIR, RANDOM_STATE

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="notebook")

TOP_N_FEATURES = 300  # 最多保留前 300 个重要特征，可按需要调整

def main():
    # 1. 定义路径
    X_varfs_path = os.path.join(FEATURE_DIR, "X_varfs.npy")
    feat_varfs_path = os.path.join(FEATURE_DIR, "feature_names_varfs.pkl")
    y_all_path = os.path.join(FEATURE_DIR, "y_all.npy")

    X_fs_path = os.path.join(FEATURE_DIR, "X_fs.npy")
    feat_fs_path = os.path.join(FEATURE_DIR, "feature_names_fs.pkl")
    fi_csv_path = os.path.join(RESULTS_DIR, "feature_importance_lgbm.csv")
    fi_png_path = os.path.join(RESULTS_DIR, "feature_importance_lgbm.png")

    # 2. 加载数据
    print("正在加载数据...")
    X = np.load(X_varfs_path)
    feat_names = joblib.load(feat_varfs_path)
    y = np.load(y_all_path)

    print("X_varfs 形状:", X.shape, "特征数:", len(feat_names), "y 长度:", len(y))

    # 3. 划分验证集 (仅用于观察 AUC，特征选择本身依赖 importances)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4. 训练模型
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("训练用于特征选择的 LightGBM 模型 ...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    print(f"用于特征选择的模型验证 AUC: {auc:.4f}")

    # 5. 提取特征重要性并保存
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    fi_df.to_csv(fi_csv_path, index=False)
    print("特征重要性已保存到:", fi_csv_path)

    # 6. 特征筛选逻辑
    fi_nonzero = fi_df[fi_df["importance"] > 0]
    if len(fi_nonzero) == 0:
        print("警告：所有特征重要性为 0，将保留全部特征。")
        selected_features = fi_df["feature"].tolist()
    else:
        selected_features = fi_nonzero.head(TOP_N_FEATURES)["feature"].tolist()

    print("最终选择特征数:", len(selected_features))

    # 7. 重建并保存 X_fs
    feat_to_idx = {f: i for i, f in enumerate(feat_names)}
    selected_indices = [feat_to_idx[f] for f in selected_features]
    X_fs = X[:, selected_indices]

    np.save(X_fs_path, X_fs)
    joblib.dump(selected_features, feat_fs_path)

    print("X_fs 形状:", X_fs.shape)
    print("X_fs 保存到:", X_fs_path)
    print("feature_names_fs 保存到:", feat_fs_path)

    # 8. 绘制美化后的特征重要性图
    top_plot = fi_df.head(40)  # 只展示前 40 个，避免太长
    plt.figure(figsize=(10, 12))
    
    # 使用 seaborn 的 barplot，利用 palette="viridis" 实现颜色渐变
    # y 轴为 feature，x 轴为 importance
    sns.barplot(
        x="importance", 
        y="feature", 
        data=top_plot, 
        palette="viridis", 
        edgecolor=None
    )
    
    plt.title(f"Top 40 Feature Importances (LightGBM, AUC={auc:.3f})", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tick_params(axis='y', labelsize=10)
    
    # 去除多余边框 (despine)，只保留刻度
    sns.despine(left=True, bottom=True)
    plt.grid(axis='x', linestyle='--', alpha=0.5)  # 仅保留横向网格辅助读数
    
    plt.tight_layout()
    plt.savefig(fi_png_path, dpi=300)
    plt.close()
    print("特征重要性美化图已保存到:", fi_png_path)

if __name__ == "__main__":
    main()