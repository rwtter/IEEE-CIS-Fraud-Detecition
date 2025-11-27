# 34_shap_analysis.py
# 功能：
#   使用 SHAP 对 LightGBM_tuned 模型进行解释：
#     - 全局 summary dot 图 (Beeswarm)
#     - 全局 summary bar 图 (Importance)
#     - 一个典型样本的局部特征贡献条形图 (红蓝对抗图)

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from config import FEATURE_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="notebook")

MAX_BACKGROUND = 5000     # 作为背景的训练样本数量上限
MAX_EXPLAIN_VAL = 2000    # 用于 summary 的验证样本数量上限
TOP_LOCAL_FEATURES = 20   # 局部解释显示的特征数量

def main():
    # 1. 加载数据与模型
    X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy"))
    # y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy")) # 暂时不用
    X_val = np.load(os.path.join(FEATURE_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(FEATURE_DIR, "y_val.npy"))

    feat_name_path = os.path.join(FEATURE_DIR, "feature_names_fs.pkl")
    if os.path.exists(feat_name_path):
        feature_names = joblib.load(feat_name_path)
    else:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    model_path = os.path.join(MODELS_DIR, "LightGBM_tuned.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("未找到 LightGBM_tuned.pkl，请先运行 21_optimize_lightgbm.py。")

    model = joblib.load(model_path)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 2. 准备背景数据（来自训练集，加速 SHAP 计算）
    rng = np.random.default_rng(RANDOM_STATE + 999)
    n_train = X_train.shape[0]
    bg_n = min(MAX_BACKGROUND, n_train)
    idx_bg = rng.choice(n_train, size=bg_n, replace=False)
    X_bg = X_train[idx_bg]

    # 3. 准备要解释的验证集子集
    n_val = X_val.shape[0]
    exp_n = min(MAX_EXPLAIN_VAL, n_val)
    idx_exp = rng.choice(n_val, size=exp_n, replace=False)
    X_exp = X_val[idx_exp]
    y_exp = y_val[idx_exp]

    print(f"SHAP分析配置: 背景样本={bg_n}, 解释样本={exp_n}")

    # 4. 构建 SHAP explainer
    # TreeExplainer 对 LightGBM 效率很高
    explainer = shap.TreeExplainer(model, X_bg)
    
    # 检查 additivity (有时因精度问题会报错，设为 False 可忽略微小误差)
    try:
        shap_values = explainer.shap_values(X_exp, check_additivity=False)
    except Exception as e:
        print(f"SHAP计算警告: {e}")
        shap_values = explainer.shap_values(X_exp)

    # 对二分类 LightGBM，shap_values 可能是 list [matrix_class0, matrix_class1]
    if isinstance(shap_values, list):
        shap_values_used = shap_values[1] # 取正类（Fraud）
    else:
        shap_values_used = shap_values

    # === 绘图 1: Summary Dot Plot (Beeswarm) ===
    plt.figure(figsize=(10, 10))
    # 使用 coolwarm 颜色：红=特征值高，蓝=特征值低
    shap.summary_plot(
        shap_values_used,
        X_exp,
        feature_names=feature_names,
        show=False,
        max_display=40,
        cmap="coolwarm" 
    )
    plt.title("SHAP Global Feature Impact (Beeswarm)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    out_dot = os.path.join(RESULTS_DIR, "shap_summary_dot.png")
    plt.savefig(out_dot, dpi=300)
    plt.close()
    print("SHAP Summary Dot 图已保存:", out_dot)

    # === 绘图 2: Summary Bar Plot (Importance) ===
    plt.figure(figsize=(10, 10))
    shap.summary_plot(
        shap_values_used,
        X_exp,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=40,
        color="#3498db" # 统一使用清爽的蓝色
    )
    plt.title("SHAP Global Feature Importance (Mean |SHAP|)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    out_bar = os.path.join(RESULTS_DIR, "shap_summary_bar.png")
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print("SHAP Summary Bar 图已保存:", out_bar)

    # === 绘图 3: 局部解释 (Local Explanation) ===
    # 选取一个典型 fraud 样本
    fraud_indices = np.where(y_exp == 1)[0]
    local_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
    
    x_local = X_exp[local_idx]
    shap_local = shap_values_used[local_idx]

    # 取绝对 SHAP 值最大的特征
    abs_shap = np.abs(shap_local)
    top_idx = np.argsort(abs_shap)[-TOP_LOCAL_FEATURES:][::-1]

    top_features = [feature_names[i] for i in top_idx]
    top_shap = shap_local[top_idx]
    top_values = x_local[top_idx]

    # 构造绘图数据
    df_local = pd.DataFrame({
        "feature": top_features,
        "value": top_values,
        "shap_value": top_shap,
    })
    
    # 根据正负值分配颜色
    df_local["color"] = df_local["shap_value"].apply(lambda x: "#e74c3c" if x > 0 else "#3498db")
    # 构造标签：显示 特征名 + (数值)
    df_local["label"] = df_local.apply(lambda r: f"{r['feature']} ({r['value']:.3f})", axis=1)

    # 排序：让正贡献在上一组，负贡献在下一组，或者单纯按绝对值排序
    # 这里我们保持按绝对值影响力排序，这样最重要的特征在最上面
    df_local = df_local.iloc[::-1] # 翻转以便在 barh 中最重要的在上面

    plt.figure(figsize=(8, 10))
    
    # 画水平条形图
    bars = plt.barh(
        y=np.arange(len(df_local)), 
        width=df_local["shap_value"], 
        color=df_local["color"],
        edgecolor=None,
        height=0.6
    )
    
    plt.yticks(np.arange(len(df_local)), df_local["label"], fontsize=10)
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    
    plt.title(f"Local Explanation for Sample #{local_idx}\n(Red adds to Fraud Risk, Blue reduces it)", fontsize=14, fontweight='bold')
    plt.xlabel("SHAP Value", fontsize=12)
    
    # 添加网格
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    out_local = os.path.join(RESULTS_DIR, "shap_local_bar.png")
    plt.savefig(out_local, dpi=300)
    plt.close()
    print("局部样本 SHAP 条形图已保存:", out_local)

if __name__ == "__main__":
    main()