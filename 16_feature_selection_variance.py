# 16_feature_selection_variance.py
# 功能：对 X_all 做方差过滤，删除方差过小的特征，得到 X_varfs 和 feature_names_varfs

import os
import numpy as np
import joblib
from sklearn.feature_selection import VarianceThreshold
from config import FEATURE_DIR, VARIANCE_THRESHOLD

def main():
    X_all_path = os.path.join(FEATURE_DIR, "X_all.npy")
    feat_all_path = os.path.join(FEATURE_DIR, "feature_names_all.pkl")
    X_varfs_path = os.path.join(FEATURE_DIR, "X_varfs.npy")
    feat_varfs_path = os.path.join(FEATURE_DIR, "feature_names_varfs.pkl")

    X_all = np.load(X_all_path)
    feat_all = joblib.load(feat_all_path)
    print("X_all 形状:", X_all.shape, "特征数:", len(feat_all))

    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_varfs = selector.fit_transform(X_all)
    support = selector.get_support()

    feat_varfs = [f for f, keep in zip(feat_all, support) if keep]

    np.save(X_varfs_path, X_varfs)
    joblib.dump(feat_varfs, feat_varfs_path)

    print("方差过滤后 X_varfs 形状:", X_varfs.shape, "保留特征数:", len(feat_varfs))
    print("X_varfs 保存到:", X_varfs_path)
    print("feature_names_varfs 保存到:", feat_varfs_path)

if __name__ == "__main__":
    main()
