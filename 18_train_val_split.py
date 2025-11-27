# 18_train_val_split.py
# 功能：对 X_fs / y_all 做一次 stratified 划分，得到 X_train/X_val/y_train/y_val

import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import FEATURE_DIR, RANDOM_STATE, VALID_SIZE

def main():
    X_fs_path = os.path.join(FEATURE_DIR, "X_fs.npy")
    y_all_path = os.path.join(FEATURE_DIR, "y_all.npy")

    X_train_path = os.path.join(FEATURE_DIR, "X_train.npy")
    y_train_path = os.path.join(FEATURE_DIR, "y_train.npy")
    X_val_path = os.path.join(FEATURE_DIR, "X_val.npy")
    y_val_path = os.path.join(FEATURE_DIR, "y_val.npy")

    X = np.load(X_fs_path)
    y = np.load(y_all_path)
    print("X_fs 形状:", X.shape, "y_all 长度:", len(y))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    np.save(X_train_path, X_train)
    np.save(y_train_path, y_train)
    np.save(X_val_path, X_val)
    np.save(y_val_path, y_val)

    print("train/val 划分完成：")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("已保存到 features/ 目录。")

if __name__ == "__main__":
    main()
