

import os
import sys
import numpy as np
import pandas as pd
import joblib
import argparse

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ========== 路径 & 全局配置 ==========

try:
    # 如果已有 config.py，就复用其中的一些配置
    from config import (
        DATA_RAW_DIR,
        MODELS_DIR,
        TARGET_COL,
        RANDOM_STATE,
    )
except ImportError:
    # 如果没有，就自己定义一套
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TARGET_COL = "isFraud"
    RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)

MANUAL_PREPROC_PATH = os.path.join(MODELS_DIR, "manual_app_preprocessor.pkl")
MANUAL_MODEL_PATH = os.path.join(MODELS_DIR, "manual_app_lgbm.pkl")


# ========== 预处理器定义 ==========

class ManualPreprocessor:
    """
    手动输入版预处理器：

    使用的原始字段（来自 train_transaction + train_identity）：
        TransactionAmt  (float)
        card1          (int)
        addr1          (int)
        DeviceType     (str)
        DeviceInfo     (str)
        P_emaildomain  (str)

    派生特征：
        TransactionAmt_log
        card1_avg_amt
        amt_to_card1_ratio
        DeviceType_norm      -> 再编码为 DeviceType_norm_le
        DeviceInfo_brand     -> 再编码为 DeviceInfo_brand_le
        P_emaildomain_provider -> 再编码为 P_emaildomain_provider_le

    最终特征列表：
        [
            TransactionAmt,
            TransactionAmt_log,
            card1,
            addr1,
            card1_avg_amt,
            amt_to_card1_ratio,
            DeviceType_norm_le,
            DeviceInfo_brand_le,
            P_emaildomain_provider_le,
        ]
    """

    def __init__(self):
        self.card1_mean_map = None
        self.global_amt_mean = None
        self.cat_maps = {}        # {col: {"mapping": {value: int}, "default": int}}
        self.feature_names_ = None
        self.fitted_ = False

    # ----------- 一些基础清洗 & 特征构造函数 -----------

    @staticmethod
    def _normalize_device_type(x):
        if x is None:
            s = ""
        else:
            s = str(x).strip().lower()
        if s in ("", "nan"):
            return "other"
        if "mobile" in s or "phone" in s:
            return "mobile"
        if "desktop" in s or "pc" in s:
            return "desktop"
        return "other"

    @staticmethod
    def _extract_device_brand(x):
        if x is None:
            s = ""
        else:
            s = str(x).strip().lower()
        if s in ("", "nan"):
            return "other"

        # 一些常见关键词 & 品牌
        if "iphone" in s or "ipad" in s or "mac" in s or "ios" in s or "apple" in s:
            return "apple"
        if "android" in s:
            return "android"
        for kw in ["huawei", "honor"]:
            if kw in s:
                return "huawei"
        for kw in ["xiaomi", "redmi", "mi "]:
            if kw in s:
                return "xiaomi"
        for kw in ["oppo"]:
            if kw in s:
                return "oppo"
        for kw in ["vivo"]:
            if kw in s:
                return "vivo"
        for kw in ["samsung"]:
            if kw in s:
                return "samsung"
        if "windows" in s:
            return "windows"
        if "linux" in s:
            return "linux"

        return "other"

    @staticmethod
    def _extract_email_provider(x):
        if x is None:
            s = ""
        else:
            s = str(x).strip().lower()
        if s in ("", "nan"):
            return "unknown"

        # 如果用户输入完整邮箱地址，先截取 @ 后面的部分
        if "@" in s:
            s = s.split("@", 1)[1]
        # 然后截取第一个 .
        if "." in s:
            s = s.split(".", 1)[0]

        if s == "":
            return "unknown"
        return s

    def _prepare_base_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 保证需要的列存在
        for col in ["TransactionAmt", "card1", "addr1", "DeviceType", "DeviceInfo", "P_emaildomain"]:
            if col not in df.columns:
                df[col] = np.nan

        # 数值转型
        for col in ["TransactionAmt", "card1", "addr1"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 字符串保持原样，由后续函数再处理
        return df

    # ----------- 拟合（用于训练阶段） -----------

    def fit(self, df_raw: pd.DataFrame):
        df = self._prepare_base_df(df_raw)

        # 1) 金额相关统计特征
        self.global_amt_mean = df["TransactionAmt"].mean()
        # 防止全 NaN 的极端情况
        if pd.isna(self.global_amt_mean):
            self.global_amt_mean = 0.0

        self.card1_mean_map = (
            df.groupby("card1")["TransactionAmt"]
            .mean()
            .dropna()
            .to_dict()
        )

        # 使用上述映射构造统计特征
        df["card1_avg_amt"] = df["card1"].map(self.card1_mean_map)
        df["card1_avg_amt"] = df["card1_avg_amt"].fillna(self.global_amt_mean)
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].clip(lower=0))
        # 避免除 0
        denom = df["card1_avg_amt"].replace(0, self.global_amt_mean)
        df["amt_to_card1_ratio"] = df["TransactionAmt"] / denom

        # 2) 文本归一化 & 提取业务含义字段
        df["DeviceType_norm"] = df["DeviceType"].apply(self._normalize_device_type)
        df["DeviceInfo_brand"] = df["DeviceInfo"].apply(self._extract_device_brand)
        df["P_emaildomain_provider"] = df["P_emaildomain"].apply(self._extract_email_provider)

        # 3) 类别编码
        self.cat_maps = {}
        cat_cols = ["DeviceType_norm", "DeviceInfo_brand", "P_emaildomain_provider"]

        for col in cat_cols:
            vals = df[col].fillna("__MISSING__").astype(str)
            uniques = pd.unique(vals)
            mapping = {v: i for i, v in enumerate(uniques)}
            default_key = "__MISSING__" if "__MISSING__" in mapping else uniques[0]
            self.cat_maps[col] = {"mapping": mapping, "default": mapping[default_key]}

            df[col + "_le"] = vals.map(mapping).fillna(mapping[default_key]).astype(int)

        # 4) 最终特征列表
        self.feature_names_ = [
            "TransactionAmt",
            "TransactionAmt_log",
            "card1",
            "addr1",
            "card1_avg_amt",
            "amt_to_card1_ratio",
            "DeviceType_norm_le",
            "DeviceInfo_brand_le",
            "P_emaildomain_provider_le",
        ]

        # 填充缺失
        df_feat = df[self.feature_names_].fillna(0.0)
        self.fitted_ = True

        return df_feat.values.astype(np.float32)

    # ----------- transform（用于训练/推理） -----------

    def transform_df(self, df_raw: pd.DataFrame):
        if not self.fitted_:
            raise RuntimeError("ManualPreprocessor 尚未 fit，无法 transform。")

        df = self._prepare_base_df(df_raw)

        # 使用已存的统计量生成统计特征
        df["card1_avg_amt"] = df["card1"].map(self.card1_mean_map)
        df["card1_avg_amt"] = df["card1_avg_amt"].fillna(self.global_amt_mean)
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].clip(lower=0))
        denom = df["card1_avg_amt"].replace(0, self.global_amt_mean)
        df["amt_to_card1_ratio"] = df["TransactionAmt"] / denom

        # 文本归一化
        df["DeviceType_norm"] = df["DeviceType"].apply(self._normalize_device_type)
        df["DeviceInfo_brand"] = df["DeviceInfo"].apply(self._extract_device_brand)
        df["P_emaildomain_provider"] = df["P_emaildomain"].apply(self._extract_email_provider)

        # 类别编码，使用已有映射
        for col in ["DeviceType_norm", "DeviceInfo_brand", "P_emaildomain_provider"]:
            info = self.cat_maps[col]
            mapping = info["mapping"]
            default = info["default"]

            vals = df[col].fillna("__MISSING__").astype(str)

            def map_func(v):
                return mapping.get(v, default)

            df[col + "_le"] = vals.map(map_func).astype(int)

        df_feat = df[self.feature_names_].fillna(0.0)
        return df_feat.values.astype(np.float32)

    def transform_single(self, record: dict):
        """
        record 形如：
            {
                "TransactionAmt": 100.0,
                "card1": 10000,
                "addr1": 325,
                "DeviceType": "desktop",
                "DeviceInfo": "Windows 10",
                "P_emaildomain": "gmail.com",
            }
        """
        df = pd.DataFrame([record])
        return self.transform_df(df)


# ========== 模型训练部分（只需偶尔跑一次） ==========

def load_raw_train_data():
    """
    从 data_raw 加载 train_transaction 和 train_identity，
    并在 TransactionID 上进行左连接。
    """
    trans_path = os.path.join(DATA_RAW_DIR, "train_transaction.csv")
    id_path = os.path.join(DATA_RAW_DIR, "train_identity.csv")

    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"未找到 {trans_path}，请确认 data_raw 目录和文件名。")
    if not os.path.exists(id_path):
        raise FileNotFoundError(f"未找到 {id_path}，请确认 data_raw 目录和文件名。")

    print(f"读取 {trans_path} ...")
    df_trans = pd.read_csv(trans_path)
    print(f"读取 {id_path} ...")
    df_id = pd.read_csv(id_path)

    print("合并 transaction 和 identity ...")
    df = df_trans.merge(df_id, on="TransactionID", how="left")

    return df


def train_manual_model():
    print("========== [步骤 1] 加载原始训练数据 ==========")
    df = load_raw_train_data()
    print(f"原始训练数据形状: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"数据中未找到目标列 {TARGET_COL}，请检查数据集。")

    y = df[TARGET_COL].astype(int).values
    # 我们只用一小部分与业务相关、可人工输入的字段
    cols_needed = ["TransactionAmt", "card1", "addr1", "DeviceType", "DeviceInfo", "P_emaildomain"]
    df_feat = df[cols_needed].copy()

    print("========== [步骤 2] 拟合预处理器并生成特征 ==========")
    preproc = ManualPreprocessor()
    X_all = preproc.fit(df_feat)

    print("特征维度:", X_all.shape)

    print("========== [步骤 3] 划分训练 / 验证集 ==========")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("训练集:", X_train.shape, "验证集:", X_val.shape)

    print("========== [步骤 4] 训练 LightGBM 模型 ==========")
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        verbose=50,
    )

    print("========== [步骤 5] 在验证集上评估 ==========")
    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    print(f"AUC: {auc:.4f}, ACC: {acc:.4f}, PREC: {prec:.4f}, REC: {rec:.4f}, F1: {f1:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("========== [步骤 6] 保存预处理器和模型 ==========")
    joblib.dump(preproc, MANUAL_PREPROC_PATH)
    joblib.dump(clf, MANUAL_MODEL_PATH)
    print(f"预处理器已保存到: {MANUAL_PREPROC_PATH}")
    print(f"模型已保存到: {MANUAL_MODEL_PATH}")


# ========== 手动输入 & 风险评分部分 ==========

def risk_level(prob):
    if prob < 0.01:
        return "极低风险"
    elif prob < 0.05:
        return "较低风险"
    elif prob < 0.20:
        return "中等风险"
    else:
        return "高风险"


def ask_float(prompt, allow_empty=True):
    while True:
        s = input(prompt).strip()
        if s == "":
            if allow_empty:
                return None
            else:
                print("该字段不能为空，请重新输入。")
                continue
        try:
            return float(s)
        except ValueError:
            print("请输入数字。")


def ask_int(prompt, allow_empty=True):
    while True:
        s = input(prompt).strip()
        if s == "":
            if allow_empty:
                return None
            else:
                print("该字段不能为空，请重新输入。")
                continue
        try:
            return int(s)
        except ValueError:
            print("请输入整数。")


def ask_str(prompt, allow_empty=True):
    while True:
        s = input(prompt).strip()
        if s == "":
            if allow_empty:
                return None
            else:
                print("该字段不能为空，请重新输入。")
                continue
        return s


def interactive_predict():
    # 如果模型不存在，先训练一遍
    if not (os.path.exists(MANUAL_PREPROC_PATH) and os.path.exists(MANUAL_MODEL_PATH)):
        print("未检测到手动输入版模型，先进行一次训练（只需执行一次）...")
        train_manual_model()

    print("\n========== 加载预处理器和模型 ==========")
    preproc = joblib.load(MANUAL_PREPROC_PATH)
    model = joblib.load(MANUAL_MODEL_PATH)
    print("加载完成，可以开始手动输入交易信息。")

    while True:
        print("\n==========================================")
        print(" 手动输入一条交易进行欺诈风险评分")
        print(" 说明：不想填的数值字段可以直接回车，表示缺失")
        print("==========================================")

        amt = ask_float("1) 交易金额 TransactionAmt (如 100.0)：", allow_empty=False)
        card1 = ask_int("2) 用户主卡号 card1 (如 10000，可空)：", allow_empty=True)
        addr1 = ask_int("3) 账单地址编号 addr1 (如 325，可空)：", allow_empty=True)

        print("4) 设备类型 DeviceType（建议输入 desktop / mobile / 其它随意）：")
        dev_type = ask_str("   DeviceType：", allow_empty=True)

        print("5) 设备信息 DeviceInfo（如 Windows 10 / iOS / Android，可空）：")
        dev_info = ask_str("   DeviceInfo：", allow_empty=True)

        print("6) 邮箱域名 P_emaildomain：")
        print("   例如：gmail.com / qq.com / 163.com")
        print("   如果你直接输入完整邮箱，如 xxx@gmail.com，我会自动取 @ 后面的部分")
        email_dom = ask_str("   P_emaildomain：", allow_empty=True)

        sample = {
            "TransactionAmt": amt,
            "card1": card1,
            "addr1": addr1,
            "DeviceType": dev_type,
            "DeviceInfo": dev_info,
            "P_emaildomain": email_dom,
        }

        X_one = preproc.transform_single(sample)
        prob = model.predict_proba(X_one)[0, 1]
        pred = int(prob >= 0.5)

        print("\n========== 预测结果 ==========")
        print(f"输入交易信息：")
        for k, v in sample.items():
            print(f"  {k}: {v}")
        print("----------------------------------")
        print(f"模型预测欺诈概率: {prob:.4f}")
        print(f"模型判断标签: {pred}  (1 = fraud, 0 = normal)")
        print(f"风险等级: {risk_level(prob)}")
        print("==================================")

        cont = input("\n是否继续输入下一条？(y 继续 / 其它键退出)：").strip().lower()
        if cont != "y":
            print("结束手动输入 Demo。")
            break


# ========== 主入口 ==========

def main():
    parser = argparse.ArgumentParser(
        description="手动输入版信用卡欺诈检测 Demo",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "interactive"],
        default="interactive",
        help="train: 仅训练模型; interactive: 进入手动输入交互模式(默认)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_manual_model()
    else:
        interactive_predict()


if __name__ == "__main__":
    main()
