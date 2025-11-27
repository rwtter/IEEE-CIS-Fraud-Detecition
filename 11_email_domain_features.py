# 11_email_domain_features.py
import os
import pandas as pd
from config import DATA_STAGE_DIR, LOGS_DIR

# 一些常见邮箱域名分组，可以按需扩展
EMAIL_PROVIDER_MAP = {
    # gmail
    "gmail.com": "gmail",
    "googlemail.com": "gmail",

    # yahoo
    "yahoo.com": "yahoo",
    "yahoo.co.uk": "yahoo",
    "yahoo.co.jp": "yahoo",

    # hotmail / outlook / live
    "hotmail.com": "microsoft",
    "outlook.com": "microsoft",
    "live.com": "microsoft",

    # qq / 163 / 126 / sina 等
    "qq.com": "china_qq",
    "163.com": "china_163",
    "126.com": "china_126",
    "sina.com": "china_sina",
    "sina.com.cn": "china_sina",

    # 其它可归为 "other"
}

def normalize_email_domain(x: str) -> str:
    if not isinstance(x, str) or x.strip() == "" or x.lower() == "nan":
        return "unknown"
    s = x.strip().lower()
    return s

def map_email_provider(domain: str) -> str:
    if domain in EMAIL_PROVIDER_MAP:
        return EMAIL_PROVIDER_MAP[domain]
    if domain == "unknown":
        return "unknown"
    # 简单归类：包含 qq / 163 / 126 / sina / edu / gov 等
    if "qq.com" in domain:
        return "china_qq"
    if "163.com" in domain:
        return "china_163"
    if "126.com" in domain:
        return "china_126"
    if "sina" in domain:
        return "china_sina"
    if domain.endswith(".edu") or ".edu." in domain:
        return "edu"
    if domain.endswith(".gov") or ".gov." in domain:
        return "gov"
    return "other"

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_with_device.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_with_email.csv")
    log_path = os.path.join(LOGS_DIR, "email_domain_stats.txt")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    # 预处理 P_emaildomain / R_emaildomain
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            df[col + "_norm"] = df[col].astype(str).apply(normalize_email_domain)
            df[col + "_provider"] = df[col + "_norm"].apply(map_email_provider)
        else:
            df[col + "_norm"] = "unknown"
            df[col + "_provider"] = "unknown"

    # 是否同域、是否同提供商
    df["email_same_domain"] = (
        df["P_emaildomain_norm"] == df["R_emaildomain_norm"]
    ).astype("int8")

    df["email_same_provider"] = (
        df["P_emaildomain_provider"] == df["R_emaildomain_provider"]
    ).astype("int8")

    # 是否常见邮箱（gmail / qq / 163 / microsoft 等）
    common_providers = {
        "gmail", "china_qq", "china_163", "microsoft", "yahoo"
    }
    df["P_email_is_common"] = df["P_emaildomain_provider"].isin(common_providers).astype("int8")
    df["R_email_is_common"] = df["R_emaildomain_provider"].isin(common_providers).astype("int8")

    # 写一些统计信息
    with open(log_path, "w", encoding="utf-8") as f:
        for col in ["P_emaildomain_provider", "R_emaildomain_provider"]:
            if col in df.columns:
                f.write(f"{col} value_counts (top 20):\n")
                f.write(str(df[col].value_counts().head(20)) + "\n\n")
        f.write("email_same_domain 分布:\n")
        f.write(str(df["email_same_domain"].value_counts()) + "\n\n")
        f.write("email_same_provider 分布:\n")
        f.write(str(df["email_same_provider"].value_counts()) + "\n")

    print("添加邮箱特征后形状:", df.shape)
    df.to_csv(out_path, index=False)
    print("已保存到:", out_path)
    print("邮箱特征统计已写入:", log_path)

if __name__ == "__main__":
    main()
