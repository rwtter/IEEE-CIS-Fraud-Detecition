# 10_device_features.py
import os
import pandas as pd
from config import DATA_STAGE_DIR, LOGS_DIR

def extract_device_brand(device_info: str) -> str:
    """
    简单从 DeviceInfo 中提取品牌前缀：
    比如 'Windows', 'iOS', 'Android', 'SM-G960F' -> 取第一段或前两段。
    """
    if not isinstance(device_info, str) or device_info.strip() == "":
        return "unknown"
    s = device_info.lower().strip()
    # 常见关键字归类
    if "iphone" in s or "ios" in s:
        return "apple"
    if "ipad" in s:
        return "apple_ipad"
    if "mac" in s:
        return "apple_mac"
    if "android" in s:
        return "android"
    if "windows" in s:
        return "windows"
    if "linux" in s:
        return "linux"
    # 其它情况取第一段
    return s.split()[0].split("/")[0][:20]

def normalize_device_type(x: str) -> str:
    if not isinstance(x, str) or x.strip() == "":
        return "unknown"
    s = x.lower()
    if "mobile" in s:
        return "mobile"
    if "desktop" in s:
        return "desktop"
    return s

def main():
    in_path = os.path.join(DATA_STAGE_DIR, "train_with_time.csv")
    out_path = os.path.join(DATA_STAGE_DIR, "train_with_device.csv")
    log_path = os.path.join(LOGS_DIR, "device_value_counts.txt")

    print("读取:", in_path)
    df = pd.read_csv(in_path)
    print("输入形状:", df.shape)

    if "DeviceType" in df.columns:
        df["DeviceType_norm"] = df["DeviceType"].astype(str).apply(normalize_device_type)
    else:
        df["DeviceType_norm"] = "unknown"

    if "DeviceInfo" in df.columns:
        df["DeviceInfo_brand"] = df["DeviceInfo"].astype(str).apply(extract_device_brand)
    else:
        df["DeviceInfo_brand"] = "unknown"

    with open(log_path, "w", encoding="utf-8") as f:
        if "DeviceType_norm" in df.columns:
            f.write("DeviceType_norm value_counts:\n")
            f.write(str(df["DeviceType_norm"].value_counts()) + "\n\n")
        if "DeviceInfo_brand" in df.columns:
            f.write("DeviceInfo_brand top20:\n")
            f.write(str(df["DeviceInfo_brand"].value_counts().head(20)) + "\n")

    print("添加设备特征后形状:", df.shape)
    df.to_csv(out_path, index=False)
    print("已保存到:", out_path)
    print("设备特征统计已写入:", log_path)

if __name__ == "__main__":
    main()
