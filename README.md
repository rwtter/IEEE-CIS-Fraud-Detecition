1.项目名称

高级信用卡欺诈检测系统
（IEEE-CIS Fraud Detection：离线建模流水线 + 手动输入风控 Demo）

2.项目概述

本项目基于 IEEE-CIS Fraud Detection 数据集（可于kaggle官方网站下载该数据集，本实验采用阿里云镜像网站，下载链接为https://tianchi.aliyun.com/dataset/87814），
构建了一个完整的信用卡欺诈检测系统，包含两条核心链路：

2.1 离线建模流水线

从原始 CSV 数据出发，通过数据清洗、特征工程、模型训练、随机搜索调参、成本敏感学习等步骤，自动生成可用于线上部署的欺诈检测模型。

2.2 手动输入风控 Demo（app.py）

面向业务/风控场景，单独设计了一套轻量级模型与预处理逻辑，用于在命令行中手动输入一笔交易信息，并实时输出欺诈概率、欺诈标签以及风险等级。

项目目标：
（1）工程维度：搭建一条可复现、可扩展的完整建模流水线；

（2）业务维度：贴近真实风控需求，引入成本敏感学习和“可交互风控原型”；

（3）课程维度：远超一般分类实验的复杂度，且报告具备高分潜力。

3.数据与业务背景

3.1 数据来源

IEEE-CIS Fraud Detection（真实线上信用卡交易数据）

3.2 数据特点

（1）极度不平衡：绝大多数是正常交易，欺诈只占极小比例
（2）特征众多且复杂，包括金额、时间、设备信息、邮箱域、地址、卡号等
（3）真实风控强调 Recall 与业务成本，而非简单准确率

3.3 项目任务

给定一条交易记录，预测是否欺诈（isFraud = 0/1），并给出欺诈概率。

4.项目目录结构

Fraud Detection/
app.py 手动输入风控 Demo（最终业务系统）
config.py 全局路径与常量配置
run_all.py 一键运行脚本（01–35 全流程）
README.txt / README.md 本文件（说明文档）

data_raw/                        原始数据（train_transaction 与 train_identity）
    train_transaction.csv
    train_identity.csv

data_processed/                  清洗后的数据
data_intermediate/               特征工程的中间结果
features/                        模型训练所用特征矩阵 .npy
models/                          存放所有模型与预处理器
    LightGBM_tuned.pkl
    LightGBM_cost_sensitive.pkl
    manual_app_preprocessor.pkl
    manual_app_lgbm.pkl

scripts/                         01–35 全部流水线脚本
    01_data_clean_transaction.py
    02_data_clean_identity.py
    ...
    35_model_save_cost_sensitive.py


说明：

（1）app.py 是一套独立的小系统，与 01–35 脚本互不影响。
（2）流水线脚本自动生成模型、特征矩阵和可分析文件。

5.离线建模流水线（01–35 步概览）

5.1 数据预处理阶段（01–07）
任务：

（1）读取与清洗 train_transaction.csv
（2）清洗 train_identity.csv
（3）按 TransactionID 合并两表
（4）处理缺失值、脏数据、异常值
（5）生成基础 EDA 报告与处理后的表

典型处理：

（1）数值填补：均值/中位数
（2）类别填补："MISSING"
（3）删除无意义列（如全空、全常数列）

5.2 特征工程阶段（08–25）
任务：构造欺诈识别最关键的高阶特征，包括以下类别：

（1）金额类特征：
TransactionAmt_log
金额标准化
用户平均金额
当前金额 / 用户均值比值
（2）时间行为特征：
DT_hour, DT_day, DT_week
高频交易特征（短间隔重复交易）
（3）设备信息特征：
DeviceType（mobile、desktop）
DeviceInfo_brand（apple, huawei, windows 等）
（4）邮箱域名特征：
P_emaildomain_provider（gmail、qq、yahoo 等）
（5）类别编码与统计特征：
LabelEncoding（card、addr、device）
分组统计特征（如 card1 欺诈率、card1 平均金额）
（6）降维与筛选（可选）

5.3 模型训练与基线（26–31）
任务：

（1）划分训练集 / 验证集
（2）训练基线模型（RF、GBDT、LightGBM）
（3）衡量模型关键指标：
AUC
Precision
Recall
F1
COST（业务成本）
（4）找到初步最佳模型范围

5.4 随机搜索与调参（32）
任务：

（1）进行 LightGBM 参数空间的随机搜索
（2）每次计算验证集指标
（3）记录最佳参数组
（4）输出最优模型 LightGBM_tuned.pkl

5.5 成本敏感学习（33–35）
特点：

（1）增强欺诈样本权重
（2）减小漏判风险（FN 代价远大于 FP）
（3）保存最终业务友好模型 LightGBM_cost_sensitive.pkl

6.手动输入风控系统 app.py

该模块提供一个独立的小型“实时风控评估系统”，模拟银行风控人工录入交易的流程。

6.1 功能说明
（1）支持手动输入以下字段：
TransactionAmt
card1
addr1
DeviceType
DeviceInfo
P_emaildomain

（2）自动提取以下高级特征：
TransactionAmt_log
card1_avg_amt
amt_to_card1_ratio
DeviceInfo_brand
Email_provider

（3）LabelEncoder + 数值规范化
（4）实时返回：
欺诈概率
是否欺诈
风险等级（极低/较低/中等/高）

6.2 运行方式
一、训练小模型
python app.py --mode train

二、启动交互式应用
python app.py
或
python app.py --mode interactive

模型与预处理器的独立文件：
manual_app_preprocessor.pkl
manual_app_lgbm.pkl

7.环境准备

7.1 创建虚拟环境
python -m venv .venv

7.2 激活环境
..venv\Scripts\activate

7.3 安装依赖
pip install -r requirements.txt

一键运行整个离线流水线

在项目根目录执行：

python run_all.py


该指令将自动顺序运行 scripts/01 到 scripts/35。

