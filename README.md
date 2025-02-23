# 医疗保险欺诈检测系统 / Healthcare Insurance Fraud Detection System

[English Version](#english-version)

## 中文版本

### 📋 项目概述

本项目构建了一个端到端的医疗保险欺诈检测系统，集成了数据分析、模型训练、特征解释和防控策略生成等功能。系统基于美国医疗保险和医疗补助服务中心（CMS）以及美国卫生与公众服务部监察长办公室（OIG）的公开数据构建。

### 🌟 核心特性

- **全流程自动化**：从数据预处理到模型部署的完整流程自动化
- **多模型集成**：集成了逻辑回归、随机森林、XGBoost和深度神经网络等多个模型
- **可解释性分析**：使用SHAP值提供详细的模型解释
- **防控策略生成**：自动生成可执行的风险防控规则

### 📊 数据说明

#### 数据来源与处理
本项目利用两个主要数据源构建医疗保险欺诈检测数据集：

1. **CMS Medicare Part B 数据**
- 来源：美国医疗保险和医疗补助服务中心（CMS）
- 内容：医生执行的具体诊疗项目及费用信息
- 特点：
  * 基于医生NPI（National Provider Identifier）唯一标识
  * 使用HCPCS（Healthcare Common Procedure Coding System）编码标识医疗程序
  * 包含平均支付金额、收费次数、医疗专业等信息
  * 按提供者NPI、HCPCS代码和服务地点进行聚合

2. **LEIE（List of Excluded Individuals/Entities）数据**
- 来源：美国卫生与公众服务部监察长办公室（OIG）
- 更新频率：每月更新
- 用途：识别被排除在联邦医疗保健计划之外的个人或实体
- 特点：包含医疗保险欺诈相关刑事定罪记录

#### 数据整合过程
1. 通过医疗实体唯一识别码（NPI）将CMS数据与LEIE黑名单关联
2. 提取相关特征并进行标准化处理
3. 生成最终包含95万条记录的特征库

#### 特征维度
- **样本规模**：95万条医疗记录
- **特征数量**：11个特征（7个数值特征，4个类别特征）

#### 数据结构
每条记录包含：
- 医生唯一标识（NPI）
- 提供者类型（专业领域）
- HCPCS服务代码
- 服务地点（医院/诊所）
- 索赔计数和相关属性

### 🏗️ 项目结构

```
0222modelcode/
├── config/                 # 配置文件
├── models/                # 模型实现
├── utils/                 # 工具函数
├── evaluation/           # 评估结果
├── output/               # 输出文件
└── logs/                 # 日志文件
```

### 📈 性能指标

| 模型 | PR-AUC | F2-Score | Lift@10% |
|------|--------|----------|-----------|
| XGBoost | 0.892 | 85.34% | 3.45x |
| RandomForest | 0.878 | 83.21% | 3.21x |
| DNN | 0.865 | 82.15% | 3.12x |
| LogisticRegression | 0.843 | 80.56% | 2.98x |

### 🚀 快速开始

1. **环境配置**
```bash
pip install -r requirements.txt
```

2. **运行实验**
```bash
python run_experiment.py
```

### 📦 输出文件

- 数据预处理报告（含特征分布可视化）
- 模型性能对比表（LaTeX格式）
- SHAP解释图集（300dpi TIFF格式）
- 防控策略手册（含可执行规则）
- 实验日志（含MD5校验）

### 🔍 特征说明

**数值特征**:
- Tot_Benes: 服务的受益人总数
- Tot_Srvcs: 提供的服务总次数
- Tot_Bene_Day_Srvcs: 受益人接受服务的总天数
- Avg_Sbmtd_Chrg: 平均提交费用
- Avg_Mdcr_Alowd_Amt: 医保允许的平均金额
- Avg_Mdcr_Pymt_Amt: 医保实际支付的平均金额
- Bi_Wk_Avg_SC: 双周服务费用

**类别特征**:
- Rndrng_Prvdr_Type: 提供者类型
- Rndrng_Prvdr_Gndr: 提供者性别
- HCPCS_Cd: 医疗服务代码
- Place_Of_Srvc: 服务地点

---

## English Version

### 📋 Project Overview

This project implements an end-to-end healthcare insurance fraud detection system, integrating data analysis, model training, feature interpretation, and control strategy generation. The system is built on public data from the Centers for Medicare & Medicaid Services (CMS) and the Office of Inspector General (OIG).

### 🌟 Core Features

- **Full Process Automation**: Complete automation from data preprocessing to model deployment
- **Multi-Model Integration**: Integration of Logistic Regression, Random Forest, XGBoost, and Deep Neural Networks
- **Interpretability Analysis**: Detailed model interpretation using SHAP values
- **Control Strategy Generation**: Automatic generation of executable risk control rules

### �� Data Description

#### Data Sources and Processing
This project utilizes two primary data sources to build the healthcare insurance fraud detection dataset:

1. **CMS Medicare Part B Data**
- Source: Centers for Medicare & Medicaid Services (CMS)
- Content: Doctor-specific medical procedure and cost information
- Features:
  * Based on doctor NPI (National Provider Identifier)
  * Uses HCPCS (Healthcare Common Procedure Coding System) to code medical procedures
  * Includes average payment amount, service count, medical specialty information
  * Aggregated by provider NPI, HCPCS code, and service location

2. **LEIE (List of Excluded Individuals/Entities) Data**
- Source: Office of Inspector General (OIG)
- Update Frequency: Monthly
- Use: Identify individuals or entities excluded from federal healthcare programs
- Features: Includes criminal conviction records related to healthcare fraud

#### Data Integration Process
1. CMS data is linked with the LEIE blacklist using medical entity unique identifier (NPI)
2. Relevant features are extracted and standardized
3. A final feature library containing 950,000 records is generated

#### Feature Dimensions
- **Sample Size**: 950,000 medical records
- **Feature Count**: 11 features (7 numeric, 4 categorical)

#### Data Structure
Each record contains:
- Doctor unique identifier (NPI)
- Provider type (professional field)
- HCPCS service code
- Service location (hospital/clinic)
- Claim count and related attributes

### 🏗️ Project Structure

```
0222modelcode/
├── config/                 # Configuration files
├── models/                # Model implementations
├── utils/                 # Utility functions
├── evaluation/           # Evaluation results
├── output/               # Output files
└── logs/                 # Log files
```

### 📈 Performance Metrics

| Model | PR-AUC | F2-Score | Lift@10% |
|-------|--------|----------|-----------|
| XGBoost | 0.892 | 85.34% | 3.45x |
| RandomForest | 0.878 | 83.21% | 3.21x |
| DNN | 0.865 | 82.15% | 3.12x |
| LogisticRegression | 0.843 | 80.56% | 2.98x |

### 🚀 Quick Start

1. **Environment Setup**
```bash
pip install -r requirements.txt
```

2. **Run Experiment**
```bash
python run_experiment.py
```

### 📦 Output Files

- Data preprocessing report (with feature distribution visualization)
- Model performance comparison table (LaTeX format)
- SHAP explanation figures (300dpi TIFF format)
- Control strategy manual (with executable rules)
- Experiment logs (with MD5 checksum)

### 🔍 Feature Description

**Numeric Features**:
- Tot_Benes: Total number of beneficiaries served
- Tot_Srvcs: Total number of services provided
- Tot_Bene_Day_Srvcs: Total beneficiary service days
- Avg_Sbmtd_Chrg: Average submitted charge
- Avg_Mdcr_Alowd_Amt: Average Medicare allowed amount
- Avg_Mdcr_Pymt_Amt: Average Medicare payment amount
- Bi_Wk_Avg_SC: Bi-weekly average service cost

**Categorical Features**:
- Rndrng_Prvdr_Type: Provider type
- Rndrng_Prvdr_Gndr: Provider gender
- HCPCS_Cd: Healthcare service code
- Place_Of_Srvc: Service location 
