import os
import sys
from pathlib import Path

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = r"C:\Users\跳跳虎\Desktop\医疗保险：欺诈识别和风险防范\数据集"

# 数据路径配置
DATA_CONFIG = {
    'train_path': r"C:\Users\跳跳虎\Desktop\医疗保险：欺诈识别和风险防范\数据集\data_clean\processed_data\train.npz",
    'val_path': r"C:\Users\跳跳虎\Desktop\医疗保险：欺诈识别和风险防范\数据集\data_clean\processed_data\val.npz",
    'test_path': r"C:\Users\跳跳虎\Desktop\医疗保险：欺诈识别和风险防范\数据集\data_clean\processed_data\test.npz"
}

# 输出配置
OUTPUT_CONFIG = {
    'base_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output'),
    'log_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'logs'),
    'model_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'models'),
    'evaluation_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'evaluation'),
    'figure_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'figures')
}

# 可视化配置
VIS_CONFIG = {
    'font_family': 'SimHei',
    'figure_dpi': 300,
    'figure_format': 'png',
    'style': 'seaborn'
}

# 确保所有输出目录存在
for dir_path in OUTPUT_CONFIG.values():
    os.makedirs(dir_path, exist_ok=True)

# 随机种子配置
RANDOM_SEED = 42

# 更新特征配置
FEATURES = {
    'numeric': [
        'Tot_Benes',           # 服务的受益人总数
        'Tot_Srvcs',          # 提供的服务总次数
        'Tot_Bene_Day_Srvcs', # 受益人接受服务的总天数
        'Avg_Sbmtd_Chrg',     # 平均提交费用
        'Avg_Mdcr_Alowd_Amt', # 医保允许的平均金额
        'Avg_Mdcr_Pymt_Amt',  # 医保实际支付的平均金额
        'Bi_Wk_Avg_SC'        # 每两周平均服务费用
    ],
    'categorical': [
        'Rndrng_Prvdr_Type',  # 医疗服务提供者的专业或实践
        'Rndrng_Prvdr_Gndr',  # 供应商的性别
        'HCPCS_Cd',           # 医疗服务代码
        'Place_Of_Srvc'       # 服务地点
    ]
}

# 所有特征列表
ALL_FEATURES = FEATURES['numeric'] + FEATURES['categorical']

# 特征类型映射
FEATURE_TYPES = {
    feature: 'numeric' if feature in FEATURES['numeric'] else 'categorical'
    for feature in ALL_FEATURES
}

# 可视化配置
VIS_CONFIG = {
    'font_family': 'SimHei',  # 中文字体
    'figure_dpi': 300,
    'figure_format': 'png',
    'style': 'seaborn'
} 