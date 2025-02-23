import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.vis_config import VIS_CONFIG
from config.base_config import FEATURES, OUTPUT_CONFIG
import os

class DataAnalyzer:
    """数据预处理报告生成器"""
    def __init__(self):
        self.report = []
        
    def analyze_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """生成完整的数据分析报告"""
        self._add_section("数据预处理报告")
        
        # 1. 数据集基本信息
        self._add_dataset_info(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # 2. 特征分布分析
        self._analyze_distributions(X_train, "train")
        
        # 3. 特征相关性分析
        self._analyze_correlations(X_train)
        
        # 4. 类别不平衡分析
        self._analyze_class_imbalance(y_train, y_val, y_test)
        
        # 保存报告
        self._save_report()
        
    def _add_section(self, title):
        """添加报告章节"""
        self.report.append(f"\n{title}\n{'='*50}\n")
        
    def _add_dataset_info(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """添加数据集基本信息"""
        info = f"""
数据集大小:
- 训练集: {X_train.shape[0]:,} 样本, {X_train.shape[1]} 特征
- 验证集: {X_val.shape[0]:,} 样本, {X_val.shape[1]} 特征
- 测试集: {X_test.shape[0]:,} 样本, {X_test.shape[1]} 特征

特征类型:
- 数值特征 ({len(FEATURES['numeric'])}): {', '.join(FEATURES['numeric'])}
- 类别特征 ({len(FEATURES['categorical'])}): {', '.join(FEATURES['categorical'])}
"""
        self.report.append(info)
        
    def _analyze_distributions(self, X, dataset_name):
        """分析并可视化特征分布"""
        self._add_section("特征分布分析")
        
        # 数值特征分布图
        fig, axes = plt.subplots(2, 4, figsize=(12.5, 8))  # A4纸1/4大小
        axes = axes.ravel()
        
        # 将数据转换为DataFrame
        df = pd.DataFrame(X, columns=FEATURES['numeric'] + FEATURES['categorical'])
        
        for i, feature in enumerate(FEATURES['numeric']):
            # 使用matplotlib直接绘制直方图
            axes[i].hist(df[feature].values, bins=50, density=True)
            axes[i].set_title(feature, fontsize=VIS_CONFIG['font']['sizes']['main_title'])
            axes[i].tick_params(labelsize=VIS_CONFIG['font']['sizes']['values'])
            
            # 添加描述性统计
            mean = df[feature].mean()
            std = df[feature].std()
            self.report.append(f"\n{feature}统计信息:")
            self.report.append(f"- 均值: {mean:.2f}")
            self.report.append(f"- 标准差: {std:.2f}")
            self.report.append(f"- 最小值: {df[feature].min():.2f}")
            self.report.append(f"- 最大值: {df[feature].max():.2f}")
            
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_CONFIG['figure_dir'], 'numeric_distributions.tiff')
        plt.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
        plt.close()
        
    def _analyze_correlations(self, X):
        """分析特征相关性"""
        self._add_section("特征相关性分析")
        
        # 将数据转换为DataFrame
        df = pd.DataFrame(X, columns=FEATURES['numeric'] + FEATURES['categorical'])
        corr_matrix = df.corr()
        
        plt.figure(figsize=(12.5, 10))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdBu_r',
                   square=True)
        plt.title("特征相关性矩阵", fontsize=VIS_CONFIG['font']['sizes']['main_title'])
        
        # 记录重要的相关性
        self.report.append("\n重要特征相关性:")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # 只记录相关性大于0.5的
                    self.report.append(
                        f"- {corr_matrix.columns[i]} 与 {corr_matrix.columns[j]}: {corr:.2f}"
                    )
        
        save_path = os.path.join(OUTPUT_CONFIG['figure_dir'], 'feature_correlations.tiff')
        plt.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
        plt.close()
        
    def _analyze_class_imbalance(self, y_train, y_val, y_test):
        """分析类别不平衡情况"""
        self._add_section("类别不平衡分析")
        
        datasets = {
            '训练集': y_train,
            '验证集': y_val,
            '测试集': y_test
        }
        
        # 绘制类别分布图
        plt.figure(figsize=(10, 6))
        for name, y in datasets.items():
            fraud_rate = np.mean(y)
            self.report.append(f"{name}欺诈率: {fraud_rate:.2%}")
            
            # 计算每个类别的样本数
            counts = np.bincount(y)
            plt.bar([name + '_正常', name + '_欺诈'], counts)
            
        plt.title("数据集类别分布", fontsize=VIS_CONFIG['font']['sizes']['main_title'])
        plt.xticks(rotation=45)
        plt.ylabel("样本数量")
        
        save_path = os.path.join(OUTPUT_CONFIG['figure_dir'], 'class_distribution.tiff')
        plt.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
        plt.close()
            
    def _save_report(self):
        """保存完整报告"""
        report_text = '\n'.join(self.report)
        save_path = os.path.join(OUTPUT_CONFIG['evaluation_dir'], 'data_preprocessing_report.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text) 