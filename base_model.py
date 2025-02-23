import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import precision_recall_curve, auc, f1_score
from utils.logger import Logger
from config.base_config import OUTPUT_CONFIG
import os

class BaseModel(ABC):
    """基础模型类"""
    def __init__(self, name):
        self.name = name
        self.logger = Logger(name)
        self.model = None
        # 初始化特征名称
        self.feature_names = None
        
    def load_data(self, file_path):
        """加载npz格式的数据"""
        self.logger.start_timer()
        data = np.load(file_path, allow_pickle=True)
        X, y = data['X'], data['y']
        if 'feature_names' in data:
            self.feature_names = data['feature_names']
        else:
            # 如果数据中没有特征名称，使用配置文件中的特征名称
            from config.base_config import FEATURES
            self.feature_names = FEATURES['numeric'] + FEATURES['categorical']
        
        self.logger.info(f"数据加载完成，形状: {X.shape}")
        self.logger.end_timer('数据加载')
        return X, y
    
    @abstractmethod
    def build(self):
        """构建模型"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        pass
    
    def predict(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X, y, dataset_name="测试集"):
        """评估模型性能"""
        self.logger.start_timer()
        
        # 获取预测概率
        y_pred_proba = self.predict(X)
        
        # 1. 计算PR-AUC
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # 2. 计算F2-Score (使用最优阈值)
        best_f2, best_threshold = 0, 0
        for threshold in np.arange(0.1, 0.9, 0.1):
            y_pred = (y_pred_proba >= threshold).astype(int)
            # 计算precision和recall
            tp = np.sum((y == 1) & (y_pred == 1))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # 手动计算F2 Score: (5×P×R)/(4×P+R)
            if precision + recall > 0:
                f2 = (5 * precision * recall) / (4 * precision + recall)
                if f2 > best_f2:
                    best_f2 = f2
                    best_threshold = threshold
        
        # 3. 计算Lift@10%
        n_samples = len(y)
        n_top_10 = int(n_samples * 0.1)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        top_10_percent = y[sorted_indices[:n_top_10]]
        actual_fraud_rate = np.mean(top_10_percent)
        random_fraud_rate = np.mean(y)
        lift = actual_fraud_rate / random_fraud_rate
        
        # 保存评估结果
        results = {
            'dataset': dataset_name,
            'pr_auc': pr_auc,
            'f2_score': best_f2,
            'f2_threshold': best_threshold,
            'lift_10': lift,
            'n_samples': n_samples
        }
        
        # 输出评估报告
        report = f"\n{self.name} - {dataset_name}评估报告\n"
        report += "=" * 50 + "\n"
        report += f"样本数量: {n_samples:,}\n"
        report += f"PR-AUC: {pr_auc:.4f}\n"
        report += f"F2-Score: {best_f2*100:.2f}%\n"  # 按要求使用百分比形式
        report += f"最优阈值: {best_threshold:.2f}\n"
        report += f"Lift@10%: {lift:.2f}x (基准提升率: {(lift-1)*100:.1f}%)\n"
        
        # 保存评估结果
        save_path = os.path.join(
            OUTPUT_CONFIG['evaluation_dir'],
            f"{self.name}_{dataset_name}_evaluation.txt"
        )
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(report)
        self.logger.end_timer('模型评估')
        
        return results
    
    def check_overfitting(self, X_train, y_train, X_val, y_val):
        """过拟合检测"""
        self.logger.start_timer()
        
        # 1. 学习曲线分析
        train_scores = []
        val_scores = []
        train_sizes = np.linspace(0.1, 1.0, 5)
        
        for size in train_sizes:
            n_samples = int(len(X_train) * size)
            # 训练子集
            self.train(X_train[:n_samples], y_train[:n_samples], X_val, y_val)
            # 计算得分
            train_score = self.evaluate(X_train[:n_samples], y_train[:n_samples], "训练子集")
            val_score = self.evaluate(X_val, y_val, "验证集")
            train_scores.append(train_score['pr_auc'])
            val_scores.append(val_score['pr_auc'])
        
        # 计算验证集波动
        val_std = np.std(val_scores)
        train_std = np.std(train_scores)
        is_overfitting = val_std > 2 * train_std
        
        # 2. 特征重要性扰动测试
        if hasattr(self.model, 'feature_importances_'):
            baseline_score = self.evaluate(X_val, y_val, "基准验证集")['pr_auc']
            important_features = []
            
            for i, importance in enumerate(self.model.feature_importances_):
                if importance > np.mean(self.model.feature_importances_):
                    # 打乱重要特征
                    X_val_shuffled = X_val.copy()
                    np.random.shuffle(X_val_shuffled[:, i])
                    # 计算性能下降
                    shuffled_score = self.evaluate(X_val_shuffled, y_val, "扰动验证集")['pr_auc']
                    if (baseline_score - shuffled_score) / baseline_score > 0.05:
                        important_features.append(
                            (self.feature_names[i], importance, baseline_score - shuffled_score)
                        )
        
        # 生成报告
        report = f"\n{self.name} - 过拟合检测报告\n"
        report += "=" * 50 + "\n"
        report += "1. 学习曲线分析\n"
        report += f"训练集得分: {train_scores}\n"
        report += f"验证集得分: {val_scores}\n"
        report += f"验证集波动/训练集波动: {val_std/train_std:.2f}\n"
        report += f"过拟合风险: {'高' if is_overfitting else '低'}\n\n"
        
        if hasattr(self.model, 'feature_importances_'):
            report += "2. 重要特征扰动测试\n"
            for feature, importance, impact in sorted(important_features, key=lambda x: x[2], reverse=True):
                report += f"特征 {feature}: 重要度={importance:.4f}, 性能影响={impact:.4f}\n"
        
        # 保存报告
        save_path = os.path.join(
            OUTPUT_CONFIG['evaluation_dir'],
            f"{self.name}_overfitting_check.txt"
        )
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(report)
        self.logger.end_timer('过拟合检测')
        
        return {
            'is_overfitting': is_overfitting,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'important_features': important_features if hasattr(self.model, 'feature_importances_') else None
        } 