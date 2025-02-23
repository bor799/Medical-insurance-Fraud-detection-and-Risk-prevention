import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
from config.base_config import FEATURES, OUTPUT_CONFIG
import os

class StrategyGenerator:
    """防控策略生成器"""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.rules = None
        
    def generate_rules(self, X, y, max_depth=3):
        """生成决策规则"""
        # 使用决策树提取规则
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(X, y)
        
        # 获取规则文本
        self.rules = export_text(dt, feature_names=self.feature_names)
        
        # 保存规则
        self._save_rules()
        
        return self.rules
        
    def generate_strategy_manual(self, X, y, shap_values):
        """生成防控策略手册"""
        manual = "医疗保险欺诈防控策略手册\n"
        manual += "=" * 50 + "\n\n"
        
        # 1. 高风险特征识别
        manual += "1. 高风险特征识别\n"
        manual += "-" * 30 + "\n"
        
        # 计算特征重要性
        importance = np.abs(shap_values).mean(0)
        feature_importance = list(zip(self.feature_names, importance))
        sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        
        for feature, imp in sorted_importance[:5]:
            manual += f"- {feature}: 重要性得分 {imp:.4f}\n"
            
        # 2. 风险阈值建议
        manual += "\n2. 风险阈值建议\n"
        manual += "-" * 30 + "\n"
        
        # 计算不同阈值下的效果
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precision = np.mean(y[y_pred == 1])
            recall = np.sum((y == 1) & (y_pred == 1)) / np.sum(y == 1)
            
            manual += f"阈值 {threshold:.1f}:\n"
            manual += f"  - 精确率: {precision:.2%}\n"
            manual += f"  - 召回率: {recall:.2%}\n"
            
        # 3. 执行规则
        manual += "\n3. 可执行规则集\n"
        manual += "-" * 30 + "\n"
        manual += self.rules
        
        # 保存手册
        save_path = os.path.join(OUTPUT_CONFIG['evaluation_dir'], 'fraud_prevention_manual.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(manual)
            
        return manual
        
    def _save_rules(self):
        """保存决策规则"""
        save_path = os.path.join(OUTPUT_CONFIG['evaluation_dir'], 'decision_rules.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(self.rules) 