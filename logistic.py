import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
from config.base_config import OUTPUT_CONFIG
import joblib

class LogisticModel(BaseModel):
    """逻辑回归模型"""
    def __init__(self):
        super().__init__("LogisticRegression")
        self.build()
        
    def build(self):
        """构建逻辑回归模型"""
        self.model = LogisticRegression(
            penalty='l2',           # L2正则化
            class_weight='balanced',# 类别权重平衡
            solver='sag',          # 随机平均梯度优化
            max_iter=1000,         # 最大迭代次数
            n_jobs=-1,             # 使用所有CPU
            random_state=42        # 随机种子
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        self.logger.start_timer()
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 计算特征重要性（系数绝对值）
        feature_importance = np.abs(self.model.coef_[0])
        
        # 保存特征重要性报告
        report = f"\n{self.name} - 特征重要性报告\n"
        report += "=" * 50 + "\n"
        
        # 将特征重要性和名称组合并排序
        feature_importance_pairs = list(zip(self.feature_names, feature_importance))
        sorted_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_pairs:
            report += f"{feature}: {importance:.4f}\n"
            
        # 保存报告
        save_path = os.path.join(
            OUTPUT_CONFIG['evaluation_dir'],
            f"{self.name}_feature_importance.txt"
        )
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        # 保存模型
        model_path = os.path.join(
            OUTPUT_CONFIG['model_dir'],
            f"{self.name}.joblib"
        )
        joblib.dump(self.model, model_path)
        
        self.logger.info(report)
        self.logger.end_timer('模型训练')
        
        # 如果提供了验证集，进行评估
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val, "验证集")
            
        return self.model 