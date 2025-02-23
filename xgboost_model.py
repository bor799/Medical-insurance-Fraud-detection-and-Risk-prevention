import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
import xgboost as xgb
import numpy as np
from config.base_config import OUTPUT_CONFIG
import matplotlib.pyplot as plt
import joblib

class XGBoostModel(BaseModel):
    """XGBoost Model for Healthcare Fraud Detection"""
    def __init__(self):
        super().__init__("XGBoost")
        self.build()
        
    def build(self):
        """Build XGBoost model with specified configuration"""
        self.model = xgb.XGBClassifier(
            n_estimators=200,        # Number of boosting rounds
            max_depth=8,             # Maximum tree depth
            learning_rate=0.1,       # Learning rate
            subsample=0.8,           # Subsample ratio of training instances
            colsample_bytree=0.8,    # Subsample ratio of columns
            scale_pos_weight=10,     # Control class balance
            tree_method='hist',      # Use histogram-based algorithm
            n_jobs=4,                # Parallel processing
            random_state=42,         # Random seed
            early_stopping_rounds=10  # Early stopping rounds
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model"""
        self.logger.start_timer()
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train, 'train')]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val, 'validation'))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)] if X_val is not None else None,
            eval_metric=['auc', 'aucpr'],
            verbose=True
        )
        
        # Get evaluation results
        results = self.model.evals_result()
        
        # Plot training history
        if results:
            self._plot_training_history(results)
        
        # Generate feature importance report
        feature_importance = self.model.feature_importances_
        report = self._generate_importance_report(feature_importance)
        
        # Save model and report
        model_path = os.path.join(OUTPUT_CONFIG['model_dir'], f"{self.name}.joblib")
        joblib.dump(self.model, model_path)
        
        report_path = os.path.join(
            OUTPUT_CONFIG['evaluation_dir'],
            f"{self.name}_feature_importance.txt"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(report)
        self.logger.end_timer('Model Training')
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val, "Validation Set")
            
        return self.model
    
    def _generate_importance_report(self, feature_importance):
        """Generate feature importance report"""
        report = f"\n{self.name} - Feature Importance Report\n"
        report += "=" * 50 + "\n"
        
        # Sort features by importance
        feature_importance_pairs = list(zip(self.feature_names, feature_importance))
        sorted_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_pairs:
            report += f"{feature}: {importance:.4f}\n"
            
        return report
    
    def _plot_training_history(self, results):
        """Plot training metrics history"""
        plt.figure(figsize=(12, 5))
        
        # Plot AUC-PR curve
        plt.subplot(1, 2, 1)
        for dataset in ['train', 'validation']:
            if dataset in results:
                plt.plot(results[dataset]['aucpr'], 
                        label=f'{dataset.capitalize()} AUCPR')
        plt.title('AUC-PR History')
        plt.xlabel('Iteration')
        plt.ylabel('AUC-PR Score')
        plt.legend()
        plt.grid(True)
        
        # Plot AUC curve
        plt.subplot(1, 2, 2)
        for dataset in ['train', 'validation']:
            if dataset in results:
                plt.plot(results[dataset]['auc'], 
                        label=f'{dataset.capitalize()} AUC')
        plt.title('AUC History')
        plt.xlabel('Iteration')
        plt.ylabel('AUC Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(
            OUTPUT_CONFIG['figure_dir'], 
            f'{self.name}_training_history.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 