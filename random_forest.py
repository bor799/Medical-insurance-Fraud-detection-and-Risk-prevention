import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from config.base_config import OUTPUT_CONFIG
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve

class RandomForestModel(BaseModel):
    """Random Forest Model for Healthcare Fraud Detection"""
    def __init__(self):
        super().__init__("RandomForest")
        self.build()
        
    def build(self):
        """Build Random Forest model with specified configuration"""
        self.model = RandomForestClassifier(
            n_estimators=300,      # Number of trees
            max_depth=10,          # Maximum depth of trees
            min_samples_split=5,   # Minimum samples required to split
            min_samples_leaf=2,    # Minimum samples required at leaf node
            max_features='sqrt',   # Number of features to consider for best split
            class_weight='balanced', # Handle imbalanced dataset
            n_jobs=4,             # Parallel processing
            random_state=42,      # Random seed
            verbose=1             # Enable progress messages
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Random Forest model"""
        self.logger.start_timer()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Generate feature importance report
        feature_importance = self.model.feature_importances_
        report = self._generate_importance_report(feature_importance)
        
        # Plot feature importance
        self._plot_feature_importance(feature_importance)
        
        # Plot ROC and PR curves
        if X_val is not None and y_val is not None:
            self._plot_model_curves(X_train, y_train, X_val, y_val)
        
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
    
    def _plot_feature_importance(self, feature_importance):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        
        # Plot feature importance
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xticks(range(len(feature_importance)), 
                  [self.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(
            OUTPUT_CONFIG['figure_dir'], 
            f'{self.name}_feature_importance.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_curves(self, X_train, y_train, X_val, y_val):
        """Plot ROC and PR curves for both training and validation sets"""
        plt.figure(figsize=(12, 5))
        
        # Plot PR curve
        plt.subplot(1, 2, 1)
        for X, y, label in [(X_train, y_train, 'Train'), 
                           (X_val, y_val, 'Validation')]:
            y_pred = self.model.predict_proba(X)[:, 1]
            precision, recall, _ = precision_recall_curve(y, y_pred)
            plt.plot(recall, precision, label=f'{label} PR Curve')
            
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        
        # Plot ROC curve
        plt.subplot(1, 2, 2)
        for X, y, label in [(X_train, y_train, 'Train'), 
                           (X_val, y_val, 'Validation')]:
            y_pred = self.model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred)
            plt.plot(fpr, tpr, label=f'{label} ROC Curve')
            
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(
            OUTPUT_CONFIG['figure_dir'], 
            f'{self.name}_curves.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 