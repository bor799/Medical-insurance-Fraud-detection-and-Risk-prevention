import shap
import matplotlib.pyplot as plt
from config.vis_config import VIS_CONFIG
from config.base_config import FEATURES, OUTPUT_CONFIG
import os

class ModelExplainer:
    """模型解释器，使用SHAP值分析特征重要性"""
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None
        
    def explain(self, X_train, X_test):
        """计算SHAP值并生成解释图"""
        # 根据模型类型选择合适的explainer
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(X_train, 100)
            )
        
        # 计算SHAP值
        self.shap_values = self.explainer.shap_values(X_test)
        
        # 生成各类解释图
        self._plot_summary()
        self._plot_dependence()
        self._plot_interaction()
        
    def _plot_summary(self):
        """生成SHAP值汇总图"""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values,
            features=FEATURES['numeric'] + FEATURES['categorical'],
            plot_type="bar",
            show=False
        )
        plt.title(f"{self.model_name} Feature Importance (SHAP)", fontsize=VIS_CONFIG['font']['sizes']['main_title'])
        save_path = os.path.join(OUTPUT_CONFIG['figure_dir'], f'{self.model_name}_shap_summary.tiff')
        plt.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
        plt.close()
        
    def _plot_dependence(self):
        """生成SHAP依赖图"""
        for feature in FEATURES['numeric']:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values,
                show=False
            )
            plt.title(f"{feature} Dependence Plot", fontsize=VIS_CONFIG['font']['sizes']['main_title'])
            save_path = os.path.join(OUTPUT_CONFIG['figure_dir'], f'{self.model_name}_{feature}_dependence.tiff')
            plt.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
            plt.close()
            
    def _plot_interaction(self):
        """生成SHAP交互图"""
        plt.figure(figsize=(15, 10))
        shap.summary_plot(
            self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values,
            features=FEATURES['numeric'] + FEATURES['categorical'],
            plot_type="violin",
            show=False
        )
        plt.title(f"{self.model_name} Feature Interactions", fontsize=VIS_CONFIG['font']['sizes']['main_title'])
        save_path = os.path.join(OUTPUT_CONFIG['figure_dir'], f'{self.model_name}_shap_interaction.tiff')
        plt.savefig(save_path, dpi=300, format='tiff', bbox_inches='tight')
        plt.close() 