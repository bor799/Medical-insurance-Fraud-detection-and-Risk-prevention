import pandas as pd
import os
from config.base_config import OUTPUT_CONFIG

def generate_latex_table(results_dict):
    """生成模型评估结果的LaTeX表格"""
    df = pd.DataFrame(results_dict).T
    
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{模型性能对比}\n"
    latex_table += "\\begin{tabular}{lcccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "模型 & PR-AUC & F2-Score & Lift@10\\% & 训练时间(s) \\\\\n"
    latex_table += "\\midrule\n"
    
    for model_name, results in results_dict.items():
        latex_table += f"{model_name} & "
        latex_table += f"{results['pr_auc']:.4f} & "
        latex_table += f"{results['f2_score']*100:.2f}\\% & "
        latex_table += f"{results['lift_10']:.2f}x & "
        latex_table += f"{results['training_time']:.2f} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{tab:model_comparison}\n"
    latex_table += "\\end{table}"
    
    # 保存LaTeX表格
    save_path = os.path.join(
        OUTPUT_CONFIG['evaluation_dir'],
        'model_comparison.tex'
    )
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_table) 