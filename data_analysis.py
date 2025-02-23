import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from config.base_config import VIS_CONFIG, OUTPUT_CONFIG
from utils.logger import Logger

class DataAnalyzer:
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.family'] = VIS_CONFIG['font_family']
        plt.style.use(VIS_CONFIG['style'])
        self.logger = Logger('DataAnalysis')
        
        # 基础特征名称
        self.numeric_features = [
            'Tot_Benes',           # 服务的受益人总数
            'Tot_Srvcs',          # 提供的服务总次数
            'Tot_Bene_Day_Srvcs', # 受益人接受服务的总天数
            'Avg_Sbmtd_Chrg',     # 平均提交费用
            'Avg_Mdcr_Alowd_Amt', # 医保允许的平均金额
            'Avg_Mdcr_Pymt_Amt',  # 医保实际支付的平均金额
            'Bi_Wk_Avg_SC'        # 每两周平均服务费用
        ]
        
        self.categorical_features = [
            'Rndrng_Prvdr_Type',  # 医疗服务提供者的专业或实践
            'Rndrng_Prvdr_Gndr',  # 供应商的性别
            'HCPCS_Cd',           # 医疗服务代码
            'Place_Of_Srvc'       # 服务地点
        ]
        
    def load_data(self, file_path):
        """加载npz格式的数据"""
        self.logger.start_timer()
        data = np.load(file_path, allow_pickle=True)
        X, y = data['X'], data['y']
        
        # 创建DataFrame
        df = pd.DataFrame(X, columns=self.numeric_features + self.categorical_features)
        df['fraud_label'] = y
        
        self.logger.info(f"数据加载完成，形状: {df.shape}")
        self.logger.end_timer('数据加载')
        return df
        
    def correlation_analysis(self, df, save_dir):
        """Perform Spearman correlation analysis and output results"""
        self.logger.start_timer()
        
        # 设置matplotlib参数
        plt.rcParams.update({
            'font.size': 30,
            'font.weight': 'bold',
            'axes.titlesize': 30,
            'axes.labelsize': 30,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'figure.figsize': (10, 10),  # 10cm × 10cm
            'figure.dpi': 100,  # 确保1cm = 100像素
        })
        
        # 计算相关系数矩阵
        numeric_cols = self.numeric_features + ['fraud_label']
        corr_matrix = df[numeric_cols].corr(method='spearman')
        
        # 计算p值和FDR校正
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                              index=corr_matrix.index, 
                              columns=corr_matrix.columns)
        
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                if i != j:
                    stat, p = stats.spearmanr(df[numeric_cols[i]], 
                                            df[numeric_cols[j]])
                    p_values.iloc[i,j] = p
        
        p_values_flat = p_values.values[np.triu_indices_from(p_values, k=1)]
        _, p_values_adj = fdrcorrection(p_values_flat)
        
        # 创建图形
        fig = plt.figure(figsize=(10, 10))
        
        # 创建掩码矩阵（只显示下三角）
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        
        # 绘制热力图
        ax = sns.heatmap(corr_matrix, 
                        mask=mask,
                        annot=True,
                        fmt='.2f',
                        cmap='RdBu_r',
                        center=0,
                        square=True,
                        annot_kws={
                            'size': 25,
                            'weight': 'bold',
                            'color': 'black'  # 统一使用黑色文字
                        },
                        cbar_kws={
                            'label': 'Spearman Correlation Coefficient',
                            'ticks': [-1.0, -0.5, 0, 0.5, 1.0],
                            'format': '%.1f'
                        })
        
        # 获取所有文本对象并设置颜色
        texts = ax.texts
        for i in range(len(texts)):
            # 计算矩阵位置
            row = i // len(corr_matrix)
            col = i % len(corr_matrix)
            if not mask[row, col]:  # 只处理下三角部分
                correlation = corr_matrix.iloc[row, col]
                if abs(correlation) > 0.5:
                    texts[i].set_color('white')
                else:
                    texts[i].set_color('black')
        
        # 设置标题
        plt.title('Feature Correlation Heatmap', 
                 pad=120,  # 1.2cm间距
                 color='#333333',
                 fontsize=30,
                 fontweight='bold')
        
        # 添加样本量说明
        plt.figtext(0.5, 0.02,  # 底部居中
                   f'N = {len(df):,}',
                   ha='center',
                   color='#666666',
                   fontsize=25)
        
        # 调整轴标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 突出显示欺诈标签列
        fraud_idx = numeric_cols.index('fraud_label')
        ax.add_patch(plt.Rectangle((fraud_idx, 0), 1, len(numeric_cols),
                                 fill=False, edgecolor='red',
                                 linestyle='--', linewidth=2))
        
        # 标注高相关单元格
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                if not mask[i,j] and i != j:
                    if abs(corr_matrix.iloc[i,j]) > 0.8:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                                 fill=False,
                                                 edgecolor='gold',
                                                 linewidth=2))
        
        # 调整布局
        plt.subplots_adjust(top=0.85,    # 上方留2cm
                           bottom=0.15,   # 下方留3cm
                           left=0.15,
                           right=0.85)
        
        # 保存图形
        save_path = f"{save_dir}/correlation_heatmap.{VIS_CONFIG['figure_format']}"
        plt.savefig(save_path, 
                    dpi=VIS_CONFIG['figure_dpi'],
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()
        
        # 保存相关系数矩阵和p值到CSV
        results = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                results.append({
                    'Feature1': numeric_cols[i],
                    'Feature2': numeric_cols[j],
                    'Correlation': corr_matrix.iloc[i,j],
                    'P_value': p_values.iloc[i,j],
                    'P_value_adjusted': p_values_adj[len(results)],
                    'Significance': '**' if p_values_adj[len(results)] < 0.01 else ('*' if p_values_adj[len(results)] < 0.05 else '')
                })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{save_dir}/correlation_matrix.csv", index=False)
        
        # 生成LaTeX格式的相关系数表格
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{特征相关性分析结果}\n"
        latex_table += "\\begin{tabular}{lcccc}\n"
        latex_table += "\\toprule\n"
        latex_table += "特征1 & 特征2 & 相关系数 & p值 & 显著性 \\\\\n"
        latex_table += "\\midrule\n"
        
        for _, row in results_df.iterrows():
            latex_table += f"{row['Feature1']} & {row['Feature2']} & "
            latex_table += f"{row['Correlation']:.3f} & "
            latex_table += f"{row['P_value']:.3e} & {row['Significance']} \\\\\n"
        
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\label{tab:correlation}\n"
        latex_table += "\\end{table}"
        
        with open(f"{save_dir}/correlation_table.tex", 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        self.logger.info(f"相关性分析结果已保存至: {save_dir}")
        self.logger.end_timer('相关性分析')
        
    def detect_outliers(self, df, save_dir):
        """Outlier detection and visualization"""
        self.logger.start_timer()
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': 30,
            'axes.titlesize': 30,
            'axes.labelsize': 30,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25
        })
        
        # Set style
        plt.style.use('seaborn-darkgrid')
        
        # Analyze each numeric feature
        for feature in self.numeric_features:
            # Create 1x2 subplot matrix
            fig, axes = plt.subplots(1, 2, figsize=(24, 10))
            
            # 减小主标题和图表之间的间距
            plt.suptitle(f'Outlier Analysis: {feature}', fontsize=30, y=1.02)
            
            # 1. Hexbin density plot (left)
            ax = axes[0]
            hb = ax.hexbin(df[feature], df['fraud_label'], 
                          gridsize=30,
                          cmap='Blues',
                          bins='log',
                          mincnt=1,
                          extent=[df[feature].min(), df[feature].max(), -0.5, 1.5])
            
            # 减小子标题的上边距
            ax.set_title('Hexbin Plot', pad=10)
            ax.set_xlabel('Feature Value', labelpad=15)
            ax.set_ylabel('Fraud Label', labelpad=15)
            
            # Add colorbar
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Log(Count)', size=25, labelpad=15)
            cbar.ax.tick_params(labelsize=20)
            
            # Calculate and mark outlier bounds
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Mark outlier regions
            ax.axvline(x=lower_bound, color='red', linestyle='--', 
                      alpha=1.0, linewidth=3, label='Outlier Bounds')
            ax.axvline(x=upper_bound, color='red', linestyle='--', 
                      alpha=1.0, linewidth=3)
            ax.legend(fontsize=20)
            
            # 2. Q-Q plot (right)
            ax = axes[1]
            # 绘制Q-Q图
            (osm, osr), (slope, intercept, r) = stats.probplot(df[feature], dist="norm", plot=None)
            # 绘制散点
            ax.scatter(osm, osr, color='darkblue', alpha=0.6, s=80)
            # 绘制参考线
            ax.plot(osm, slope * osm + intercept, color='red', linewidth=3)
            
            ax.set_title('Q-Q Plot', pad=10)
            ax.set_xlabel('Theoretical Quantiles', labelpad=15)
            ax.set_ylabel('Sample Quantiles', labelpad=15)
            
            # 调整子图之间的间距
            plt.subplots_adjust(wspace=0.2, top=0.9)
            
            # Save plot
            save_path = f"{save_dir}/{feature}_analysis.{VIS_CONFIG['figure_format']}"
            plt.savefig(save_path, dpi=VIS_CONFIG['figure_dpi'], bbox_inches='tight')
            plt.close()
            
            # Save outlier samples
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            outliers_df = outliers[['fraud_label', feature]].copy()
            outliers_df.index.name = 'Sample_ID'
            outliers_df.to_csv(f"{save_dir}/{feature}_outliers.csv")
            
            self.logger.info(f"Outlier analysis saved for {feature}")
        
        # Generate outlier analysis report
        report = "Outlier Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Identify typical outlier patterns
        typical_patterns = []
        
        for feature in self.numeric_features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            
            # Calculate outlier percentages by fraud label
            fraud_pct = len(outliers[outliers['fraud_label']==1]) / len(outliers) * 100
            
            # Identify typical patterns
            if fraud_pct > 20:  # If more than 20% outliers are fraudulent
                pattern = {
                    'feature': feature,
                    'fraud_percentage': fraud_pct,
                    'total_outliers': len(outliers),
                    'description': f"High fraud concentration in outliers ({fraud_pct:.1f}%)"
                }
                typical_patterns.append(pattern)
            
            report += f"\n{feature} Outlier Analysis:\n"
            report += "-" * 30 + "\n"
            report += f"Feature Range: [{df[feature].min():.2f}, {df[feature].max():.2f}]\n"
            report += f"IQR Range: [{Q1:.2f}, {Q3:.2f}]\n"
            report += f"Outlier Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]\n"
            report += f"Total Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)\n"
            
            # Statistics by fraud label
            outliers_by_label = outliers.groupby('fraud_label').size()
            for label, count in outliers_by_label.items():
                label_name = 'Fraud' if label == 1 else 'Normal'
                report += f"{label_name} samples in outliers: {count} ({count/len(outliers)*100:.2f}%)\n"
        
        # Add typical patterns section
        report += "\nTypical Outlier Patterns:\n"
        report += "=" * 30 + "\n"
        for i, pattern in enumerate(typical_patterns[:3], 1):
            report += f"\nPattern {i}:\n"
            report += f"Feature: {pattern['feature']}\n"
            report += f"Description: {pattern['description']}\n"
            report += f"Total Outliers: {pattern['total_outliers']}\n"
        
        with open(f"{save_dir}/outliers_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.end_timer('Outlier Detection')
        
    def group_difference_test(self, df, save_path):
        """进行组间差异检验"""
        self.logger.start_timer()
        
        results = []
        for feature in self.numeric_features:
            # 获取两组数据
            fraud_group = df[df['fraud_label'] == 1][feature]
            normal_group = df[df['fraud_label'] == 0][feature]
            
            # 进行Mann-Whitney U检验
            statistic, p_value = stats.mannwhitneyu(fraud_group, normal_group, 
                                                  alternative='two-sided')
            
            # 计算效应量
            n1, n2 = len(fraud_group), len(normal_group)
            effect_size = abs(statistic - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            
            # 计算描述统计量
            fraud_stats = fraud_group.describe()
            normal_stats = normal_group.describe()
            
            results.append({
                'feature': feature,
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size,
                'fraud_mean': fraud_stats['mean'],
                'normal_mean': normal_stats['mean'],
                'fraud_std': fraud_stats['std'],
                'normal_std': normal_stats['std']
            })
        
        # 进行FDR校正
        _, p_values_adj = fdrcorrection([r['p_value'] for r in results])
        for r, p_adj in zip(results, p_values_adj):
            r['p_value_adjusted'] = p_adj
            r['significance'] = '**' if p_adj < 0.01 else ('*' if p_adj < 0.05 else '')
        
        # 生成LaTeX格式的表格
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{组间差异检验结果}\n"
        latex_table += "\\begin{tabular}{lccccccc}\n"
        latex_table += "\\toprule\n"
        latex_table += "特征 & U统计量 & p值 & 校正后p值 & 效应量 & 欺诈组均值 & 正常组均值 & 显著性 \\\\\n"
        latex_table += "\\midrule\n"
        
        for r in results:
            latex_table += f"{r['feature']} & {r['statistic']:.2e} & {r['p_value']:.3e} & "
            latex_table += f"{r['p_value_adjusted']:.3e} & {r['effect_size']:.3f} & "
            latex_table += f"{r['fraud_mean']:.3f} & {r['normal_mean']:.3f} & {r['significance']} \\\\\n"
        
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\label{tab:group_difference}\n"
        latex_table += "\\end{table}"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        self.logger.end_timer('组间差异检验')
        return pd.DataFrame(results) 