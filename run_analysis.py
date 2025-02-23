from utils.data_analysis import DataAnalyzer
from config.base_config import DATA_CONFIG, OUTPUT_CONFIG, VIS_CONFIG
import os

def main():
    # 创建所需的目录
    os.makedirs(OUTPUT_CONFIG['figure_dir'], exist_ok=True)
    os.makedirs(OUTPUT_CONFIG['evaluation_dir'], exist_ok=True)
    
    correlation_dir = os.path.join(OUTPUT_CONFIG['figure_dir'], 'correlation')
    outlier_dir = os.path.join(OUTPUT_CONFIG['figure_dir'], 'outliers')
    os.makedirs(correlation_dir, exist_ok=True)
    os.makedirs(outlier_dir, exist_ok=True)
    
    # 创建数据分析器实例
    analyzer = DataAnalyzer()
    
    # 加载训练数据
    df = analyzer.load_data(DATA_CONFIG['train_path'])
    
    # 进行相关性分析
    analyzer.correlation_analysis(df, correlation_dir)
    
    # 2. 异常值检测
    analyzer.detect_outliers(df, outlier_dir)
    
    # 3. 组间差异检验
    test_result_path = os.path.join(
        OUTPUT_CONFIG['evaluation_dir'], 
        'group_difference_test.txt'
    )
    analyzer.group_difference_test(df, test_result_path)

if __name__ == "__main__":
    main() 