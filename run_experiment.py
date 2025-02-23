import os
import sys
import time
import logging
from datetime import datetime
from config.base_config import DATA_CONFIG, OUTPUT_CONFIG, FEATURES
from utils.data_analyzer import DataAnalyzer
from utils.model_explainer import ModelExplainer
from utils.strategy_generator import StrategyGenerator
from utils.checksum import ChecksumGenerator
from utils.evaluation import generate_latex_table
from models.logistic import LogisticModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.dnn import DNNModel

def setup_logging():
    """设置日志记录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(OUTPUT_CONFIG['log_dir'], f'experiment_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('Experiment')

def create_output_dirs():
    """创建所有必要的输出目录"""
    for dir_path in OUTPUT_CONFIG.values():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"创建目录: {dir_path}")

def load_all_data():
    """加载所有数据集"""
    logging.info("开始加载数据...")
    temp_model = LogisticModel()
    
    X_train, y_train = temp_model.load_data(DATA_CONFIG['train_path'])
    X_val, y_val = temp_model.load_data(DATA_CONFIG['val_path'])
    X_test, y_test = temp_model.load_data(DATA_CONFIG['test_path'])
    
    logging.info(f"数据加载完成. 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_experiment():
    """运行完整实验"""
    # 1. 设置日志
    logger = setup_logging()
    logger.info("开始实验...")
    start_time = time.time()
    
    # 2. 创建输出目录
    create_output_dirs()
    
    # 3. 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data()
    
    # 4. 生成数据预处理报告
    logger.info("生成数据预处理报告...")
    analyzer = DataAnalyzer()
    analyzer.analyze_data(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 5. 定义所有模型
    models = {
        'Logistic': LogisticModel,
        'RandomForest': RandomForestModel,
        'XGBoost': XGBoostModel,
        'DNN': DNNModel
    }
    
    # 6. 训练和评估所有模型
    all_results = {}
    for name, model_class in models.items():
        try:
            logger.info(f"开始训练 {name} 模型...")
            model = model_class()
            model.feature_names = FEATURES['numeric'] + FEATURES['categorical']
            
            # 训练模型
            start = time.time()
            model.train(X_train, y_train, X_val, y_val)
            training_time = time.time() - start
            
            # 评估模型
            results = model.evaluate(X_test, y_test, "Test Set")
            results['training_time'] = training_time
            all_results[name] = results
            
            # 生成SHAP解释
            explainer = ModelExplainer(model.model, name)
            explainer.explain(X_train, X_test)
            
            # 生成防控策略
            strategy_gen = StrategyGenerator(model.model, model.feature_names)
            strategy_gen.generate_rules(X_train, y_train)
            strategy_gen.generate_strategy_manual(X_test, y_test, explainer.shap_values)
            
            logger.info(f"{name} 模型完成")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    # 7. 生成性能对比表
    logger.info("生成模型性能对比表...")
    generate_latex_table(all_results)
    
    # 8. 生成MD5校验
    logger.info("生成MD5校验...")
    checksum_gen = ChecksumGenerator()
    checksum_gen.generate_checksums()
    
    # 9. 完成实验
    total_time = time.time() - start_time
    logger.info(f"实验完成! 总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    run_experiment() 