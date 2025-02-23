# 逻辑回归配置
LOGISTIC_CONFIG = {
    'penalty': 'l2',
    'C': 1.0,
    'class_weight': 'balanced',
    'random_state': 42,
    'max_iter': 1000,
    'n_jobs': 4
}

# 随机森林配置
RF_CONFIG = {
    'n_estimators': 300,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': 4,
    'random_state': 42
}

# XGBoost配置
XGB_CONFIG = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 10,
    'tree_method': 'hist',
    'n_jobs': 4,
    'random_state': 42
}

# DNN配置 (CPU版本)
DNN_CONFIG = {
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.3,
    'batch_size': 512,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'validation_split': 0.2
}

# 评估指标配置
METRICS_CONFIG = {
    'classification_metrics': [
        'accuracy', 'precision', 'recall', 'f1',
        'roc_auc', 'pr_auc'
    ],
    'threshold_metrics': [
        'confusion_matrix',
        'classification_report'
    ]
} 