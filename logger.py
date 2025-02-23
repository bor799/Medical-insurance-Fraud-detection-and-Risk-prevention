import logging
import os
import time
from config.base_config import OUTPUT_CONFIG

class Logger:
    """Custom logger for model training and evaluation"""
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 创建日志文件名（使用时间戳避免冲突）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            OUTPUT_CONFIG['log_dir'],
            f'{name}_{timestamp}.log'
        )
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.start_time = None
    
    def start_timer(self):
        """开始计时"""
        self.start_time = time.time()
        self.info("开始计时")
    
    def end_timer(self, task_name="任务"):
        """结束计时并记录耗时"""
        if self.start_time is None:
            self.warning("计时器未启动")
            return
        
        elapsed_time = time.time() - self.start_time
        self.info(f"{task_name} 执行完成，耗时: {elapsed_time:.2f} 秒")
        self.start_time = None
    
    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)
        
    def log_memory_usage(self):
        """记录当前内存使用情况"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # 转换为MB
        self.info(f"当前内存使用: {memory_usage:.2f} MB")
        
    def log_cpu_usage(self):
        """记录CPU使用情况"""
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        self.info(f"CPU使用率: {cpu_percent}%") 