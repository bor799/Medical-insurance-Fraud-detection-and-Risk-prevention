import hashlib
import os
from config.base_config import OUTPUT_CONFIG

class ChecksumGenerator:
    """生成实验文件的MD5校验"""
    def __init__(self):
        self.checksums = {}
        
    def generate_checksums(self):
        """为所有输出文件生成MD5校验值"""
        # 遍历所有输出目录
        for dir_name, dir_path in OUTPUT_CONFIG.items():
            if os.path.exists(dir_path):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        self.checksums[file_path] = self._calculate_md5(file_path)
                        
        # 保存校验文件
        self._save_checksums()
        
    def _calculate_md5(self, file_path):
        """计算单个文件的MD5值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def _save_checksums(self):
        """保存所有文件的MD5校验值"""
        output = "实验文件MD5校验值\n"
        output += "=" * 50 + "\n\n"
        
        for file_path, checksum in sorted(self.checksums.items()):
            rel_path = os.path.relpath(file_path, os.path.dirname(OUTPUT_CONFIG['base_dir']))
            output += f"{rel_path}\n{checksum}\n\n"
            
        save_path = os.path.join(OUTPUT_CONFIG['evaluation_dir'], 'checksums.md5')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output) 