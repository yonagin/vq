#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tarfile
import zipfile
import shutil
from pathlib import Path
import urllib.request
from tqdm import tqdm
import glob

# ========== 配置部分 ==========
# 修改这里的路径为你想要保存数据的位置
DATA_ROOT = "/mnt/vq/dataset/libritts"  # 改成你想要的路径

# LibriTTS 下载链接
LIBRITTS_URLS = {
    "train-clean-100": "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/60/train-clean-360.tar.gz", 
    "train-other-500": "https://www.openslr.org/resources/60/train-other-500.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/60/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/60/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/60/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/60/test-other.tar.gz"
}

# 选择要下载的子集（可以注释掉不需要的）
DOWNLOAD_SUBSETS = [
    "train-clean-100",     # 约6GB
    "train-clean-360",   # 约23GB
    # "train-other-500",   # 约31GB
    "dev-clean",           # 约350MB
    "test-clean",          # 约740MB
]

# ========== 功能函数 ==========

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """下载文件并显示进度条"""
    print(f"正在下载: {url}")
    print(f"保存到: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"下载完成: {output_path}\n")

def extract_tar(tar_path, extract_to):
    """解压tar.gz文件"""
    print(f"正在解压: {tar_path}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="解压中"):
            tar.extract(member, extract_to)
    print(f"解压完成\n")

def save_file_lists(data_root, file_lists):
    """保存文件列表到文本文件"""
    print("[步骤 4/4] 生成文件列表")
    print("-"*30)
    
    for list_name, file_paths in file_lists.items():
        list_file = Path(data_root) / f"{list_name}.txt"
        
        # 去重并排序
        file_paths = sorted(set(file_paths))
        
        # 保存到文件
        with open(list_file, 'w', encoding='utf-8') as f:
            for path in file_paths:
                f.write(path + '\n')
        
        print(f"✓ {list_name}.txt 已生成 ({len(file_paths)} 个文件)")
    
    print(f"所有列表文件保存到: {data_root}\n")

def organize_files(data_root):
    """整理文件结构"""
    print("正在整理文件结构...")
    
    # 创建音频文件目录
    audio_dir = Path(data_root) / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # 查找所有LibriTTS目录
    libritts_dir = Path(data_root) / "LibriTTS"
    if not libritts_dir.exists():
        print("错误：找不到LibriTTS目录")
        return
    
    # 初始化文件列表字典
    file_lists = {
        "train": [],
        "dev": [],
        "test-other": []
    }
    
    # 子集到列表映射关系
    subset_mapping = {
        "train-clean-100": "train",
        "train-clean-360": "train",
        "train-other-500": "train",
        "dev-clean": "dev",
        "dev-other": "dev",
        "test-clean": "test-other",
        "test-other": "test-other"
    }
    
    # 移动所有音频文件到新的结构
    for subset_dir in libritts_dir.iterdir():
        if subset_dir.is_dir():
            subset_name = subset_dir.name
            print(f"处理子集: {subset_name}")
            
            # 确定当前子集属于哪个列表
            list_name = subset_mapping.get(subset_name)
            if not list_name:
                print(f"警告：未知的子集 {subset_name}")
                continue
            
            # 遍历每个speaker
            for speaker_dir in subset_dir.iterdir():
                if speaker_dir.is_dir():
                    speaker_id = speaker_dir.name
                    
                    # 创建speaker目录
                    new_speaker_dir = audio_dir / speaker_id
                    new_speaker_dir.mkdir(exist_ok=True)
                    
                    # 遍历每个chapter
                    for chapter_dir in speaker_dir.iterdir():
                        if chapter_dir.is_dir():
                            chapter_id = chapter_dir.name
                            
                            # 创建chapter目录
                            new_chapter_dir = new_speaker_dir / chapter_id
                            new_chapter_dir.mkdir(exist_ok=True)
                            
                            # 复制所有wav文件
                            wav_files = list(chapter_dir.glob("*.wav"))
                            for wav_file in wav_files:
                                # 重命名文件格式: speaker_chapter_xxxx.wav
                                file_id = wav_file.stem.split('_')[-1]
                                new_name = f"{speaker_id}_{chapter_id}_{file_id}.wav"
                                new_path = new_chapter_dir / new_name
                                
                                shutil.copy2(wav_file, new_path)
                                
                                # 添加相对路径到列表
                                relative_path = f"audio/{speaker_id}/{chapter_id}/{new_name}"
                                file_lists[list_name].append(relative_path)
    
    print("文件结构整理完成\n")
    return file_lists

def main():
    """主函数"""
    print("="*50)
    print("LibriTTS 数据集下载和整理脚本")
    print("="*50)
    print(f"数据将保存到: {DATA_ROOT}\n")
    
    # 创建数据根目录
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    # 创建临时下载目录
    download_dir = Path(DATA_ROOT) / "downloads"
    download_dir.mkdir(exist_ok=True)
    
    # 1. 下载数据集
    print("\n[步骤 1/4] 下载数据集")
    print("-"*30)
    for subset in DOWNLOAD_SUBSETS:
        if subset not in LIBRITTS_URLS:
            print(f"警告: 未知的子集 {subset}")
            continue
            
        url = LIBRITTS_URLS[subset]
        filename = f"{subset}.tar.gz"
        filepath = download_dir / filename
        
        # 如果文件已存在，跳过下载
        if filepath.exists():
            print(f"文件已存在，跳过下载: {filepath}")
        else:
            try:
                download_file(url, str(filepath))
            except Exception as e:
                print(f"下载失败: {e}")
                continue
    
    # 2. 解压数据集
    print("\n[步骤 2/4] 解压数据集")
    print("-"*30)
    for tar_file in download_dir.glob("*.tar.gz"):
        try:
            extract_tar(str(tar_file), DATA_ROOT)
        except Exception as e:
            print(f"解压失败 {tar_file}: {e}")
    
    # 3. 整理文件结构
    print("\n[步骤 3/4] 整理文件结构")
    print("-"*30)
    try:
        file_lists = organize_files(DATA_ROOT)
    except Exception as e:
        print(f"整理文件失败: {e}")
        return
    
    # 4. 生成文件列表
    if file_lists:
        try:
            save_file_lists(DATA_ROOT, file_lists)
        except Exception as e:
            print(f"生成列表文件失败: {e}")
    

if __name__ == "__main__":
    # 检查依赖
    try:
        from tqdm import tqdm
    except ImportError:
        print("请先安装tqdm: pip install tqdm")
        sys.exit(1)
    
    main()