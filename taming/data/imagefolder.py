import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class ImageFolderBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


def get_image_paths_from_folder(folder):
    """
    递归遍历文件夹，获取所有图片文件路径
    支持常见图片格式: jpg, jpeg, png, bmp, gif, tiff, webp
    """
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    paths = []
    
    for root, dirs, files in os.walk(folder):
        # 排序保证顺序一致性
        dirs.sort()
        files.sort()
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VALID_EXTENSIONS:
                full_path = os.path.join(root, file)
                paths.append(full_path)
    
    return paths


class ImageFolderTrain(ImageFolderBase):
    def __init__(self, train_dir, size=256, random_crop=False, labels=None):
        """
        Args:
            train_dir (str): 训练集文件夹路径，支持 ImageFolder 格式
                             例如: /path/to/train/
                                     ├── class1/
                                     │   ├── img1.jpg
                                     │   └── img2.jpg
                                     └── class2/
                                         ├── img3.jpg
                                         └── img4.jpg
            size (int): 图像resize的目标尺寸
            random_crop (bool): 是否随机裁剪
            labels (dict): 可选，额外的标签信息
        """
        super().__init__()
        self.train_dir = train_dir
        self.size = size
        self.random_crop = random_crop

        # 获取所有图片路径
        paths = get_image_paths_from_folder(train_dir)
        assert len(paths) > 0, f"No images found in {train_dir}"
        print(f"[ImageFolderTrain] Found {len(paths)} images in {train_dir}")

        self.data = ImagePaths(
            paths=paths,
            size=size,
            random_crop=random_crop,
            labels=labels
        )


class ImageFolderValidation(ImageFolderBase):
    def __init__(self, val_dir, size=256, random_crop=False, labels=None):
        """
        Args:
            val_dir (str): 验证集文件夹路径，支持 ImageFolder 格式
                           例如: /path/to/val/
                                   ├── class1/
                                   │   ├── img1.jpg
                                   │   └── img2.jpg
                                   └── class2/
                                       ├── img3.jpg
                                       └── img4.jpg
            size (int): 图像resize的目标尺寸
            random_crop (bool): 是否随机裁剪
            labels (dict): 可选，额外的标签信息
        """
        super().__init__()
        self.val_dir = val_dir
        self.size = size
        self.random_crop = random_crop

        # 获取所有图片路径
        paths = get_image_paths_from_folder(val_dir)
        assert len(paths) > 0, f"No images found in {val_dir}"
        print(f"[ImageFolderValidation] Found {len(paths)} images in {val_dir}")

        self.data = ImagePaths(
            paths=paths,
            size=size,
            random_crop=random_crop,
            labels=labels
        )