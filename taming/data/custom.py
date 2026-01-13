import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, list_file, root="data/custom"):
        super().__init__()
        with open(list_file, "r") as f:
            relpaths = f.read().splitlines()
        relpaths = [relpath.strip() for relpath in relpaths if relpath.strip()]
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, list_file, root="data/custom"):
        super().__init__()
        with open(list_file, "r") as f:
            relpaths = f.read().splitlines()
        relpaths = [relpath.strip() for relpath in relpaths if relpath.strip()]
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)