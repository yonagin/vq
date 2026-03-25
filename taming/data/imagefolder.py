import os
from typing import Optional, Sequence, Union, List, Any

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


_INTERP_MAP = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
    "hamming": InterpolationMode.HAMMING,
    "box": InterpolationMode.BOX,
}


InterpolationLike = Union[str, InterpolationMode, None]


def _resolve_interpolation(
    mode: Optional[InterpolationLike],
) -> Optional[InterpolationMode]:
    if mode is None:
        return None
    if isinstance(mode, InterpolationMode):
        return mode
    key = str(mode).lower()
    if key not in _INTERP_MAP:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    return _INTERP_MAP[key]


def _default_transform(
    resize: Optional[Union[int, Sequence[int]]] = 256,
    crop_size: Optional[Union[int, Sequence[int]]] = None,
    random_crop: bool = False,
    random_flip: bool = False,
    mean: Optional[Sequence[float]] = [0.5]*3,
    std: Optional[Sequence[float]] = [0.5]*3,
    interpolation: Optional[InterpolationLike] = InterpolationMode.BICUBIC,
):
    ops: List[Any] = []

    if resize is not None:
        ops.append(
            transforms.Resize(
                resize, interpolation=interpolation or InterpolationMode.BICUBIC
            )
        )

    if crop_size is not None:
        crop_cls = transforms.RandomCrop if random_crop else transforms.CenterCrop
        ops.append(crop_cls(crop_size))

    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())

    ops.append(transforms.ToTensor())

    if mean is not None and std is not None:
        ops.append(transforms.Normalize(mean=mean, std=std))
    else:
        ops.append(transforms.Lambda(lambda t: t * 2.0 - 1.0))

    return transforms.Compose(ops)


class ImageFolderDataset(Dataset):
    """Generic ImageFolder dataset that outputs dicts compatible with taming data pipeline."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        resize: Optional[Union[int, Sequence[int]]] = 256,
        crop_size: Optional[Union[int, Sequence[int]]] = None,
        random_crop: bool = False,
        random_flip: bool = False,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
        interpolation: Optional[InterpolationLike] = InterpolationMode.BICUBIC,
    ) -> None:
        if not os.path.isdir(data_dir):
            raise ValueError(f"ImageFolder path does not exist: {data_dir}")

        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(data_dir)
        interp = _resolve_interpolation(interpolation)
        self.transform = transform or _default_transform(
            resize=resize,
            crop_size=crop_size,
            random_crop=random_crop,
            random_flip=random_flip,
            mean=normalize_mean,
            std=normalize_std,
            interpolation=interp,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)

        path, _ = self.dataset.samples[index]
        class_name = self.dataset.classes[label]

        return {
            "image": image,
            "class_label": label,
            "class_name": class_name,
            "file_path_": path,
        }


class ImageFolderTrain(ImageFolderDataset):
    def __init__(self, train_dir: str, **kwargs) -> None:
        super().__init__(
            data_dir=train_dir, random_crop=True, random_flip=True, **kwargs
        )


class ImageFolderValidation(ImageFolderDataset):
    def __init__(self, val_dir: str, **kwargs) -> None:
        super().__init__(
            data_dir=val_dir, random_crop=False, random_flip=False, **kwargs
        )


def create_imagefolder_datasets(
    train_dir: str,
    val_dir: str,
    **dataset_kwargs,
):
    """Utility to build ImageFolder train/val datasets with shared parameters."""

    train_dataset = ImageFolderTrain(train_dir=train_dir, **dataset_kwargs)
    val_dataset = ImageFolderValidation(val_dir=val_dir, **dataset_kwargs)
    return train_dataset, val_dataset
