from glob import glob

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from nibabel.viewers import OrthoSlicer3D
from torch.utils.data import DataLoader, Dataset


class CT3DDataset(Dataset):
    """
    :param images: 3D CT图像的路径
    """

    def __init__(self, images, targets=None, transforms=None):
        self.transforms = transforms
        self.image_slices = []
        self.target_slices = []
        if targets is not None:
            for image_path, target_path in zip(images, targets):
                image_data = nib.load(image_path)
                target_data = nib.load(target_path)
                image = image_data.get_fdata()
                target = target_data.get_fdata()

                image, target = remove_no_lung_slice(image, target)
                image = image.astype('float32')
                target = target.astype('float32')

                slices = image.shape[-1]
                for i in range(slices):
                    self.image_slices.append(np.expand_dims(image[:, :, i], -1))
                    self.target_slices.append(np.expand_dims(target[:, :, i], -1))
        else:
            for image_path in images:
                image_data = nib.load(image_path)
                image = image_data.get_fdata()
                slices = image.shape[-1]
                for i in range(slices):
                    self.image_slices.append(np.expand_dims(image[:, :, i], -1))

    def __getitem__(self, item):
        assert len(self.image_slices) == len(self.target_slices)
        image_slice = self.image_slices[item]
        target_slice = None
        if self.target_slices is not None:
            target_slice = self.target_slices[item]
        if self.transforms is not None:
            image_slice, target_slice = self.transforms(image_slice, target_slice)
        return image_slice, target_slice

    def __len__(self):
        return len(self.image_slices)


def split_train_val(data_path, target_path=None, val_size=0.2):
    """从训练集中划分出验证集

    :param data_path: 整个训练集的CT图像路径集合
    :param target_path: 整个训练集的CT图像真实分割图的路径集合
    :param val_size: 验证集的比例，默认为0.2
    :return: 划分后的训练集和验证集的图像路径集合/图像路径及对应的图像真实分割图路径集合
    """
    data_length = len(data_path)
    val_length = int(data_length * val_size)
    train_length = data_length - val_length

    np.random.seed(42)
    np.random.shuffle(data_path)

    train_path = data_path[: train_length]
    val_path = data_path[train_length:]
    if target_path is not None:
        assert len(data_path) == len(target_path)
        np.random.seed(42)
        np.random.shuffle(target_path)
        train_mask_path = target_path[: train_length]
        val_mask_path = target_path[train_length:]
        return train_path, train_mask_path, val_path, val_mask_path
    return train_path, val_path


def get_relate_target(image_paths, target_paths, dataset='B'):
    """可能由于系统的缘故，文件夹下面的文件排列顺序是不一致的，因此需要将图像和其对应的标签图按顺序进行对齐

    """
    reordered_target_paths = []
    if dataset == 'B':
        for path in image_paths:
            split_str = path.split('_ct')
            name_prefix = split_str[0]
            name_suffix = split_str[-1]
            target_path = name_prefix + '_seg' + name_suffix
            if target_path not in target_paths:
                raise RuntimeError('target path is not exist!')
            reordered_target_paths.append(target_path)
    elif dataset == 'A':
        pass
    return reordered_target_paths


def volume_resample(image_volume, target_volume=None):
    """对三维体素进行重采样到相同大小，因为不同的数据可能它的体素值不一样，这样不利于模型训练

    :param image_volume: 图像
    :param target_volume: 图像对应的标注
    :return: 重采样后的图像 or 重采样后的图像以及对应的标注
    """
    # todo 体素重采样还是有必要做一下的，以防万一，先用没有重采样的数据训练，然后使用重采样后的数据训练，看一下是否有变化
    pass


def remove_no_lung_slice(image_volume, target_volume):
    """移除CT图像中沿着深度方向没有肺部的slice

    :param image_volume: CT图像, numpy ndarray
    :param target_volume: CT图像对应的标注, numpy ndarray
    :return:
    """
    assert image_volume.shape == target_volume.shape
    depth = image_volume.shape[-1]
    head_index = _search_index(target_volume, 0, depth // 2)
    foot_index = _search_index(target_volume, depth // 2, depth, reverse=True)
    target_volume = target_volume[:, :, head_index:foot_index]
    image_volume = image_volume[:, :, head_index:foot_index]
    return image_volume, target_volume


def _search_index(target_volume, left, right, reverse=False):
    """

    :param target_volume:
    :param left:
    :param right:
    :param reverse: 如果reverse为False，表示寻找从头部向脚部方向的索引，否则表示寻找从脚部向头部方向的索引
    :return:
    """
    mid = 0
    while left < right:
        mid = (left + right) // 2
        if mid == left or mid == right:
            break
        if np.max(target_volume[:, :, :mid]) == 0:
            left = mid
            if reverse:
                right = mid
        else:
            right = mid
            if reverse:
                left = mid
    return mid

    # if np.max(target_volume[:, :, mid]) == 0.:  # 全部像素相同，即不包含肺部
    #     left = mid
    #     if reverse:
    #         right = mid
    # else:
    #     right = mid
    #     if reverse:
    #         left = mid
    # if left >= right:
    #     return mid
    # else:
    #     _search_index(target_volume, left, right)


def load_dataset(dataset_select='B', batch_size=1, train=True, train_transforms=None, test_transforms=None):
    image_paths = []
    target_paths = []
    if train:
        if dataset_select.find('A') != -1:
            image_paths.extend(glob('/home/ivan/Xiong/COVID-19-CT-Seg_20cases/COVID-19-CT-Seg_20cases/*'))
            target_paths.extend(glob('/home/ivan/Xiong/COVID-19-CT-Seg_20cases/Lung_and_Infection_Mask/*'))
            target_paths = get_relate_target(image_paths, target_paths, dataset='A')
        if dataset_select.find('B') != -1:
            data_path = glob('/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/*')
            image_paths.extend([path for path in data_path if path.find('ct') != -1])
            target_paths.extend([path for path in data_path if path.find('seg') != -1])
            target_paths = get_relate_target(image_paths, target_paths, dataset='B')
        # 从训练集中分割出验证集
        train_paths, train_mask_paths, val_paths, val_mask_paths = split_train_val(image_paths, target_paths)
        train_dataset = CT3DDataset(train_paths, train_mask_paths, transforms=train_transforms)
        val_dataset = CT3DDataset(val_paths, val_mask_paths, transforms=test_transforms)
        # 获取训练集和验证集的DataLoader
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=False)
        return train_loader, val_loader
    else:
        pass


def show_ct(path):
    image_path = '/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0011_ct.nii.gz'
    target_path = '/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0011_seg.nii.gz'
    image_data = nib.load(image_path)
    target_data = nib.load(target_path)
    image = image_data.get_fdata()
    target = target_data.get_fdata()
    print(target)
    # OrthoSlicer3D(image).show()
    OrthoSlicer3D(target).show()
    depth = image.shape[-1]
    figure, axes = plt.subplots(6, 6)
    for i in range(depth):
        if i >= 36:
            break
        axes[i // 6][i - (i // 6) * 6].imshow(target[:, :, i], cmap='gray')
    plt.show()


if __name__ == '__main__':
    image_path = '/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0003_ct.nii.gz'
    target_path = '/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0003_seg.nii.gz'
    image_data = nib.load(image_path)
    target_data = nib.load(target_path)
    image = image_data.get_fdata()
    target = target_data.get_fdata()
    image_, target_ = remove_no_lung_slice(image, target)
    OrthoSlicer3D(target_).show()
