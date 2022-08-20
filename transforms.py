import random

import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms.functional as Func


class ToPILImage(object):
    def __call__(self, image, target=None):
        image = Func.to_pil_image(image)
        if target is not None:
            target = Func.to_pil_image(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        seed = np.random.randint(65536)
        torch.manual_seed(seed)
        crop = torchvision.transforms.RandomCrop(self.size)
        image = crop(image)
        if target is not None:
            target = crop(target)
        return image, target


class Resize(object):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def __call__(self, image, target=None):
        image = Func.resize(image, self.size)
        if target is not None:
            target = Func.resize(target, self.size, interpolation=Func.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = Func.hflip(image)
            if target is not None:
                target = Func.hflip(target)
        return image, target


class ColorJitter(object):
    def __call__(self, image, target):
        color_jitter = torchvision.transforms.ColorJitter()
        image = color_jitter(image)
        return image, target


class GrayScale(object):
    def __call__(self, image, target):
        gray_scale = torchvision.transforms.Grayscale()
        image = gray_scale(image)
        return image, target


class RandomRotation(object):
    def __init__(self, degrees):
        super(RandomRotation, self).__init__()
        self.degrees = degrees

    def __call__(self, image, target=None):
        degree = random.randint(self.degrees[0], self.degrees[1])
        image = Func.rotate(image, degree)
        if target is not None:
            target = Func.rotate(target, degree)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = Func.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        image = Func.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
