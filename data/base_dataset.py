import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []

    if opt.isTrain:
        if 'resize' in opt.resize_or_crop:
            transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        transform_list.append(transforms.RandomHorizontalFlip(p=0))
    else:
        if 'resize' in opt.resize_or_crop:
            osize = [opt.fineSize, opt.fineSize]
            transform_list.append(transforms.Resize(osize, Image.BICUBIC))


    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform_flip(opt):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        transform_list.append(transforms.RandomHorizontalFlip(p=1.0))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
