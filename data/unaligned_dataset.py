import os.path, glob, torch
from data.base_dataset import BaseDataset, get_transform, get_transform_flip
from PIL import Image
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)
        self.transform_flip = get_transform_flip(opt)
        if opt.phase == 'train':
            datapath = os.path.join(opt.dataroot, opt.phase + '*')
        elif opt.name[:8] == 'robotcar':
            datapath = os.path.join(opt.dataroot, opt.phase + '*')
        else:
            datapath = os.path.join(opt.dataroot, 's' + str(opt.which_slice), opt.phase + '*')

        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        old_img = Image.open(path).convert('RGB')
        img = self.transform(old_img)
        return img, path

    def load_image_flip(self, dom, idx):
        path = self.paths[dom][idx]
        old_img = Image.open(path).convert('RGB')
        img = self.transform_flip(old_img)
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d, s in enumerate(self.sizes):
                    if index < s:
                        DA = d;
                        break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # DA is not equal to DB
            # DA = random.randint(0, len(self.dirs) - 1)
            # DB = random.randint(0, len(self.dirs) - 1)
            DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)

        flip_prob_A = random.random()
        if not self.opt.random_flip: flip_prob_A = 0
        if flip_prob_A < 0.5:
            A_img, A_path = self.load_image(DA, index_A)
        else:
            A_img, A_path = self.load_image_flip(DA, index_A)

        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            index_B = random.randint(0, self.sizes[DB] - 1)
            if flip_prob_A < 0.5:
                B_img, _ = self.load_image_flip(DB, index_B)
            else:
                B_img, _ = self.load_image(DB, index_B)
            bundle.update({'B': B_img, 'DB': DB})
            neg_B_tensor = B_img.unsqueeze(0)
            neg_DB_list = []
            if self.opt.hard_negative:
                for i in list(range(self.opt.num_hard_neg - 1)):
                    # DB = random.randint(0, 11) # whether to choose B from all the domains for hard negative samples
                    index_B_hard = random.randint(0, self.sizes[DB] - 1)
                    B_img_hard, _ = self.load_image_flip(DB, index_B_hard)
                    neg_B_tensor = torch.cat((neg_B_tensor, B_img_hard.unsqueeze(0)), 0)
                    neg_DB_list.append(DB)
            bundle.update({'neg_B_tensor': neg_B_tensor, 'neg_DB_list': neg_DB_list})

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'
