import argparse
import os
import torch


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs_now(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--name', required=True, type=str,
                                 help='Name of the experiment. It decides where to store models, '
                                      'if test on RobotCar dataset, --name must begin with robotcar')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Models are saved here')

        self.parser.add_argument('--dataroot', required=True, type=str,
                                 help='Path to dataset images (should have subfolders trainA, trainB, etc)')
        self.parser.add_argument('--n_domains', required=True, type=int, help='Number of domains to transfer among')

        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='Scaling and cropping of images at load time [resize|resize_and_crop|crop]')

        self.parser.add_argument('--loadSize', type=int, default=286, help='Scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='Crop to this size')

        self.parser.add_argument('--batchSize', type=int, default=1, help='Input batch size, try 1 if 12G memory')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--netG_n_blocks', type=int, default=9,
                                 help='Number of residual blocks to use for netG')
        self.parser.add_argument('--netG_n_shared', type=int, default=0,
                                 help='Number of blocks to use for netG shared center module')
        self.parser.add_argument('--netD_n_layers', type=int, default=4, help='Number of layers to use for netD')

        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='Instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='Insert dropout for the generator')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')

        self.parser.add_argument('--display_id', type=int, default=0,
                                 help='Window id of the web display (set >1 to use visdom)')
        self.parser.add_argument('--display_port', type=int, default=8097, help='Visdom port of the web display')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='Display window size')
        self.parser.add_argument('--margin', type=float, default=5,
                                 help='Margin parameter for triplet loss')

        self.parser.add_argument('--adapt', type=float, default=2,
                                 help='Adaptation parameter for adapted triplet loss, using 0 for original triplet loss')
        self.parser.add_argument('--margin_sam_triplet', type=float, default=0.1,
                                 help='Margin parameter for SAM triplet loss')

        self.parser.add_argument('--adapt_sam_triplet', type=float, default=1000,
                                 help='Adaptation parameter for SAM adapted triplet loss')
        self.parser.add_argument('--random_flip', action='store_true',
                                 help='If specified, flip randomly for positive and negative pairs, '
                                      'keeping negative opposite to positive as well')
        self.parser.add_argument('--mean_cos', action='store_true',
                                 help='Using mean cosine for loss metric or retrieval metric instead of original cosine'
                                      'effective only if --train_using_cos or --test_using_cos is specified')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdirs_now(expr_dir)
        if self.opt.phase == 'train':
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
