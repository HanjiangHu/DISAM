import os
from options.test_options import TestOptions
from data.data_loader import DataLoader
from models.disam_model import DISAM_Model

import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

dataset = DataLoader(opt)
model = DISAM_Model(opt)

split_file = os.path.join(opt.dataroot, 's' + str(opt.which_slice), 'pose_new_s' + str(opt.which_slice) + '.txt')
names = np.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=0, usecols=(0))
with open(split_file, 'r') as f:
    poses = f.read().splitlines()


if opt.test_using_cos:
    mode = "cos"
else:
    mode = "l2"
if opt.mean_cos:
    mean_cos = "meancos"
else:
    mean_cos = "plaincos"

if opt.use_two_stage:
    if opt.meancos_finer:
        mean_cos_finer = "meancosfiner"
    else:
        mean_cos_finer = "plaincosfiner"
else:
    mean_cos_finer = "nofiner"

if opt.use_two_stage:
    f = open("result_two_" + opt.name + "_" + opt.name_finer + "_" + '_s' + str(
        opt.which_slice) + "_" + str(opt.top_n) + "_" + mean_cos + "_" + mean_cos_finer + ".txt", 'w')
else:
    f = open("result_" + opt.name + "_" + str(opt.which_epoch) + '_s' + str(
        opt.which_slice) + "_" + mode + "_" + mean_cos + ".txt", 'w')

# test
for i, data in enumerate(dataset):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    retrieved_path = model.test(i)

    img_path = model.get_image_paths()  # query image path

    for k in range(len(names)):
        if retrieved_path == "database": break
        if names[k].split('/')[-1] == retrieved_path.split('/')[-1]:

            f.write(img_path[0].split('/')[-1] + poses[k][len(poses[k].split(' ')[0]):] + '\n')

    print('Now:  %s' % img_path[0].split('/')[-1])

f.close()
