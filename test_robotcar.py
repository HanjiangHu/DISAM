import os
from options.RobotcarTestOptions import RobotcarTestOptions
from data.data_loader import DataLoader
from models.disam_model import DISAM_Model

import numpy as np

opt = RobotcarTestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

dataset = DataLoader(opt)
model = DISAM_Model(opt)

split_file = os.path.join(opt.dataroot, 'pose_new_rear.txt')
names = np.loadtxt(split_file, dtype=str, delimiter=' ', skiprows=0, usecols=(0))
with open(split_file, 'r') as f:
    poses = f.read().splitlines()

if opt.mean_cos:
    mean_cos = "meancos"
else:
    mean_cos = "plaincos"
if opt.meancos_finer:
    mean_cos_finer = "meancosfiner"
else:
    mean_cos_finer = "plaincosfiner"
if opt.use_two_stage:
    f = open("result_two_" + opt.name + "_" + opt.name_finer + "_" + opt.dataroot.split('/')[2] + "_env" + str(
        opt.test_condition) + '_' + str(opt.top_n) + "_" + mean_cos + "_" + mean_cos_finer + ".txt", 'w')
else:
    f = open("result_" + opt.name + '_' + str(opt.which_epoch) + '_' + opt.dataroot.split('/')[2] + "_env" + str(
        opt.test_condition) + "_" + mean_cos + ".txt", 'w')

# find query features
for i, data in enumerate(dataset):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    if model.get_domain() != opt.test_condition:
        continue
    model.find_query_features()
if opt.load_dist_mat:
    model.load_dist_mat()
else:
    model.find_dist_mat()
query_path, retrieved_path = model.find_retrieval()
for j in range(len(query_path)):
    print('Now:  %s' % query_path[j])
    for k in range(len(names)):
        if names[k].split('/')[-1][:-4] == retrieved_path[j].split('/')[-1][:-4]:
            name_to_be_written = query_path[j].split('/')[2] + '/' + query_path[j].split('/')[-1]
            f.write(name_to_be_written + poses[k][len(poses[k].split(' ')[0]):] + '\n')

f.close()
