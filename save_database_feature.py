from options.test_options import TestOptions
from data.data_loader import DataLoader
from models.disam_model import DISAM_Model

opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1

dataset = DataLoader(opt)
model = DISAM_Model(opt)


for i, data in enumerate(dataset):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    if model.get_domain() == 0:
        img_path = model.get_image_paths()  # query image path
        model.save_features()
    else:
        break
    print('Now:  %s' % img_path[0].split('/')[-1], " No: ", i)



