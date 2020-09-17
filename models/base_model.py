import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not self.isTrain: self.save_dir_finer = os.path.join(opt.checkpoints_dir, opt.name_finer)
    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch, gpu_ids):
        save_filename = '%d_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.save(save_path)
        if gpu_ids and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch,use_two_stage=False):
        save_filename = '%d_net_%s' % (epoch, network_label)
        if use_two_stage:
            save_path = os.path.join(self.save_dir_finer,save_filename)
            print('two stage', save_path)
        else:
            save_path = os.path.join(self.save_dir, save_filename)
            print('no two stage',save_path)
        network.load(save_path)

    def save_att_network(self,network, network_label, epoch, gpu_ids):
        save_filename = '%d_att_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.save(save_path)
        if gpu_ids and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_att_network(self,network, network_label, epoch):
        save_filename = '%d_att_net_%s' % (epoch, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load(save_path)

    def update_learning_rate():
        pass
