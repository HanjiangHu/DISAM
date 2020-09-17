import numpy as np
import os
import ntpath
import time
from .util import util

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port)

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0:
            # show images in the browser
            idx = 1
            for label, image_numpy in visuals.items():
                self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                idx += 1


    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        #print("#################")
        #print(errors)
        #print('#################')
        self.plot_data['Y'].append([np.mean(errors[k]) for k in self.plot_data['legend']])
        #print(np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1))
        #print(np.array(self.plot_data['Y']))
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id
        )

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            v = ['%.3f' % iv for iv in v]
            message += k + ': ' + ', '.join(v) + ' | '

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util().save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_image_matrix(self, visuals_list, save_path):
        images_list = []
        get_domain = lambda x: x.split('_')[-1]

        for visuals in visuals_list:
            pairs = list(visuals.items())
            real_label, real_img = pairs[0]
            real_dom = get_domain(real_label)

            for label, img in pairs:
                if 'fake' not in label:
                    continue
                if get_domain(label) == real_dom:
                    images_list.append(real_img)
                else:
                    images_list.append(img)

        immat = self.stack_images(images_list)
        util().save_image(immat, save_path)

    # reshape a list of images into a square matrix of them
    def stack_images(self, list_np_images):
        n = int(np.ceil(np.sqrt(len(list_np_images))))

        # add padding between images
        for i, im in enumerate(list_np_images):
            val = 255 if i%n == i//n else 0
            r_pad = np.pad(im[:,:,0], (3,3), mode='constant', constant_values=0)
            g_pad = np.pad(im[:,:,1], (3,3), mode='constant', constant_values=val)
            b_pad = np.pad(im[:,:,2], (3,3), mode='constant', constant_values=0)
            list_np_images[i] = np.stack([r_pad,g_pad,b_pad], axis=2)

        data = np.array(list_np_images)
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        return data
