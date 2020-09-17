import numpy as np
import torch
import copy, os
from collections import OrderedDict
from util.util import util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import glob
import torch.nn.functional as F
import cv2
from skimage import io


def norm_image(image):
    """
    :param image: image with [H,W,C]
    :return: image in np uint8
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


class DISAM_Model(BaseModel):
    def name(self):
        return 'DISAM_Model'

    def __init__(self, opt):
        super(DISAM_Model, self).__init__(opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None
        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # used metrics
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        self.mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.L2loss = torch.nn.MSELoss()

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        if not self.isTrain:
            self.use_two_stage = opt.use_two_stage
            if self.use_two_stage:
                self.netG_finer = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                                    opt.netG_n_blocks, opt.netG_n_shared,
                                                    self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
                self.top_n = opt.top_n

        self.last_retrieval_index_c0 = 0
        self.last_retrieval_index_c1 = 0
        self.last_domain = 0

        if self.isTrain:
            blur_fn = lambda x: torch.nn.functional.conv2d(x, self.Tensor(util().gkern_2d()), groups=3, padding=2)
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, blur_fn, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if not self.isTrain:
                if opt.use_two_stage:
                    self.load_network(self.netG_finer, 'G', opt.which_epoch_finer, self.use_two_stage)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if not self.isTrain:
            self.test_using_cos = opt.test_using_cos
            # used for retrieval
            self.database_feature_c0 = []
            self.database_path_c0 = []
            self.database_feature_c1 = []
            self.database_path_c1 = []
            self.database_dist_list_c0 = []  # only for visualization
            self.query_feature_list = []
            self.dist_mat_torch = None
            self.robotcar_database = []

        if self.isTrain:
            self.neg_B = self.Tensor(opt.num_hard_neg, opt.input_nc, opt.fineSize, opt.fineSize)
            self.train_using_cos = opt.train_using_cos
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # used in the adaptive triplet loss
            self.margin = opt.margin
            self.adapt = opt.adapt
            self.margin_sam_triplet = opt.margin_sam_triplet
            self.adapt_sam_triplet = opt.adapt_sam_triplet
            self.use_realAB_as_negative = opt.use_realAB_as_negative
            self.hard_negative = opt.hard_negative
            # define loss functions
            self.criterionCycle = torch.nn.SmoothL1Loss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0] * self.n_domains, [0] * self.n_domains
            self.loss_cycle = [0] * self.n_domains
            self.loss_triplet = [0] * self.n_domains
            self.loss_sam = [0] * self.n_domains
            self.loss_sam_triplet = [0] * self.n_domains
            self.feature_distance = [0] * self.n_domains
            self.feature_cos = [0] * self.n_domains
            self.use_cos_latent_with_L2 = opt.use_cos_latent_with_L2
            # initialize loss multipliers
            self.lambda_triplet, self.lambda_cyc, self.lambda_latent = opt.lambda_triplet, opt.lambda_cycle, opt.lambda_latent
            self.lambda_sam, self.lambda_sam_triplet = opt.lambda_sam, opt.lambda_sam_triplet

    def set_input(self, input):
        input_A = input['A']
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
            if self.hard_negative:
                self.neg_B = input['neg_B_tensor'][0].cuda()
                self.neg_DB_list = input['neg_DB_list'][0]
        self.image_paths = input['path']

    def image_retrieval(self, query_encoded, query_path, query_encoded_finer=None, test_index=-1):
        """
        Used to retrieve the target image in the database given the query encoded feature
        :param query_encoded: the query code
        :param query_path: the path of query image
        :param query_encoded_finer: the query code in the finer retrieval model
        :param test_index: the index of input query images when testing
        :return: the retrieved iamge path and the encoded feature in the database
        """
        min_dix = 100000
        if self.use_two_stage:
            top_n_tensor = torch.ones(self.top_n) * 100000
            top_n_tensor = top_n_tensor.cuda()
            top_n_index = torch.ones(self.top_n)
        path = None
        final_index = 0

        if query_path.split('/')[-1][11] == '0':
            # for c0, camera 0 in the CMU-Seasons dataset
            self.database_dist_list_c0 = []
            for i, db_path in enumerate(self.database_path_c0):
                if self.test_using_cos:
                    # use the cosine retrieval metric
                    if self.opt.mean_cos:
                        dist = -self.mean_cos(query_encoded.view(256, -1),
                                              self.database_feature_c0[i][0].view(256, -1)).mean(0)
                    else:
                        dist = -self.cos(query_encoded.view(-1),
                                         self.database_feature_c0[i][0].view(-1))
                else:
                    # use L2 metric
                    dist = self.L2loss(query_encoded.view(-1), self.database_feature_c0[i][0].view(-1))
                self.database_dist_list_c0.append(dist.item())
                if not self.use_two_stage:
                    if dist < min_dix:
                        min_dix = dist
                        final_index = i
                        path = db_path
                else:
                    # find top N for finer retrieval
                    if dist < top_n_tensor[self.top_n - 1]:
                        top_n_tensor[self.top_n - 1] = dist
                        top_n_index[self.top_n - 1] = i
                        tmp = top_n_tensor.sort()
                        top_n_tensor = tmp[0]
                        top_n_index = top_n_index[tmp[1]]
            if self.use_two_stage:
                # from coarse to fine strategy
                for i in list(range(self.top_n)):
                    if self.test_using_cos:
                        if self.opt.meancos_finer:
                            dist = -self.mean_cos(query_encoded_finer.view(256, -1),
                                                  self.database_feature_c0[top_n_index[i].int()][1].view(256, -1)).mean(0)
                        else:
                            dist = -self.cos(query_encoded_finer.view(-1),
                                             self.database_feature_c0[top_n_index[i].int()][1].view(-1))
                    else:
                        dist = self.L2loss(query_encoded_finer.view(-1),
                                           self.database_feature_c0[top_n_index[i].int()][1].view(-1))
                    if dist < min_dix:
                        min_dix = dist
                        final_index = top_n_index[i].int()
                        path = self.database_path_c0[final_index]
                if self.opt.save_sam_visualization and test_index % 10 == 0:
                    # save the visualized SAM maps
                    self.find_grad_sam(query_encoded_finer, query_path, self.database_feature_c0[
                        self.database_dist_list_c0.index(sorted(self.database_dist_list_c0)[100])][1], test_index, 100)
                    self.find_grad_sam(self.database_feature_c0[
                                           self.database_dist_list_c0.index(sorted(self.database_dist_list_c0)[100])][
                                           1], self.database_path_c0[
                                           self.database_dist_list_c0.index(sorted(self.database_dist_list_c0)[100])],
                                       query_encoded_finer, test_index, 100)
                    self.find_grad_sam(query_encoded_finer, self.image_paths[0],
                                       self.database_feature_c0[final_index][1], test_index)
                    self.find_grad_sam(self.database_feature_c0[final_index][1], path, query_encoded_finer, test_index)
            print("Minimun distance is :", min_dix.item(), " least index: ", final_index)
            print("Retrieved path: ", path.split('/')[-1], " query path: ", query_path.split('/')[-1])
        else:
            for i, db_path in enumerate(self.database_path_c1):
                # for camera 1
                if self.test_using_cos:
                    if self.opt.mean_cos:
                        dist = -self.mean_cos(query_encoded.view(256, -1),
                                         self.database_feature_c1[i][0].view(256, -1)).mean(0)
                    else:
                        dist = -self.cos(query_encoded.view(-1),
                                    self.database_feature_c1[i][0].view(-1))  # + L2loss(query_encoded,item[1])*0
                else:
                    dist = self.L2loss(query_encoded.view(-1), self.database_feature_c1[i][0].view(-1))
                if not self.use_two_stage:
                    if dist < min_dix:
                        min_dix = dist
                        final_index = i
                        path = db_path
                else:
                    if dist < top_n_tensor[self.top_n - 1]:
                        top_n_tensor[self.top_n - 1] = dist
                        top_n_index[self.top_n - 1] = i
                        tmp = top_n_tensor.sort()
                        top_n_tensor = tmp[0]
                        top_n_index = top_n_index[tmp[1]]
            if self.use_two_stage:
                for i in list(range(self.top_n)):
                    if self.test_using_cos:
                        if self.opt.meancos_finer:
                            dist = -self.mean_cos(query_encoded_finer.view(256, -1),
                                             self.database_feature_c1[top_n_index[i].int()][1].view(256, -1)).mean(0)
                        else:
                            dist = -self.cos(query_encoded_finer.view(-1),
                                        self.database_feature_c1[top_n_index[i].int()][1].view(-1))
                    else:
                        dist = self.L2loss(query_encoded_finer.view(-1),
                                      self.database_feature_c1[top_n_index[i].int()][1].view(-1))
                    if dist < min_dix:
                        min_dix = dist
                        final_index = top_n_index[i].int()
                        path = self.database_path_c1[final_index]
            print("Minimun distance is :", min_dix.item(), " least index: ", final_index)
            print("Retrieved path: ", path.split('/')[-1], " query path: ", query_path.split('/')[-1])
        if query_path.split('/')[-1][11] == '0':
            if self.use_two_stage:
                return path, self.database_feature_c0[final_index][1]
            else:
                return path, self.database_feature_c0[final_index][0]
        else:
            if self.use_two_stage:
                return path, self.database_feature_c1[final_index][1]
            else:
                return path, self.database_feature_c1[final_index][0]

    def test(self, index=0):
        with torch.no_grad():
            self.visuals = [self.real_A]
            self.labels = ['query_image_%d' % self.DA]
            raw_encoded = self.netG.encode(self.real_A, self.DA)
            raw_encoded_finer = None
            if self.use_two_stage: raw_encoded_finer = self.netG_finer.encode(self.real_A, self.DA)
            if self.DA == 0:
                # building the feature database
                db_path = copy.deepcopy(self.image_paths[0])
                if db_path.split('/')[-1][11] == '0':
                    self.database_feature_c0.append((raw_encoded, raw_encoded_finer))
                    self.database_path_c0.append(db_path)
                else:
                    self.database_feature_c1.append((raw_encoded, raw_encoded_finer))
                    self.database_path_c1.append(db_path)
                return "database"
            else:
                path, retrieved_image = self.image_retrieval(raw_encoded, self.image_paths[0], raw_encoded_finer, index)
                return path

    def find_grad_sam(self, raw_encoded, query_path, retrieved_image, index, rank=-1):
        with torch.set_grad_enabled(True):
            new_raw_encoded = copy.deepcopy(raw_encoded.view(256, 64, 64)).cuda()
            new_raw_encoded.requires_grad_(True)
            new_retrieved_image = copy.deepcopy(retrieved_image.view(256, 64, 64)).cuda()
            mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
            mean_cos_similarity = mean_cos(new_raw_encoded.view(256, -1), new_retrieved_image.view(256, -1)).mean(0)
            mean_cos_similarity.backward()

            mask = F.relu(torch.mul(new_raw_encoded,
                                    new_raw_encoded.grad.sum(1).sum(1).view(256, 1, 1).expand([256, 64, 64])).sum(
                dim=0))
            # normalization
            mask -= mask.min()
            mask /= mask.max()
            mask = cv2.resize(mask.cpu().detach().numpy(), (256, 256))
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]  # gbr to rgb
            img = io.imread(query_path)
            img = np.float32(cv2.resize(img, (256, 256))) / 255

            sam = heatmap + np.float32(img)
            sam = norm_image(sam)
            heatmap = norm_image(heatmap)
            img = norm_image(img)
            if not os.path.exists(self.opt.sam_matched_dir):
                os.makedirs(self.opt.sam_matched_dir)
            if not os.path.exists(self.opt.sam_mismatched_dir):
                os.makedirs(self.opt.sam_mismatched_dir)
            if rank == -1:
                io.imsave(self.opt.sam_matched_dir + self.opt.name + "_" + self.opt.name_finer + '_s' + str(
                    self.opt.which_slice) + "_top" + str(self.opt.top_n) + "_" + str(index) + '_sam' + '_' +
                          query_path.split('/')[-1], sam)
                io.imsave(self.opt.sam_matched_dir + self.opt.name + "_" + self.opt.name_finer + '_s' + str(
                    self.opt.which_slice) + "_top" + str(self.opt.top_n) + "_" + str(index) + '_heat' + '_' +
                          query_path.split('/')[-1], heatmap)
                io.imsave(self.opt.sam_matched_dir + self.opt.name + "_" + self.opt.name_finer + '_s' + str(
                    self.opt.which_slice) + "_top" + str(self.opt.top_n) + "_" + str(index) + '_img' + '_' +
                          query_path.split('/')[-1], img)

            else:
                io.imsave(self.opt.sam_mismatched_dir + self.opt.name + "_" + self.opt.name_finer + '_s' + str(
                    self.opt.which_slice) + "_top" + str(self.opt.top_n) + "_" + str(index) + '_sam_' + str(
                    rank) + '_' + query_path.split('/')[
                              -1], sam)
                io.imsave(self.opt.sam_mismatched_dir + self.opt.name + "_" + self.opt.name_finer + '_s' + str(
                    self.opt.which_slice) + "_top" + str(self.opt.top_n) + "_" + str(index) + '_heat_' + str(
                    rank) + '_' + query_path.split('/')[
                              -1], heatmap)
                io.imsave(self.opt.sam_mismatched_dir + self.opt.name + "_" + self.opt.name_finer + '_s' + str(
                    self.opt.which_slice) + "_top" + str(self.opt.top_n) + "_" + str(index) + '_img_' + str(
                    rank) + '_' + query_path.split('/')[
                              -1], img)

    def find_sam_weight(self, query, db):
        mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        mean_cos_similarity = mean_cos(query.view(256, -1), db.view(256, -1)).mean(0)
        grad_map = torch.autograd.grad(mean_cos_similarity, query, create_graph=True)[0]
        weight = grad_map.sum(1).sum(1).view(256, 1, 1).expand([256, 64, 64])
        return weight

    def get_image_paths(self):
        return self.image_paths

    def save_features(self):
        with torch.no_grad():
            self.labels = ['query_image_%d' % self.DA]
            raw_encoded = self.netG.encode(self.real_A, self.DA)

            encoded = raw_encoded.view(-1)  # encoded_new1
            encoded_np = encoded.cpu().numpy()
            db_path = copy.deepcopy(self.image_paths[0])
            if not os.path.exists("./features/" + db_path.split('/')[-3]):
                os.makedirs("./features/" + db_path.split('/')[-3])
            print("./features/" + db_path.split('/')[-3] + '/' + db_path.split('/')[-1][:-4])
            np.savez("./features/" + db_path.split('/')[-3] + '/' + db_path.split('/')[-1][:-4], encoded_np, db_path)
            if self.use_two_stage:
                if not os.path.exists("./features_finer/" + db_path.split('/')[-3]):
                    os.makedirs("./features_finer/" + db_path.split('/')[-3])
                raw_encoded_finer = self.netG_finer.encode(self.real_A, self.DA)
                np.savez("./features_finer/" + db_path.split('/')[-3] + '/' + db_path.split('/')[-1][:-4],
                         raw_encoded_finer.view(-1).cpu().numpy(),
                         db_path)

    def find_query_features(self):
        with torch.no_grad():
            self.labels = ['query_image_%d' % self.DA]
            raw_encoded = self.netG.encode(self.real_A, self.DA)
            encoded = raw_encoded.view(-1)  # encoded_new1

            # image = copy.deepcopy(self.real_A)
            qr_path = copy.deepcopy(self.image_paths[0])
            if self.use_two_stage:
                raw_encoded_finer = self.netG_finer.encode(self.real_A, self.DA).view(-1)
            else:
                raw_encoded_finer = None
            pair = (encoded, qr_path, raw_encoded_finer)
            # if len(list) % 1 == 0:
            self.query_feature_list.append(pair)  # image and coder

    def find_dist_mat(self):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        dist_mat = []
        if self.opt.only_for_finer:
            self.dirs = sorted(glob.glob("./features_finer/" + self.opt.dataroot.split('/')[3] + "/*"))
        else:
            self.dirs = sorted(glob.glob("./features/" + self.opt.dataroot.split('/')[3] + "/*"))
        for i, name in enumerate(self.dirs):
            print(i, name)
            feature_path = np.load(name)
            self.robotcar_database.append(torch.from_numpy(feature_path['arr_0']).view(256, 64, 64))
            dist_mat_row = []
            for j, query_feat in enumerate(self.query_feature_list):
                if self.opt.mean_cos:
                    dist = 1 - mean_cos(query_feat[0].view(256, -1),
                                        torch.from_numpy(feature_path['arr_0']).cuda().view(256, -1)).mean()
                else:
                    dist = 1 - cos(query_feat[0],
                                   torch.from_numpy(
                                       feature_path['arr_0']).cuda()) * 1  # + L2loss(query_encoded,item[1])*0
                dist_mat_row.append(dist.cpu().numpy().tolist())
            dist_mat.append(dist_mat_row)
        if self.opt.mean_cos:
            mean_cos = "meancos"
        else:
            mean_cos = "plaincos"
        if self.opt.meancos_finer:
            mean_cos_finer = "meancosfiner"
        else:
            mean_cos_finer = "plaincosfiner"
        np.savez(
            "dist_mat_cos_" + self.opt.dataroot.split('/')[3] + "_env" + str(self.opt.test_condition) + '_' + mean_cos
            + '_' + mean_cos_finer,
            np.array(dist_mat))
        self.dist_mat_torch = torch.from_numpy(np.array(dist_mat)).cuda()

    def load_dist_mat(self):
        if self.opt.mean_cos:
            mean_cos = "meancos"
        else:
            mean_cos = "plaincos"
        if self.opt.meancos_finer:
            mean_cos_finer = "meancosfiner"
        else:
            mean_cos_finer = "plaincosfiner"
        if self.opt.only_for_finer:
            self.dirs = sorted(glob.glob("./features_finer/" + self.opt.dataroot.split('/')[3] + "/*"))
        else:
            self.dirs = sorted(glob.glob("./features/" + self.opt.dataroot.split('/')[3] + "/*"))
        for i, name in enumerate(self.dirs):
            print(i, name)
            self.robotcar_database.append(torch.from_numpy(np.load(name)['arr_0']).view(256, 64, 64))
        self.dist_mat_torch = torch.from_numpy(
            np.load("dist_mat_cos_" + self.opt.dataroot.split('/')[3] + "_env" + str(
                self.opt.test_condition) + '_' + mean_cos
                    + '_' + mean_cos_finer + ".npz")[
                'arr_0']).cuda()

    def find_retrieval(self):
        query_path = []
        retrieved_path = []
        if self.use_two_stage:
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
            _, least_dist_index_topN = torch.sort(self.dist_mat_torch, 0)
            least_dist_index_topN = least_dist_index_topN.transpose(0, 1)[:, :self.top_n]
            self.dirs_finer = sorted(glob.glob("./features_finer/" + self.opt.dataroot.split('/')[3] + "/*"))
            for query_index, top_n_index in enumerate(least_dist_index_topN):
                query_feature = self.query_feature_list[query_index][2]
                least_value = 1000000
                path = None
                for _index in top_n_index:
                    if self.opt.mean_cos:
                        dist = 1 - mean_cos(query_feature.view(256, -1),
                                            torch.from_numpy(
                                                np.load(self.dirs_finer[_index.cpu().numpy()])['arr_0']).cuda().view(
                                                256, -1)).mean()
                    else:
                        dist = 1 - cos(query_feature,
                                       torch.from_numpy(
                                           np.load(self.dirs_finer[_index.cpu().numpy()])['arr_0']).cuda()) * 1
                    if dist < least_value:
                        least_value = dist
                        path = self.dirs_finer[_index.cpu().numpy()]
                retrieved_path.append(path)
                query_path.append(self.query_feature_list[query_index][1])
            print("query_path: ", query_path)
            print("retrieved_path: ", retrieved_path)
        else:
            if not self.opt.save_sam_visualization:
                least_dist_index = torch.argmin(self.dist_mat_torch, 0).cpu().numpy()
                for i in list(range(least_dist_index.size)):
                    query_path.append(self.query_feature_list[i][1])
                    retrieved_path.append(self.dirs[least_dist_index[i]])
            else:
                _, least_dist_index_topN = torch.sort(self.dist_mat_torch, 0)
                least_dist_index = least_dist_index_topN.transpose(0, 1)[:, :100].cpu().numpy()
                for i in list(range(least_dist_index.size)):
                    query_path.append(self.query_feature_list[i][1])
                    retrieved_path.append(self.dirs[least_dist_index[i][0]])
                    if i % 10 == 0:
                        self.find_grad_sam(self.query_feature_list[i][0], self.query_feature_list[i][1],
                                           self.robotcar_database[least_dist_index[i][0]], i)
                        self.find_grad_sam(self.robotcar_database[least_dist_index[i][0]],
                                           self.opt.dataroot + "test00/" + self.dirs[least_dist_index[i][0]].split('/')[
                                                                               -1][
                                                                           :-4] + ".jpg", self.query_feature_list[i][0],
                                           i)
                        self.find_grad_sam(self.query_feature_list[i][0], self.query_feature_list[i][1],
                                           self.robotcar_database[least_dist_index[i][99]], i, 100)
                        self.find_grad_sam(self.robotcar_database[least_dist_index[i][99]],
                                           self.opt.dataroot + "test00/" + self.dirs[least_dist_index[i][0]].split('/')[
                                                                               -1][
                                                                           :-4] + ".jpg",
                                           self.query_feature_list[i][99], i,
                                           100)

            print("query_path: ", query_path)
            print("retrieved_path: ", retrieved_path)
        return query_path, retrieved_path

    def backward_D_basic(self, real, fake, domain):
        # Real
        pred_real = self.netD.forward(real, domain)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        # D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.real_B, fake_B, self.DB)
        # D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.real_A, fake_A, self.DA)

    def backward_G(self):
        encoded_A = self.netG.encode(self.real_A, self.DA)
        encoded_B = self.netG.encode(self.real_B, self.DB)

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG.decode(encoded_A, self.DB)
        pred_fake = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        self.fake_A = self.netG.decode(encoded_B, self.DA)
        pred_fake = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        rec_encoded_A = self.netG.encode(self.fake_B, self.DB)
        self.rec_A = self.netG.decode(rec_encoded_A, self.DA)
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A)
        # Backward cycle loss
        rec_encoded_B = self.netG.encode(self.fake_A, self.DA)
        self.rec_B = self.netG.decode(rec_encoded_B, self.DB)
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B)

        if self.hard_negative:
            neg_B_features = self.netG.encode(self.neg_B, self.DB)

        if self.use_realAB_as_negative:
            if self.hard_negative:
                least_index = torch.argmin(torch.Tensor(
                    [self.L2loss(encoded_A.view(-1), neg_B.view(-1).detach()) for neg_B in neg_B_features])).item()
                dnA = self.L2loss(encoded_A.view(-1), neg_B_features[least_index].view(-1).detach())
            else:
                dnA = self.L2loss(encoded_A.view(-1), encoded_B.view(-1).detach())
        else:
            if self.hard_negative:
                least_index = torch.argmin(torch.Tensor(
                    [self.L2loss(rec_encoded_A.view(-1), neg_B.view(-1).detach()) for neg_B in neg_B_features])).item()
                dnA = self.L2loss(rec_encoded_A.view(-1), neg_B_features[least_index].view(-1).detach())
            else:
                dnA = self.L2loss(rec_encoded_A.view(-1), encoded_B.view(-1).detach())
        dpA = self.L2loss(rec_encoded_A.view(-1), encoded_A.view(
            -1).detach())

        if self.opt.mean_cos:
            cospA = 1 - self.mean_cos(rec_encoded_A.view(256, -1), encoded_A.view(256, -1)).mean(0)
        else:
            cospA = 1 - self.cos(rec_encoded_A.view(-1), encoded_A.view(-1))
        if self.lambda_triplet > 0:

            self.loss_triplet[self.DA] = torch.max(torch.cuda.FloatTensor([0.0]),
                                                   1 - dnA /
                                                   (
                                                           self.margin * torch.exp(
                                                       self.adapt * (-dpA))  # dnA/dpA#s#self.margin
                                                           + dpA))
            if self.hard_negative:
                self.loss_triplet[self.DA] = 2 * self.loss_triplet[self.DA]

        self.feature_distance[self.DA] = dpA
        self.feature_cos[self.DA] = cospA

        if not self.hard_negative:
            if self.use_realAB_as_negative:
                dnB = self.L2loss(encoded_B.view(-1), encoded_A.view(-1).detach())
            else:
                dnB = self.L2loss(rec_encoded_B.view(-1), encoded_A.view(-1).detach())
        dpB = self.L2loss(rec_encoded_B.view(-1), encoded_B.view(
            -1).detach())
        if self.opt.mean_cos:
            cospB = 1 - self.mean_cos(rec_encoded_B.view(256, -1), encoded_B.view(256, -1)).mean(0)
        else:
            cospB = 1 - self.cos(rec_encoded_B.view(-1), encoded_B.view(-1))
        if self.lambda_triplet > 0 and not self.hard_negative:
            self.loss_triplet[self.DB] = torch.max(torch.cuda.FloatTensor([0.0]),
                                                   1 - dnB / (self.margin * torch.exp(self.adapt * (-dpB)) + dpB))
        if self.hard_negative:
            self.loss_triplet[self.DB] = 0

        self.feature_distance[self.DB] = dpB
        self.feature_cos[self.DB] = cospB
        if self.lambda_latent > 0:
            if self.train_using_cos:
                loss_latent_A = cospA
                loss_latent_B = cospB
                if self.use_cos_latent_with_L2:
                    loss_latent_A += dpA
                    loss_latent_B += dpB
            else:
                loss_latent_A = dpA
                loss_latent_B = dpB
        else:
            loss_latent_A, loss_latent_B = 0, 0

        if self.lambda_sam > 0:
            self.loss_sam[self.DA] = self.L2loss(F.relu(torch.mul(encoded_A.view(256, 64, 64),
                                                                  self.find_sam_weight(encoded_A.view(256, 64, 64),
                                                                                       rec_encoded_A.view(256, 64, 64))).sum(dim=0)).view(-1),
                                                 F.relu(torch.mul(rec_encoded_A.view(256, 64, 64),
                                                                  self.find_sam_weight(rec_encoded_A.view(256, 64, 64),
                                                                                       encoded_A.view(256, 64, 64))).sum(dim=0)).view(-1).detach())
            self.loss_sam[self.DB] = self.L2loss(F.relu(torch.mul(encoded_B.view(256, 64, 64),
                                                                  self.find_sam_weight(encoded_B.view(256, 64, 64),
                                                                                       rec_encoded_B.view(256, 64, 64))).sum(dim=0)).view(-1),
                                                 F.relu(torch.mul(rec_encoded_B.view(256, 64, 64),
                                                                  self.find_sam_weight(rec_encoded_B.view(256, 64, 64),
                                                                                       encoded_B.view(256, 64, 64))).sum(dim=0)).view(-1).detach())

            if self.lambda_sam_triplet > 0:
                dp_samA = self.loss_sam[self.DA].cuda()
                dp_samB = self.loss_sam[self.DB].cuda()
                if self.use_realAB_as_negative:
                    if not self.hard_negative:
                        dn_samA = self.L2loss(F.relu(torch.mul(encoded_A.view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   encoded_A.view(256, 64, 64),
                                                                    encoded_B.view(256, 64, 64))).sum(dim=0)).view(-1), F.relu(torch.mul(encoded_B.view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   encoded_B.view(256, 64, 64),
                                                                   encoded_A.view(256, 64, 64))).sum(dim=0)).view(-1).detach())
                    if self.hard_negative:
                        least_index = torch.argmin(torch.Tensor(
                            [self.L2loss(F.relu(torch.mul(neg_B.view(256, 64, 64),
                                                          self.find_sam_weight(neg_B.view(256, 64, 64),
                                                                               encoded_A.view(256, 64,64))).sum(dim=0)).view(-1),
                                         F.relu(torch.mul(encoded_A.view(256, 64, 64),
                                                          self.find_sam_weight(
                                                              encoded_A.view(256, 64, 64),
                                                              neg_B.view(256, 64,64))).sum(dim=0)).view(-1).detach()) for neg_B in neg_B_features])).item()
                        dn_samB = self.L2loss(F.relu(torch.mul(neg_B_features[least_index].view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   neg_B_features[least_index].view(256, 64, 64),
                                                                   encoded_A.view(256, 64,64))).sum(dim=0)).view(-1),
                                              F.relu(torch.mul(encoded_A.view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   encoded_A.view(256, 64, 64),
                                                                   neg_B_features[least_index].view(256, 64,64))).sum(dim=0)).view(-1).detach())
                    else:
                        dn_samB = self.L2loss(F.relu(torch.mul(encoded_B.view(256, 64, 64),
                                                               self.find_sam_weight(encoded_B.view(256, 64, 64),encoded_A.view(256, 64,64))).sum(dim=0)).view(-1),
                                              F.relu(torch.mul(encoded_A.view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   encoded_A.view(256, 64, 64),
                                                                   encoded_B.view(256, 64,64))).sum(dim=0)).view(-1).detach())
                else:
                    if not self.hard_negative:
                        dn_samA = self.L2loss(F.relu(torch.mul(encoded_A.view(256, 64, 64),
                                                               self.find_sam_weight(encoded_A.view(256, 64, 64),
                                                                                    rec_encoded_B.view(256, 64,64))).sum(dim=0)).view(-1),

                                              F.relu(torch.mul(rec_encoded_B.view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   rec_encoded_B.view(256, 64, 64),
                                                                   encoded_A.view(256, 64,64))).sum(dim=0)).view(-1).detach())
                    if self.hard_negative:
                        least_index = torch.argmin(torch.Tensor(
                            [self.L2loss(F.relu(torch.mul(neg_B.view(256, 64, 64),
                                                          self.find_sam_weight(neg_B.view(256, 64, 64),
                                                                               rec_encoded_A.view(256, 64,64))).sum(dim=0)).view(-1),
                                         F.relu(torch.mul(rec_encoded_A.view(256, 64, 64),
                                                          self.find_sam_weight(
                                                              rec_encoded_A.view(256, 64, 64),
                                                              neg_B.view(256, 64,64))).sum(dim=0)).view(-1).detach()) for neg_B in neg_B_features])).item()
                        dn_samB = self.L2loss(F.relu(torch.mul(neg_B_features[least_index].view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   neg_B_features[least_index].view(256, 64, 64),
                                                                   rec_encoded_A.view(256, 64,64))).sum(dim=0)).view(-1),
                                              F.relu(torch.mul(rec_encoded_A.view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   rec_encoded_A.view(256, 64, 64),
                                                                   neg_B_features[least_index].view(256, 64, 64))).sum(dim=0)).view(-1).detach())
                    else:
                        dn_samB = self.L2loss(F.relu(torch.mul(encoded_B.view(256, 64, 64),
                                                               self.find_sam_weight(encoded_B.view(256, 64, 64),
                                                                                    rec_encoded_A.view(256, 64,64))).sum(dim=0)).view(-1),
                                              F.relu(torch.mul(rec_encoded_A.view(256, 64, 64),
                                                               self.find_sam_weight(
                                                                   rec_encoded_A.view(256, 64, 64),
                                                                   encoded_B.view(256, 64,64))).sum(dim=0)).view(-1).detach())
                if not self.hard_negative:
                    dn_samA = dn_samA.cuda()
                    self.loss_sam_triplet[self.DA] = torch.max(torch.cuda.FloatTensor([0.0]),
                                                               1 - dn_samA /
                                                               (self.margin_sam_triplet * torch.exp(self.adapt_sam_triplet * (-dp_samA))
                                                                       + dp_samA))
                dn_samB = dn_samB.cuda()
                self.loss_sam_triplet[self.DB] = torch.max(torch.cuda.FloatTensor([0.0]),
                                                           1 - dn_samB /
                                                           (self.margin_sam_triplet * torch.exp(self.adapt_sam_triplet * (-dp_samB))  + dp_samB))
                if self.hard_negative:
                    self.loss_sam_triplet[self.DA] = 0
                    self.loss_sam_triplet[self.DB] = 2 * self.loss_sam_triplet[self.DB]

        # combined loss

        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc + \
                 (self.loss_triplet[self.DA] + self.loss_triplet[self.DB]) * self.lambda_triplet + \
                 (self.loss_sam[self.DA] + self.loss_sam[self.DB]) * self.lambda_sam + \
                 (self.loss_sam_triplet[self.DA] + self.loss_sam_triplet[self.DB]) * self.lambda_sam_triplet + \
                 (loss_latent_A + loss_latent_B) * self.lambda_latent
        loss_G.backward()

    def optimize_parameters(self):
        # G_A and G_B
        self.netG.zero_grads(self.DA, self.DB)
        self.backward_G()
        self.netG.step_grads(self.DA, self.DB)
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses, feat_losses, feat_dist, cos_dist, sam_losses, loss_sam_triplet = \
            extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle), \
            extract(self.loss_triplet), extract(self.feature_distance), extract(self.feature_cos), extract(self.loss_sam), extract(self.loss_sam_triplet)

        return OrderedDict(
            [('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses), ('Feat', feat_losses), ('Feat_dist', feat_dist),
             ('Cosine_dist', cos_dist), ('SAM', list(map(lambda x: x * 100000000, sam_losses))),
             ('SAM_triplet_feat', loss_sam_triplet)])

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        images = [util().tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def get_domain(self):
        return self.DA

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)

        if self.opt.lambda_latent > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_latent = self.opt.lambda_latent * decay_frac
            print("latent: ", self.lambda_latent)

        if self.opt.lambda_triplet > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_triplet = self.opt.lambda_triplet * decay_frac
            print("triplet_feature: ", self.lambda_triplet)

        if self.opt.lambda_sam > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_sam = self.opt.lambda_sam * decay_frac
            print("SAM: ", self.lambda_sam)
