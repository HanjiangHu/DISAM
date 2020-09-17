from .base_options import BaseOptions


class RobotcarTestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

        self.parser.add_argument('--which_epoch', required=True, type=int, help='which epoch to load for inference?')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc (determines name of folder to load from)')

        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run (if serial_test not enabled)')
        self.parser.add_argument('--serial_test', action='store_true', help='read each image once from folders in sequential order')

        self.parser.add_argument('--autoencode', action='store_true', help='translate images back into its own domain')
        self.parser.add_argument('--reconstruct', action='store_true', help='do reconstructions of images during testing')

        self.parser.add_argument('--show_matrix', action='store_true', help='visualize images in a matrix format as well')
        self.parser.add_argument('--which_slice', type=int, default=2,
                                 help='which slice of images to be test')
        self.parser.add_argument('--test_using_cos', action='store_true',
                                 help='which slice of images to be test')
        self.parser.add_argument('--test_after_pca', action='store_true',
                                 help='which slice of images to be test')
        self.parser.add_argument('--resize64', action='store_true',
                                 help='64*64 feature map')
        self.parser.add_argument('--test_condition', type=int, default=1,
                                 help='condition to be test, 1~9')
        self.parser.add_argument('--test_with_sequence', action='store_true',
                                 help='test_with_sequence')
        self.parser.add_argument('--sequence_interval', type=int, default=11,
                                 help='sequence_interval total')
        self.parser.add_argument('--load_dist_mat', action='store_true',
                                 help='which slice of images to be test')
        self.parser.add_argument('--use_two_stage', action='store_true',
                                 help='use two-stage strategy')
        self.parser.add_argument('--top_n', type=int, default=5,
                                 help='top n candidates for the finer retrieval')
        self.parser.add_argument('--which_epoch_finer', type=int, default=1200,
                                 help='for the finer generator')
        self.parser.add_argument('--name_finer', type=str, default='noFiner',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--meancos_finer', action='store_true',
                                 help=' mean cos only for siam related')
        self.parser.add_argument('--only_for_finer', action='store_true',
                                 help=' mean cos only for siam related')
        self.parser.add_argument('--save_sam_visualization', action='store_true',
                                 help='Whether to save the visualization of SAM of CMU images or not, only effective '
                                      'when --use_two_stage is NOT specified')
        self.parser.add_argument('--sam_matched_dir', type=str, default='./sam_matched_RobotCar/',
                                 help='Saves the matched SAM visualization here')
        self.parser.add_argument('--sam_mismatched_dir', type=str, default='./sam_mismatched_RobotCar/',
                                 help='Saves the mismatched SAM here')

