from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        self.parser.add_argument('--continue_train', action='store_true', help='Continue training')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='Which epoch to load if continuing training')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc '
                                                                            '(determines name of folder to load from)')

        self.parser.add_argument('--niter', required=True, type=int, help='# of epochs at starting learning rate, '
                                                                          'using 300 if fine-tune ')

        self.parser.add_argument('--niter_decay', required=True, type=int, help='# of epochs to linearly decay learning'
                                                                                'rate to zero, try 300 * n epochs')

        self.parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for ADAM')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of ADAM')

        self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight for cycle loss')
        self.parser.add_argument('--lambda_triplet', type=float, default=1.0, help='Weight for triplet loss, try 1.0')
        self.parser.add_argument('--lambda_latent', type=float, default=0.1, help='Weight for latent DIF loss, try 0.1')

        self.parser.add_argument('--lambda_sam', type=float, default=1000.0,
                                 help='Weight for SAM loss, try 1000.0')

        self.parser.add_argument('--lambda_sam_triplet', type=float, default=1.0,
                                 help='Weight for SAM triplet loss, try 1.0')

        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='Frequency of saving checkpoints'
                                                                                 ' at the end of epochs')
        self.parser.add_argument('--display_freq', type=int, default=100, help='Frequency of showing training results '
                                                                               'on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='Frequency of showing training results '
                                                                             'on console')

        self.parser.add_argument('--no_lsgan', action='store_true', help='Use vanilla discriminator in place of '
                                                                         'least-squares one')
        self.parser.add_argument('--pool_size', type=int, default=50, help='The size of image buffer that stores'
                                                                           'previously generated images')
        self.parser.add_argument('--train_using_cos', action='store_true',
                                 help='Use cosine or mean cosine for latent loss while training')
        self.parser.add_argument('--num_hard_neg', type=int, default=10,
                                 help='How many negative samples to choose the hardest one, try 10 for 12G memory')
        self.parser.add_argument('--hard_negative', action='store_true',
                                 help='Use the strategy of hard negative triplet loss')
        self.parser.add_argument('--use_realAB_as_negative', action='store_true',
                                 help='To use the real A and B as negative pairs, use translated A and real B otherwise')
        self.parser.add_argument('--use_cos_latent_with_L2', action='store_true',
                                 help='Use L2 metric along with cosine metric for latent loss, '
                                      'only effective if --train_using_cos specified')
