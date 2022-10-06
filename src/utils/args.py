import argparse

FRAMEWORKS = ['centralized', 'federated']
FW_TASKS = {
    'centralized': ['oracle', 'source_only', 'ftda', 'mcd', 'fda', 'fda_inv'],
    'federated': ['oracle', 'ftda', 'mcd', 'fda', 'fda_inv', 'ladd']
}
SOURCE_DATASETS = ['gta5']
TARGET_DATASETS = ['cityscapes', 'crosscity', 'mapillary']
POLICIES = ['poly', 'step', 'warmuppoly']
OPTIMIZERS = ['SGD', 'Adam', 'AdamW']
ALGORITHMS = ['FedAvg', 'PartialClusterAvg', 'TotalClusterAvg', 'FedAvg-bn']
SERVER_OPTS = ['SGD', 'Adam', 'AdaGrad', 'FedAvgm']
MCD_LOSSES = ['diff2d', 'symkl2d', 'symkl2d_size_avg']

CL_TYPE = {
    'mapillary': ['clustering'],
    'cityscapes': ['heterogeneous'],
}

MODELS = {
    'cityscapes': ['deeplabv3'],
    'crosscity': ['deeplabv3'],
    'mapillary': ['deeplabv3'],
}

CLUSTER_LAYERS = ["bn", "encoder", "decoder", "all", "bn+encoder", "bn+decoder", "all-stats"]


def str2tuple(tp=int):

    def convert(s):
        return tuple(tp(i) for i in s.split(','))

    return convert


def check_steps(args, sr=None, se=None, r=None, e=None, cl_per_round=None):
    if args.num_source_rounds is not sr and \
            args.num_source_epochs is not se and \
            args.num_rounds is not r and \
            args.num_epochs is not e and \
            args.clients_per_round is not cl_per_round:
        raise AssertionError


def check_steps_by_fw_task(args):
    if args.framework == 'centralized':
        if args.fw_task == 'oracle':
            check_steps(args, sr=None, se=None, r=None, e=not None, cl_per_round=None)
        elif args.fw_task == 'source_only':
            check_steps(args, sr=None, se=not None, r=None, e=None, cl_per_round=None)
        elif args.fw_task == 'ftda':
            check_steps(args, sr=None, se=not None, r=None, e=not None, cl_per_round=None)
        elif args.fw_task == 'mcd':
            check_steps(args, sr=None, se=None, r=None, e=not None, cl_per_round=None)
    elif args.framework == 'federated':
        if args.fw_task == 'oracle':
            check_steps(args, sr=None, se=None, r=not None, e=not None, cl_per_round=not None)
        elif args.fw_task == 'ftda':
            check_steps(args, sr=None, se=not None, r=not None, e=not None, cl_per_round=not None)
        elif args.fw_task == 'mcd':
            check_steps(args, sr=None, se=None, r=not None, e=not None, cl_per_round=not None)


def check_args(args):
    assert args.framework in FRAMEWORKS
    if args.target_dataset != 'crosscity':
        assert args.clients_type in CL_TYPE[args.target_dataset]
    assert args.model in MODELS[args.target_dataset]
    assert args.fw_task in FW_TASKS[args.framework]

    check_steps_by_fw_task(args)

    if args.target_dataset in ('crosscity',):
        assert FW_TASKS[args.framework] not in ('oracle',)

    if args.target_dataset != 'cityscapes':
        args.double_dataset = None
        args.quadruple_dataset = None

    if args.framework == 'centralized':
        args.server_opt = None
        args.num_rounds = None
        args.clients_per_round = None

    if args.server_opt is None:
        args.server_lr = None
        args.server_momentum = None

    if args.optimizer != 'SGD':
        args.momentum = None
        args.nesterov = None

    if args.lr_policy is None:
        args.lr_power = None
        args.lr_decay_step = None
        args.lr_decay_factor = None
        args.warmup_iters = None
    elif args.lr_policy == 'poly':
        args.lr_decay_step = None
        args.lr_decay_factor = None
        args.warmup_iters = None
    elif args.lr_policy == 'step':
        args.lr_power = None
        args.warmup_iters = None
    elif args.lr_policy == 'warmuppoly':
        args.lr_power = None
        args.lr_decay_step = None
        args.lr_decay_factor = None


def modify_command_options(args):
    if args.target_dataset == 'cityscapes':
        args.num_classes = 19
    elif args.target_dataset == 'crosscity':
        args.num_classes = 13
    elif args.target_dataset == 'mapillary':
        args.num_classes = 19

    args.total_batch_size = len(args.device_ids) * args.batch_size
    args.device_ids = [int(device_id) for device_id in args.device_ids]
    args.n_devices = len(args.device_ids)

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # ||| Framework alternatives |||
    parser.add_argument('--framework', type=str, choices=FRAMEWORKS, required=True, help='Type of framework')
    parser.add_argument('--fw_task', type=str, required=True, help='Type of framework task')

    # ||| Distributed and GPU options |||
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device_ids', default=[0], nargs='+', help='GPU ids for multigpu mode')

    # ||| Reproducibility Options |||
    parser.add_argument('--random_seed', type=int, required=False, help='random seed')

    # ||| Dataset Options |||
    parser.add_argument('--source_dataset', type=str, choices=SOURCE_DATASETS, required=False, default='',
                        help='Name of the source dataset')
    parser.add_argument('--target_dataset', type=str, choices=TARGET_DATASETS, required=True,
                        help='Name of the target dataset')
    parser.add_argument('--double_dataset', action='store_true', default=False,
                        help='Option to double the size of the datasets of the clients')
    parser.add_argument('--quadruple_dataset', action='store_true', default=False,
                        help='Option to quadruplicate the size of the datasets of the clients')
    parser.add_argument('--cv2', action='store_true', default=False, help='use cv2 transforms if set')
    parser.add_argument('--random_flip', action='store_true', default=False, help='use random_flip on SOURCE dataset')
    parser.add_argument('--color_jitter', action='store_true', default=False, help='use color_jitter on SOURCE dataset')
    parser.add_argument('--gaussian_blur', action='store_true', default=False,
                        help='use gaussian_blur on SOURCE dataset')
    parser.add_argument('--clients_type', type=str, required=True, help='Clients distribution type')
    parser.add_argument('--centr_fda_ft_uda', action='store_true', default=False,
                        help='Use only one client in fda, to test pseudolabels vs aggregation')
    parser.add_argument('--disable_batch_norm', action='store_true', default=False, help='disable batch norm')
    parser.add_argument('--batch_norm_round_0', action='store_true', default=False,
                        help='use only batch norm at round 0')
    parser.add_argument('--hp_filtered', action='store_true', default=False, help='Use high pass filtered images')

    # ||| MCD |||
    parser.add_argument('--discr_loss', type=str, choices=MCD_LOSSES, required=False, default='diff2d',
                        help='discrepancy loss for MCD')
    parser.add_argument('--repeat_phase_3', type=int, default=4, help='how many times to repeat phase 3 per epoch')
    parser.add_argument("--discr_loss_multiplier", type=int, default=1, help='constant multiplied to discrepancy loss')

    # ||| FDA |||
    parser.add_argument('--pretrain', action='store_true', default=False, help='LADD Pretrain only if set')
    parser.add_argument('--n_images_per_style', type=int, default=1000,
                        help='number of images to extract style (avg is performed)')
    parser.add_argument('--fda_L', type=float, default=0.01, help='to control size of amplitude window')
    parser.add_argument('--fda_b', type=int, default=None, help='if != None it is used instead of fda_L:'
                                                                'b == 0 --> 1x1, b == 1 --> 3x3, b == 2 --> 5x5, ...')
    parser.add_argument('--fda_size', type=str2tuple(int), default='1024,512',
                        help='size (W,H) to which resize images before style transfer')
    parser.add_argument('--train_source_round_interval', type=int, default=None,
                        help='Train on source every interval rounds, leave to None to do single initial pretraining')
    parser.add_argument('--lr_factor_server_retrain', type=float, default=1.,
                        help='learning rate mult factor when retraining at server')
    parser.add_argument('--num_source_epochs_factor_retrain', type=float, default=1.,
                        help='num epochs mult factor when retraining at server')
    parser.add_argument('--num_source_steps_retrain', type=int, default=None,
                        help='num of retraining steps server-side, leave to None to use epochs instead')
    parser.add_argument('--source_style_to_source', action='store_true', default=False,
                        help='Apply source style to source images')
    parser.add_argument('--style_only_train', action='store_true', default=False,
                        help='Apply source style only to train images')
    parser.add_argument('--fda_loss', type=str, default='selftrain',
                        choices=['selftrain', 'advent', 'maxsquares', 'selftrainentropy', 'lovasz_entropy_joint',
                                 'lovasz_entropy_div', 'selftrain_div'], help='UDA loss')
    parser.add_argument('--lambda_selftrain', type=float, default=1,
                        help='weight of the selftrain loss when fda_loss in [lovasz_entropy_div, selftrain_div]')
    parser.add_argument('--lambda_entropy', type=float, default=0.005,
                        help='weight of the entropy loss when fda_loss in [lovasz_entropy_div, selftrain_div]')
    parser.add_argument('--teacher_step', type=int, default=1,
                        help='step size (# rounds) to update teacher model for pseudo-labeling '
                             '(leave to <=0 to disable, i.e. pseudo label is computed on current model prediction)')
    parser.add_argument('--teacher_kd_step', type=int, default=-1, help='as teacher step but for kd loss')
    parser.add_argument('--teacher_upd_step', action='store_true', default=False,
                        help='teacher_step refers to steps and not to rounds if this is true (useful for centr only)')
    parser.add_argument('--teacher_kd_upd_step', action='store_true', default=False,
                        help='teacher_kd_step refers to steps and not to rounds if this is true'
                             '(useful for centr only)')
    parser.add_argument('--teacher_kd_mult_factor', type=float, default=-1.0,
                        help='if != -1, multiply kd loss by this factor every teacher_kd_mult_step epochs')
    parser.add_argument('--teacher_kd_mult_step', type=int, default=5, help='step for teacher_kd_mult_factor')
    parser.add_argument('--ignore255_kdloss', action='store_true', default=False,
                        help="ignore background pixels for knowledge distillation")
    parser.add_argument('--count_classes_teacher_step', type=int, default=-1,
                        help='step interval for updating class prediction stats relative to each teacher update')
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature for softmax prob of the sampler')
    parser.add_argument('--weights_lovasz', action='store_true', default=False,
                        help="apply weights to lovasz (useful if count_classes_teacher_step != -1)")
    parser.add_argument('--multiple_styles', action='store_true', default=False,
                        help='Keep multiple styles for each client when True')
    parser.add_argument('--lr_fed', type=float, default=0.005, help='learning rate during federated uda')
    parser.add_argument('--stop_epoch_at_step', type=int, default=-1,
                        help='stop the epoch before the end, used in centr to simulate the same teacher step of'
                             'federated version')
    parser.add_argument('--freezing', type=str, default=None,
                        choices=[None, 'es_dc', 'ds_ec', 'all_but_one_server'],
                        help='after pretrain, when None it is performed standard selftrain, with es_dc '
                             'server trains the encoder while client trains the decoder, and ds_ec vice-versa')
    parser.add_argument('--distinct_batch_norm', action='store_true', default=False)
    parser.add_argument('--only_update_bn_server', action='store_true', default=False)
    parser.add_argument('--silobn', action='store_true', default=False,
                        help="w&b are aggregated for clients, running_mean and running_var are distinct")
    parser.add_argument('--n_clients_per_city', action='store_true', default=False,
                        help="if true select n client per city (if dataset == crosscity")
    parser.add_argument('--alpha_kd', type=float, default=0.5, help='alpha Knowledge Distillation Loss')
    parser.add_argument('--lambda_kd', type=float, default=0., help='lambda Knowledge Distillation Loss')

    parser.add_argument('--swa_start', type=int, default=-1, help='start round for SWA (-1 == do not use SWA)')
    parser.add_argument('--swa_c', type=int, default=1,
                        help='SWA model collection frequency/cycle length in rounds (default: 1)')
    parser.add_argument('--swa_lr', type=float, default=1e-4, help='SWA learning rate (alpha2)')

    parser.add_argument('--swa_teacher_start', type=int, default=-1,
                        help='start round for SWA teacher (-1 == do not use SWA)')
    parser.add_argument('--swa_teacher_c', type=int, default=1,
                        help='SWA teacher model collection frequency/cycle length in rounds (default: 1)')

    # ||| Style Clustering and Aggregation Options |||
    parser.add_argument('--train_with_global_model', action='store_true', default=False,
                        help="train with global model, instead of selecting a cluster model for each client")
    parser.add_argument('--style_clusters_dir', type=str, default=None,
                        help='path to the dir where fda clusters are saved, if a cluster for the same run is already '
                             'present, load the clusters from the file')
    parser.add_argument('--cluster_layers', type=str, default="bn", choices=CLUSTER_LAYERS,
                        help="layer to aggregate in LADD")
    parser.add_argument('--global_aggregation_round', type=int, default=None,
                        help="perform global aggregation each global_aggregation_round steps")
    parser.add_argument('--fedavg_bootstap', type=int, default=0,
                        help="number of initial FedAvg rounds before ClusterAvg and global aggregation")
    parser.add_argument('--test_only_with_global_model', action='store_true', default=False,
                        help="test with global model, instead of selecting a cluster model for each testing image")
    parser.add_argument('--save_cluster_models', action='store_true', default=False,
                        help="save the model of each cluster, each 100 rounds.., for ablation")
    parser.add_argument('--force_k', type=int, default=0, help="force the number of clusters to be k")

    # ||| Model Options |||
    parser.add_argument('--model', type=str, required=True, help='model type')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hnm or not')
    # ||| Federated Algorithm Options |||
    parser.add_argument('--server_opt', help='server optimizer', choices=SERVER_OPTS, required=False)
    parser.add_argument('--algorithm', type=str, default='FedAvg', choices=ALGORITHMS,
                        help='which federated algorithm to use')
    parser.add_argument('--server_lr', type=float, default=1, help='learning rate for server optimizers')
    parser.add_argument('--server_momentum', type=float, default=0, help='momentum for server optimizers')

    # ||| Training and Testing Options |||
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_source_rounds', type=int, help='number of source rounds')
    parser.add_argument('--num_epochs', type=int, help='number of epochs')
    parser.add_argument('--num_source_epochs', type=int, help='number of source epochs')

    parser.add_argument('--clients_per_round', type=int, default=-1, help='number of clients trained per round')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size when clients train on data (per GPU)')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size when clients test on data (per GPU)')
    parser.add_argument('--eval_interval', type=int, default=1, help='epoch/round interval for eval')
    parser.add_argument('--test_interval', type=int, default=1, help='epoch/round interval for test')
    parser.add_argument('--server_eval_interval', type=int, default=1,
                        help='epoch interval for eval on server training')
    parser.add_argument('--server_test_interval', type=int, default=1,
                        help='epoch interval for test on server training')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='Option to use mixed precision')
    parser.add_argument('--test_source', action='store_true', default=False,
                        help='Option to test on test source dataset')

    # ||| Optimizer |||
    parser.add_argument('--optimizer', type=str, default='SGD', choices=OPTIMIZERS, help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for the client optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--nesterov', action='store_true', default=False, help='Enables Nesterov momentum for SGD')

    # ||| Scheduler |||
    parser.add_argument('--lr_policy', type=str, choices=POLICIES, help='lr schedule policy')
    parser.add_argument('--lr_power', type=float, default=0.9, help='power for polyLR')
    parser.add_argument('--lr_decay_step', type=int, default=5000, help='decay step for stepLR')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='decay factor for stepLR')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='number of warmup iterations for WarmupPolyLR')

    # ||| Logging Options |||
    parser.add_argument('--name', type=str, default='Experiment', help='name of the experiment')
    parser.add_argument('--print_interval', type=int, default=10, help='print interval for the loss')
    parser.add_argument('--plot_interval', type=int, default=10, help='plot interval for the loss')
    parser.add_argument('--save_samples', type=int, default=0, help='How many samples pictures to save on cloud')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                        help='if you want wandb offline set to True, otherwise it uploads results on cloud')
    parser.add_argument('--wandb_entity', type=str, default='fl_polito_unipd', help='name of the wandb entity')

    # ||| Test and Checkpoint options |||
    parser.add_argument('--load', action='store_true', default=False, help='Whether to use pretrained or not')
    parser.add_argument('--wandb_id', type=str, required=False, help='wandb id to resume run')
    parser.add_argument('--load_FDA', action='store_true', default=False, help='Whether to use pretrained or not')
    parser.add_argument('--load_FDA_id', type=str, default=None, help='wandb id to resume run')
    parser.add_argument('--load_FDA_best', action='store_true', default=False,
                        help='load best pretrained server model')
    parser.add_argument('--load_yaml_config', type=str, default=None,
                        help='path to the yaml config downloaded from wandb')
    parser.add_argument('--fedprox', action='store_true', default=False, help="run fedprox")
    parser.add_argument('--fedprox_mu', type=float, default=0.01, help='FedProx proximal term')

    # ||| Other options |||
    parser.add_argument('--ignore_warnings', action='store_true', default=False, help='ignore all the warnings if set')
    parser.add_argument('--profiler_folder', type=str, help='profiler folder')
    parser.add_argument('--ignore_train_metrics', action='store_true', default=False,
                        help='not update train metrics if set')
    parser.add_argument('--save_clients_order', action='store_true', default=False,
                        help='save to file the order of clients per round, for debug')

    return parser
