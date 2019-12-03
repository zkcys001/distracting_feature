import argparse
from utils import get_logger

logger = get_logger()


arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')



net_arg.add_argument('--datapath', dest='datapath', type=str, default= "/home/lab/zkc/reason/process_data/reason_data/reason_data/RAVEN-10000/", help='')
net_arg.add_argument('--dataset_size', dest='dataset_size', type=int, default=1, help='')
net_arg.add_argument('--batch_size', dest='batch_size', type=int, default=16*2)
net_arg.add_argument('--lr_step', dest='lr_step', type=int, default=10)#100 for rl
net_arg.add_argument('--lr', dest='lr', type=float, default=1e-2)
net_arg.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4)
net_arg.add_argument('--mo', dest='mo', type=float, default=0.8)
net_arg.add_argument('--net', dest='net', type=str, default="Reab3p16")
net_arg.add_argument('--optm', dest='optm', type=str, default='SGD')
net_arg.add_argument('--gpunum', dest='gpunum', type=int, default=2)
net_arg.add_argument('--numwork', dest='numwork', type=int, default=10)
net_arg.add_argument('--type_loss', dest='type_loss', type=bool, default=False)
net_arg.add_argument('--random_seed', type=int, default=12345)
net_arg.add_argument('--rl', dest='rl', type=bool, default=False)
#Misc
#net_arg.add_argument('--load_path', dest='load_path', type=bool, default='save/neutral/rl.pt')
net_arg.add_argument('--log_dir', dest='log_dir', type=str, default='logs')
net_arg.add_argument('--regime', dest='regime', type=str, default='all')
net_arg.add_argument('--path_weight', dest='path_weight', type=str, default='')
net_arg.add_argument('--load_weight', dest='load_weight', type=str, default='/home/lab/zkc/reason/abstract_reason-pytorch_github/logs/all_2019-09-06_09-16-35image/epoch400')
net_arg.add_argument('--restore', dest='restore', type=bool, default=True)
net_arg.add_argument('--rl_style', dest='rl_style', type=str, default='ddpg')
net_arg.add_argument('--image_type', dest='image_type', type=str, default='image')

def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    if args.gpunum > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info("Unparsed args: "+unparsed)
    return args, unparsed
