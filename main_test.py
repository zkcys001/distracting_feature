import misc
import os
import torch.utils.data
import time
import math
from argparse import ArgumentParser
from model.model_resnet import resnet50
from model.model_wresnet import wresnet50
from model.model_b3_p import Reab3p16
from model.model_b3_plstm import b3_plstm
from model.model_b3_palstm import b3palstm
from model.model_b3_pa import b3pa
from model.model_rn_mlp import rn_mlp
from model.model_zkc import tmp
from model.model_plusMLP import WildRelationNet
from model.model_m_a import RN_ap
from model.model_r import RN_r
from model.model_pluspanda import RNap2
from model.model_esem import esemble
from model.model_nmn import nmn
from model.model_baseline_mlp import ReasonNet_p
from model.model_baseline_mlp16 import ReasonNet_p16
from model.model_baseline import ReasonNet
from model.model_a_mlp import ReasonNet_ap
from model.model_b3_p3 import b3p3
from model.model_multi3 import multi3
from model.model_split import b3_split
from rl.help_function import *
from rl.qlearning import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __getitem__(self, ind):
        data_file = self.data_files[ind]

        data = np.load(data_file.replace("/neutral", "/neutral_s"))
        x = data['shap_im'].reshape(16, 160, 160)

        # x = msk_abstract.post_data(x)
        y = data['target']
        # print( data['relation_structure_encoded'][0])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x = x.type(torch.float32)

        return x, y

    def __len__(self):
        return len(self.data_files)


def compute_data_files():
    '''Sort the data files in increasing order of complexity, then return the n least complex datapoints.'''
    data_files = []
    print('Loading structure metadata')
    structure_to_files = misc.load_file('save_state/neutral/structure_to_files.pkl')
    all_structure_strs = list(structure_to_files.keys())
    # The number of commas in the structure_str is used as a proxy for complexity
    i = 0
    all_structure_strs.sort(key=lambda x: x.count(','))
    '''
    for structure_str in all_structure_strs:
        data_i=structure_to_files[structure_str]
        if ( "SHAPE" in structure_str) and  len(data_i)>10000 and len(data_i)<20000:

            data_files.extend(data_i)
        else:
            continue
        print(structure_str, len(structure_to_files[structure_str]))


    structure_str = '(PROGRESSION,SHAPE,TYPE)'#(XOR,SHAPE,SIZE)
    data_files.extend(structure_to_files[structure_str])
    print(structure_str, len(structure_to_files[structure_str]))
    '''
    return all_structure_strs


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def save_state(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def adjust_learning_rate(optimizer, epoch, lr_steps, n):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.2

    if n > 1:
        for param_group in optimizer.module.param_groups:
            param_group['lr'] = decay * param_group['lr']
            print(("epoch %d : lr=%.5f") % (epoch, param_group['lr']))
            if epoch > 15:
                param_group['momentum'] = 0.9
                param_group['weight_decay'] = decay * param_group['lr']
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']
            param_group['weight_decay'] = decay * param_group['lr']
            print(("epoch %d : lr=%.5f") % (epoch, param_group['lr']))
            if epoch > 15:
                param_group['momentum'] = 0.9


def main(args):
    # Step 1: init data folders
    '''if os.path.exists('save_state/'+args.regime+'/normalization_stats.pkl'):
        print('Loading normalization stats')
        x_mean, x_sd = misc.load_file('save_state/'+args.regime+'/normalization_stats.pkl')
    else:
        x_mean, x_sd = preprocess.save_normalization_stats(args.regime)
        print('x_mean: %.3f, x_sd: %.3f' % (x_mean, x_sd))'''

    # Step 2: init neural networks
    print("network is:", args.net)
    if args.net == "resnet":
        model = resnet50(pretrained=True)
    elif args.net == 'wresnet':
        model = wresnet50(pretrained=True)
    elif args.net == "tmp":
        model = tmp()
    elif args.net == 'RN_mlp':
        model = WildRelationNet()
    elif args.net == 'ReasonNet':
        model = ReasonNet()
    elif args.net == 'ReaP':
        model = ReasonNet_p()
    elif args.net == 'Reap16':
        model = ReasonNet_p16()
    elif args.net == 'Reaap':
        model = ReasonNet_ap()
    elif args.net == 'RN_ap':
        model = RN_ap()
    elif args.net == 'RN_r':
        model = RN_r()
    elif args.net == 'esemble':
        model = esemble()
    elif args.net == 'RNap2':
        model = RNap2()
    elif args.net == 'rn_mlp':
        model = rn_mlp()
    elif args.net == 'Reab3p16':
        model = Reab3p16()
    elif args.net == 'b3pa':
        model = b3pa()
    elif args.net == "b3_plstm":
        model = b3_plstm()
    elif args.net == "b3_palstm":
        model = b3palstm()
    elif args.net == "nmn":
        model = nmn()
    elif args.net == "b3p3":
        model = b3p3()
    elif args.net == "multi3":
        model = multi3()
    elif args.net == "split":
        model = b3_split()
    if args.gpunum > 1:
        model = nn.DataParallel(model, device_ids=range(args.gpunum))
    if args.net != 'RN_r':
        model.apply(weights_init)
        print('weight initial')
    weights_path = args.path_weight
    if os.path.exists(weights_path) and args.restore:
        pretrained_dict = torch.load(weights_path)
        model_dict = model.state_dict()
        pretrained_dict1 = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict1[k] = v
                # print(k)
        model_dict.update(pretrained_dict1)
        model.load_state_dict(model_dict)
        # optimizer.load_state_dict(torch.load(optimizer_path))
        print('load weight')
    model.cuda()
    epoch_count = 1
    print(time.strftime('%H:%M:%S', time.localtime(time.time())), 'testing')

    print('Loading structure metadata')
    structure_to_files = misc.load_file('save_state/neutral/structure_to_files.pkl')

    all_structure_strs = list(structure_to_files.keys())
    # The number of commas in the structure_str is used as a proxy for complexity

    accuracy_all = []
    for structure_str in all_structure_strs:
        data_files = []
        data_i = structure_to_files[structure_str]
        if ("SHAPE" in structure_str) and len(
                data_i) > 10000:  # and len(data_i) < 20000:
            data_files.extend(data_i)
        else:
            continue
        test_files = [data_file for data_file in data_files if 'test' in data_file]

        test_loader = torch.utils.data.DataLoader(Dataset(test_files), batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.numwork)

        since = time.time()
        model.eval()
        accuracy_epoch = []
        for x, y in test_loader:
            x, y = Variable(x).cuda(), Variable(y).cuda()
            pred = model(x)

            pred = pred.data.max(1)[1]
            correct = pred.eq(y.data).cpu().sum().numpy()
            accuracy = correct * 100.0 / len(y)
            accuracy_epoch.append(accuracy)
            accuracy_all.append(accuracy)

        acc = sum(accuracy_epoch) / len(accuracy_epoch)

        print(('epoch:%d, acc:%.1f') % (epoch_count, acc), "test_num:", len(test_files), (structure_str))
        epoch_count += 1

    print(('epoch:%d, acc:%.1f') % (epoch_count, sum(accuracy_all) / len(accuracy_all)))


if __name__ == '__main__':
    '''
    parser = ArgumentParser()
    parser.add_argument('--regime', degst='regime', type=str, default='neutral')
    parser.add_argument('--dataset_size', dest='dataset_size', type=int, default=1, help='-1 for full dataset')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=160 )
    parser.add_argument('--lr_step', dest='lr_step', type=int, default=5)
    parser.add_argument('--lr', dest='lr', type=float, default=3e-2)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4)
    parser.add_argument('--mo', dest='mo', type=float, default=0.8)
    parser.add_argument('--net', dest='net', type=str, default='RN_e')
    parser.add_argument('--optm', dest='optm', type=str, default='SGD')
    parser.add_argument('--gpunum', dest='gpunum', type=int, default=1)
    parser.add_argument('--numwork', dest='numwork', type=int, default=6)
    args = parser.parse_args()
    main(args)
    '''
    parser = ArgumentParser()
    parser.add_argument('--regime', dest='regime', type=str, default='neutral')
    parser.add_argument('--dataset_size', dest='dataset_size', type=int, default=1, help='-1 for full dataset')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=104 * 2)
    parser.add_argument('--lr_step', dest='lr_step', type=int, default=8)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-2)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4)
    parser.add_argument('--mo', dest='mo', type=float, default=0.8)
    parser.add_argument('--net', dest='net', type=str, default="Reab3p16")  # Reab3p16 b3p3
    parser.add_argument('--optm', dest='optm', type=str, default='SGD')
    parser.add_argument('--gpunum', dest='gpunum', type=int, default=2)
    parser.add_argument('--numwork', dest='numwork', type=int, default=6)
    parser.add_argument('--restore', dest='restore', type=bool, default=True)
    parser.add_argument('--path_weight', dest='path_weight', type=str, default='save/neutral/rl.pt')
    args = parser.parse_args()

    main(args)

