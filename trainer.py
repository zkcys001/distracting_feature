import time
from argparse import ArgumentParser

import torch.optim as optim
import torch.utils.data
from data.load_data import load_data,Dataset
from data import preprocess
from model.model_b3_p import Reab3p16
from rl.ddpg import *
from rl.help_function import *
from rl.qlearning import *
import utils
from tensorboard import TensorBoard

code = ['shape', 'line', "color", 'number', 'position', 'size',
        'type', 'progression', "xor", "or", 'and', 'consistent_union']
logger=utils.get_logger()


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

def adjust_learning_rate(optimizer, epoch, lr_steps,n):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.2

    if n>1:
        for param_group in optimizer.module.param_groups:
            param_group['lr'] = decay * param_group['lr']
            print(("epoch %d : lr=%.5f") % (epoch,  param_group['lr']))
            if epoch>15:
                param_group['momentum'] = 0.9
                param_group['weight_decay'] = decay * param_group['lr']
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']
            param_group['weight_decay'] = decay * param_group['lr']
            print(("epoch %d : lr=%.5f") % (epoch, param_group['lr']))
            if epoch>15:
                param_group['momentum'] = 0.9
def main(args):

    # Step 1: init data folders
    '''if os.path.exists('save_state/'+args.regime+'/normalization_stats.pkl'):
        print('Loading normalization stats')
        x_mean, x_sd = misc.load_file('save_state/'+args.regime+'/normalization_stats.pkl')
    else:
        x_mean, x_sd = preprocess.save_normalization_stats(args.regime)
        print('x_mean: %.3f, x_sd: %.3f' % (x_mean, x_sd))'''
    data_dir =args.datapath
    data_files = []
    for x in os.listdir(data_dir):
        for y in os.listdir(data_dir + x):
            data_files.append(data_dir + x + "/" + y)
    test_files = [data_file for data_file in data_files if
                  'val' in data_file and 'npz' in data_file]

    train_files = [data_file for data_file in data_files if 'train' in data_file and 'npz' in data_file]


    print("train_num:", len(train_files), "test_num:", len(test_files))

    train_loader = torch.utils.data.DataLoader(Dataset(args,train_files), batch_size=args.batch_size, shuffle=True,num_workers=args.numwork)#
    test_loader = torch.utils.data.DataLoader(Dataset(args,test_files), batch_size=args.batch_size, num_workers=args.numwork)

    tb=TensorBoard(args.model_dir)

    # Step 2: init neural networks
    print("network is:",args.net)
    if args.net == 'Reab3p16':
        model = Reab3p16(args)

    if args.gpunum > 1:
        model = nn.DataParallel(model, device_ids=range(args.gpunum))

    weights_path = args.path_weight

    if os.path.exists(weights_path):
        pretrained_dict = torch.load(weights_path)
        model_dict = model.state_dict()
        pretrained_dict1 = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict1[k] = v
                #print(k)
        model_dict.update(pretrained_dict1)
        model.load_state_dict(model_dict)

        print('load weight: '+weights_path)


    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.mo, weight_decay=5e-4)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.gpunum>1:
        optimizer = nn.DataParallel(optimizer, device_ids=range(args.gpunum))

    iter_count = 1
    epoch_count = 1
    #iter_epoch=int(len(train_files) / args.batch_size)
    print(time.strftime('%H:%M:%S', time.localtime(time.time())), 'training')

    while True:
        since=time.time()

        with torch.no_grad():
            model.eval()
            accuracy_all = []

            for x, y, style,me in test_loader:
                x, y = Variable(x).cuda(), Variable(y).cuda()
                pred = model(x)
                pred = pred.data.max(1)[1]
                correct = pred.eq(y.data).cpu().numpy()
                accuracy = correct.sum() * 100.0 / len(y)

                accuracy_all.append(accuracy)

        accuracy_all = sum(accuracy_all) / len(accuracy_all)

        reward = accuracy_all * 100
        tb.scalar_summary("test_acc", reward, epoch_count)

        # np.expand_dims(, axis=0)



        time_elapsed = time.time() - since
        print('test epoch in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
        print('------------------------------------')
        print(('epoch:%d, acc:%.1f') % (epoch_count, accuracy_all))
        print('------------------------------------')

        model.train()
        iter_epoch = int(len(train_files) / args.batch_size)
        for x, y,style,me in train_loader:
            if x.shape[0]<10:
                print(x.shape[0])
                break
            x, y = Variable(x).cuda(), Variable(y).cuda()
            if args.gpunum > 1:
                optimizer.module.zero_grad()
            else:
                optimizer.zero_grad()
            pred = model(x)
            loss = F.nll_loss(pred, y,reduce=False)
            #train_loss=loss
            loss=loss.mean()
            loss.backward()
            if args.gpunum > 1:
                optimizer.module.step()
            else:
                optimizer.step()
            iter_count += 1
            pred = pred.data.max(1)[1]
            correct = pred.eq(y.data).cpu()
            accuracy_total = correct.sum() * 100.0 / len(y)
            if iter_count % 100 == 0:
                iter_c = iter_count % iter_epoch
                print(time.strftime('%H:%M:%S', time.localtime(time.time())),
                      ('train_epoch:%d,iter_count:%d/%d, loss:%.3f, acc:%.1f') % (
                      epoch_count, iter_c, iter_epoch, loss, accuracy_total))

                tb.scalar_summary("train_loss",loss,iter_count)


        #print(acc_part_train)
        if epoch_count %args.lr_step ==0:
            print("change lr")
            adjust_learning_rate(optimizer, epoch_count, args.lr_step,args.gpunum)
        time_elapsed = time.time() - since
        print('train epoch in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
        #acc_p=np.array([x[0]/x[1] for x in acc_part])
        #print(acc_p)


        epoch_count += 1
        if epoch_count%1==0:
            print("save!!!!!!!!!!!!!!!!")
            save_state(model.state_dict(), args.model_dir+"/epoch"+str(epoch_count))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--regime', dest='regime', type=str, default='raven')
    parser.add_argument('--dataset_size', dest='dataset_size', type=int, default=1, help='-1 for full dataset')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16*2)
    parser.add_argument('--lr_step', dest='lr_step', type=int, default=10)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4)
    parser.add_argument('--mo', dest='mo', type=float, default=0.8)
    parser.add_argument('--net', dest='net', type=str, default="Reab3p16")
    parser.add_argument('--optm', dest='optm', type=str, default='SGD')
    parser.add_argument('--gpunum', dest='gpunum', type=int, default=2)
    parser.add_argument('--numwork', dest='numwork', type=int, default=12)
    parser.add_argument('--restore', dest='restore', type=bool, default=False)
    parser.add_argument('--path_weight', dest='path_weight', type=str, default='save/neutral/rl.pt')
    parser.add_argument('--rl_style', dest='rl_style', type=str, default='ddpg')
    args = parser.parse_args()
    main(args)
