import time
from argparse import ArgumentParser

import torch.optim as optim
import torch.utils.data
from data.load_data import load_data,Dataset
from data import preprocess
from model.model_b3_p import Reab3p16
from model.model_plusMLP import WildRelationNet
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

    val_loader=load_data(args, "val")

    tb=TensorBoard(args.model_dir)

    # Step 2: init neural networks
    print("network is:",args.net)
    if args.net == 'Reab3p16':
        model = Reab3p16(args)
    elif args.net=='RN_mlp':
        model =WildRelationNet()
    if args.gpunum > 1:
        model = nn.DataParallel(model, device_ids=range(args.gpunum))

    weights_path = args.path_weight+"/"+args.load_weight

    if os.path.exists(weights_path) and args.restore:
        pretrained_dict = torch.load(weights_path)
        model_dict = model.state_dict()
        pretrained_dict1 = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict1[k] = v
                #print(k)
        model_dict.update(pretrained_dict1)
        model.load_state_dict(model_dict)

        print('load weight')

    style_raven={65:0, 129:1, 257:2, 66:3, 132:4, 36:5, 258:6, 136:7, 264:8, 72:9, 130:10
         , 260:11, 40:12, 34:13, 49:14, 18:15, 20:16, 24:17}
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.mo, weight_decay=5e-4)
   
    if args.gpunum>1:
        optimizer = nn.DataParallel(optimizer, device_ids=range(args.gpunum))

    iter_count = 1
    epoch_count = 1
    #iter_epoch=int(len(train_files) / args.batch_size)
    print(time.strftime('%H:%M:%S', time.localtime(time.time())), 'training')
    style_raven_len = len(style_raven)
    
    if args.rl_style=="dqn":
        dqn = DQN()
    elif args.rl_style=="ddpg":
        ram = MemoryBuffer(1000)
        ddpg = Trainer(style_raven_len*4+2, style_raven_len, 1, ram)
    alpha_1=0.1

    if args.rl_style=="dqn":
        a = dqn.choose_action([0.5] * 3)  # TODO
    elif args.rl_style=="ddpg":
        action_ = ddpg.get_exploration_action(np.zeros([style_raven_len*4+2]).astype(np.float32),alpha_1)
    if args.type_loss:loss_fn=nn.BCELoss()
    best_acc=0.0
    while True:
        since=time.time()
        print(action_)
        for i in range(style_raven_len):
            tb.scalar_summary("action/a"+str(i), action_[i], epoch_count)

        data_files = preprocess.provide_data(args.regime, style_raven_len, action_,style_raven)

        train_files = [data_file for data_file in data_files if 'train' in data_file]
        print("train_num:", len(train_files))
        train_loader = torch.utils.data.DataLoader(Dataset(args,train_files), batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.numwork)
        model.train()
        iter_epoch = int(len(train_files) / args.batch_size)
        acc_part_train=np.zeros([style_raven_len,2]).astype(np.float32)

        mean_loss_train= np.zeros([style_raven_len, 2]).astype(np.float32)
        loss_train=0
        for x, y,style,me in train_loader:
            if x.shape[0]<10:
                print(x.shape[0])
                break
            x, y ,meta = Variable(x).cuda(), Variable(y).cuda(), Variable(me).cuda()
            if args.gpunum > 1:
                optimizer.module.zero_grad()
            else:
                optimizer.zero_grad()
            if args.type_loss:
                pred_train, pred_meta= model(x)
            else:
                pred_train = model(x)
            loss_ = F.nll_loss(pred_train, y,reduce=False)
            loss=loss_.mean() if not args.type_loss else loss_.mean()+10*loss_fn(pred_meta,meta)
            loss.backward()
            if args.gpunum > 1:
                optimizer.module.step()
            else:
                optimizer.step()
            iter_count += 1
            pred = pred_train.data.max(1)[1]
            correct = pred.eq(y.data).cpu()
            loss_train+=loss.item()
            for num, style_pers in enumerate(style):
                style_pers = style_pers[:-4].split("/")[-1].split("_")[3:]
                for style_per in style_pers:
                    style_per=int(style_per)
                    if correct[num] == 1:
                        acc_part_train[style_per, 0] += 1
                    acc_part_train[style_per, 1] += 1
                    #mean_pred_train[style_per,0] += pred_train[num,y[num].item()].data.cpu()
                    #mean_pred_train[style_per, 1] += 1
                    mean_loss_train[style_per,0] += loss_[num].item()
                    mean_loss_train[style_per, 1] += 1
            accuracy_total = correct.sum() * 100.0 / len(y)

            if iter_count %10 == 0:
                iter_c = iter_count % iter_epoch
                print(time.strftime('%H:%M:%S', time.localtime(time.time())),
                      ('train_epoch:%d,iter_count:%d/%d, loss:%.3f, acc:%.1f') % (
                      epoch_count, iter_c, iter_epoch, loss, accuracy_total))
                tb.scalar_summary("train_loss",loss,iter_count)
        loss_train=loss_train/len(train_files)
        #mean_pred_train=[x[0]/ x[1] for x in mean_pred_train]
        mean_loss_train=[x[0]/ x[1] for x in mean_loss_train]
        acc_part_train = [x[0] / x[1] if x[1]!=0 else 0  for x in acc_part_train]
        print(acc_part_train)
        if epoch_count %args.lr_step ==0:
            print("change lr")
            adjust_learning_rate(optimizer, epoch_count, args.lr_step,args.gpunum)
        time_elapsed = time.time() - since
        print('train epoch in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
        #acc_p=np.array([x[0]/x[1] for x in acc_part])
        #print(acc_p)
        with torch.no_grad():
            model.eval()
            accuracy_all = []
            iter_test=0
            acc_part_val = np.zeros([style_raven_len, 2]).astype(np.float32)
            for x, y, style,me in val_loader:
                iter_test+=1
                x, y = Variable(x).cuda(), Variable(y).cuda()
                pred,_ = model(x)
                pred = pred.data.max(1)[1]
                correct = pred.eq(y.data).cpu().numpy()
                accuracy = correct.sum() * 100.0 / len(y)
                for num, style_pers in enumerate(style):
                    style_pers = style_pers[:-4].split("/")[-1].split("_")[3:]
                    for style_per in style_pers:
                        style_per = int(style_per)
                        if correct[num] == 1:
                            acc_part_val[style_per, 0] += 1
                        acc_part_val[style_per, 1] += 1
                accuracy_all.append(accuracy)

                # if iter_test % 10 == 0:
                #
                #     print(time.strftime('%H:%M:%S', time.localtime(time.time())),
                #           ('test_iter:%d, acc:%.1f') % (
                #               iter_test, accuracy))

        accuracy_all = sum(accuracy_all) / len(accuracy_all)
        acc_part_val = [x[0] / x[1] if x[1]!=0 else 0 for x in acc_part_val ]
        baseline_rl=70
        reward=np.mean(acc_part_val)*100-baseline_rl
        tb.scalar_summary("valreward", reward,epoch_count)
        action_list=[x for x in a]
        cur_state=np.array(acc_part_val+acc_part_train+action_list+mean_loss_train
                           +[loss_train]+[epoch_count]).astype(np.float32)
        #np.expand_dims(, axis=0)
        if args.rl_style == "dqn":
            a = dqn.choose_action(cur_state)  # TODO
        elif args.rl_style == "ddpg":
            a = ddpg.get_exploration_action(cur_state,alpha_1)

        if alpha_1<1:
            alpha_1+=0.005#0.1
        if epoch_count > 1:
            if args.rl_style == "dqn":dqn.store_transition(last_state, a, reward , cur_state)
            elif args.rl_style == "ddpg":ram.add(last_state, a, reward, cur_state)


        if epoch_count > 1:
            if args.rl_style == "dqn":dqn.learn()
            elif args.rl_style == "ddpg":loss_actor, loss_critic=ddpg.optimize()
            print('------------------------------------')
            print('learn q learning')
            print('------------------------------------')
            tb.scalar_summary("loss_actor", loss_actor, epoch_count)
            tb.scalar_summary("loss_critic", loss_critic, epoch_count)


        last_state=cur_state
        time_elapsed = time.time() - since
        print('test epoch in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
        print('------------------------------------')
        print(('epoch:%d, acc:%.1f') % (epoch_count, accuracy_all))
        print('------------------------------------')
        if accuracy_all>best_acc:
            best_acc=max(best_acc,accuracy_all)
            #ddpg.save_models(args.model_dir + '/', epoch_count)
            save_state(model.state_dict(), args.model_dir + "/epochbest")
        epoch_count += 1
        if epoch_count%20==0:
            print("save weights")
            ddpg.save_models(args.model_dir+'/',epoch_count )
            save_state(model.state_dict(), args.model_dir+"/epoch"+str(epoch_count))



