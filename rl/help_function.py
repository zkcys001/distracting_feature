import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np


def encode5(style):

    index = np.where(style == 1)
    att = index[0][1] - 2
    return att

def encode20(style,n):
    #code = ['shape', 'line', "color"5, 'number'2, 'position'3, 'size'5,'type'5,
    #        'progression', "xor", "or", 'and', 'consistent_union']

    #att=style.index(1,2)-1
    #rel = style.index(1, 3)
    index=np.where(style==1)
    att=index[0][1]-2
    rel=index[0][2]-7
    if index[0][0]==0:
        if att==0:
            att=0
        elif att==1:
            att=5
            if rel==4:
                rel=1
        elif att == 2:
            att = 7
            rel-=1
        elif att == 3:
            att = 10
        else:
            att = 15
        return att + rel
    else:
        if att==4:
            att=20
            rel = rel - 1
        else:
            att=24

        return att+rel if n!=9 else att+rel-20
def to_var(x, volatile=False, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile, requires_grad=requires_grad)

def state_func(configs):
    '''
    configs = {
                        'num_classes': num_classes,
                        'labels': labels,
                        'inputs': inputs,
                        'student': student,
                        'current_iter': i_tau,
                        'max_iter': max_t,
                        'train_loss_history': training_loss_history,
                        'val_loss_history': val_loss_history
                    }
    '''
    num_classes = configs['num_classes']
    labels = configs['labels']
    inputs = configs['inputs']
    student = configs['student']
    current_iter = configs['current_iter']
    max_iter = configs['max_iter']
    train_loss_history = configs['train_loss_history']
    val_loss_history = configs['val_loss_history']

    _inputs = {'inputs':inputs, 'labels':labels}

    predicts, _ = student(_inputs, None) # predicts are logits

    predicts = nn.LogSoftmax()(predicts)
    predicts = torch.exp(predicts)

    n_samples = inputs.size(0)
    data_features = to_var(torch.zeros(n_samples, num_classes))
    data_features[range(n_samples), labels.data] = 1

    # def sigmoid(x):
    #     return 1.0/(1.0 + math.exp(-x))
    def normalize_loss(loss):
        return loss/2.3
    # [ max_iter; averaged_train_loss; best_val_loss ]
    model_features = to_var(torch.zeros(n_samples, 3))
    model_features[:, 0] = current_iter / max_iter  # current iteration number
    model_features[:, 1] = min(1.0, 1.0 if len(train_loss_history) == 0 else sum(train_loss_history)/len(train_loss_history)/2.3)
    # sigmoid(sum(train_loss_history)/len(train_loss_history)) # averaged training loss
    model_features[:, 2] = min(1.0, 1.0 if len(val_loss_history) == 0 else min(val_loss_history)/2.3)
    # sigmoid(min(val_loss_history))

    combined_features = to_var(torch.zeros(n_samples, 12))
    combined_features[:, :10] = predicts

    eps = 1e-6
    combined_features[:, 10] = -torch.log(predicts[range(n_samples), labels.data] + eps)

    mask = to_var(torch.ones(n_samples, num_classes))

    mask[range(n_samples), labels.data] = 0
    combined_features[:, 11] = predicts[range(n_samples), labels.data] - torch.max(mask*predicts, 1)[0]

    states = torch.cat([data_features, model_features, combined_features], 1)
    return states