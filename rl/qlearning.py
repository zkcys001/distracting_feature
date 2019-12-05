"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
# Hyper Parameters
BATCH_SIZE = 8
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 10  # target update frequency
MEMORY_CAPACITY = 80

N_ACTIONS = 50#env.action_space.n  # 2 L or R
N_STATES = 20#env.observation_space.shape[0]  #



class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.out.bias.data.fill_(0.5)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY,5+ N_STATES*2  + 1))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample

        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x.cuda())

            action = np.array((torch.max(actions_value[:,:10], 1)[1].cpu().data,
                      torch.max(actions_value[:,10:20], 1)[1].cpu().data,
                      torch.max(actions_value[:,20:30], 1)[1].cpu().data,
                      torch.max(actions_value[:,30:40], 1)[1].cpu().data,
                      torch.max(actions_value[:,40:], 1)[1].cpu().data))
            action=action/10
        else:  # random
            action = [random.randint(0,9)/10 for _ in range(5)]
        return action

    def store_transition(self, s, a, r, s_):
        s=np.array(s)
        s_=np.array(s_)
        a=np.array(a)
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).cuda()
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES +5].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, N_STATES +5:N_STATES +5+1]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_e =  torch.zeros(BATCH_SIZE, 5).cuda()
        q_target =  torch.zeros(BATCH_SIZE, 5).cuda()
        for i in range(5):
            xx= torch.squeeze(q_eval.gather(1,torch.unsqueeze(10*i+b_a[:,i]*10+1,1)),1) # shape (batch, 1)
            q_e[:, i]=xx
            q_target[:,i] =torch.squeeze( b_r + GAMMA * q_next[:,i*10:10+i*10].max(1)[0].view(BATCH_SIZE, 1) ,1) # shape (batch, 1)
        loss=self.loss_func(q_e, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':

    dqn = DQN()
