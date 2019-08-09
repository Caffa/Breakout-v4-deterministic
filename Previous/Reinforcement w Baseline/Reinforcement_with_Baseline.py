import gym
import math
import random
#import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from collections import namedtuple
from itertools import count
from torch import autograd
#from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
import numpy as np

import csv

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

env = gym.make('BreakoutDeterministic-v4')
#env = gym.make('MsPacman-v0')

class MyFunc(autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp.clone()
    @staticmethod
    def backward(ctx, gO):
        # Error during the backward pass
        raise RuntimeError("Some error in backward")
        return gO.clone()

def run_fn(a):
    out = MyFunc.apply(a)
    return out.sum()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

"""3 fc"""
class three_fc(nn.Module):
    def __init__(self,inputs,outputs):
        super(three_fc,self).__init__()
        self.fc1 = nn.Linear(inputs[-1], 24, bias=True)
        self.fc2 = nn.Linear(24, 24, bias=True) # Default for all bias is true
        self.aftview = 24*inputs[1]
        self.softmax = nn.Sequential(nn.Linear(self.aftview, outputs, bias=True),nn.Softmax())
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = x.view(-1,self.aftview)
        x = self.softmax(x).clamp(0.001,1.0)
        return(x)

class BreakoutR(nn.Module):
    def __init__(self,in_channel,outputs):
        super(BreakoutR,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channel,out_channels = 7, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 7, out_channels = 6, kernel_size = 4, stride = 2)
        self.max1 = nn.MaxPool2d(kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 6, out_channels = 5, kernel_size = 5, stride = 3)
        self.fc1 = nn.Linear(70, 1)
        self.fcp = nn.Sequential(nn.Linear(5,outputs,bias=True),nn.Softmax())
        self.fcb = nn.Linear(5,1,bias=True)
    def forward(self,x):
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.max1(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1,5,10*7)
        x = self.fc1(x)
        x = x.squeeze(2)
        pi = self.fcp(x)
        b = self.fcb(x)
        return(pi,b)

"""Training"""
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400
#TARGET_UPDATE = 10


def convert_to_grey_tensor(obs):
  obs = torch.tensor(obs).float()
  obs = obs.transpose(1,2)
  obs = obs.transpose(0,1)
  obs = T.ToPILImage()(obs)
  obs = obs.convert('P')
  obs = T.ToTensor()(obs).to(device)
  return(obs)

# Get number of actions from gym action space
observation = convert_to_grey_tensor(env.reset())
obs_window = []
prev = 3
for i in range(prev):
  obs_window.append(observation)
observation = torch.cat(obs_window).unsqueeze(0)
n_inputs = list(observation.shape)
observation = observation.float()
#eps = np.finfo(np.float32).eps.item()
n_actions = env.action_space.n
print(n_inputs,n_actions)

policy_net = BreakoutR(prev,n_actions).to(device)

lr = 0.01
optimizer = optim.Adam(policy_net.parameters(),lr=lr) # default lr is 0.01

# Create cnn class
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            pi = policy_net(state)[0]
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = torch.distributions.categorical.Categorical(pi[0]).sample().unsqueeze(0).unsqueeze(0)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return(action)

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())
    last_mean = float('-inf')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        last_mean = means.numpy()[-1]

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    return(last_mean)

def optimize_model(obs_list, act_list, rew_list):
    Steps = len(obs_list)
    for i in range(Steps):
        t = Steps - 1 - i
        if t != Steps-1:
            rew_list[t] = rew_list[t] + GAMMA*rew_list[t+1]
    #rew_mean = np.mean(rew_list)
    #rew_std = np.std(rew_list)
    #for i in range(Steps):
    #  rew_list[i] = (rew_list[i] - rew_mean)/(rew_std + eps)
    act_list = torch.cat(act_list)
    obs_list = torch.cat(obs_list)
    rew_list = torch.tensor(rew_list, device=device).float().unsqueeze(1)
    #print(rew_list)
    prb_list, vst_list = policy_net(obs_list)
    L2 = F.smooth_l1_loss(vst_list,rew_list)
    #L2 = -torch.mean(rew_list-vst_list)
    prb_list = torch.gather(prb_list,1,act_list)
    L1 = torch.mean(-torch.log(prb_list)*(rew_list - vst_list.detach()))
    L = L1 + L2
    #with autograd.detect_anomaly():
    #  out = run_fn(L)
    #  optimizer.zero_grad()
    #  out.backward()
    #  for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1,1)
    #  optimizer.step()
    optimizer.zero_grad()
    L.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

max_num_episodes = 20 # Original is 50
i_episode = 0
max_time = float('-inf')
max_mean = float('-inf')

# Create the csv file name where the values will be stored
filename='Breakout_Reinforcement_Baseline'

# Check filename and add a number at the back if said file already exists
if os.path.exists(filename+'.csv'):
	i=1
	while os.path.exists(filename+'('+str(i)+').csv'):
		i+=1
	filename=filename+'('+str(i)+').csv'
else:
	filename=filename+'.csv'

readingfile=open(filename,"w", newline='')
readingfile_ascsv=csv.writer(readingfile,delimiter=',',quotechar="|",quoting=csv.QUOTE_NONE)
readingfile_ascsv.writerow(['n','score','ave score','test score'])
readingfile.flush()

while i_episode <= max_num_episodes:
    #print(policy_net.conv1.weight.grad)
    #print(policy_net.conv2.weight.grad)
    #print(policy_net.conv3.weight.grad)
    #print(policy_net.fc1.weight.grad)
    if i_episode % 50 == 0 and i_episode != 0:
        for param_group in optimizer.param_groups:
            lr = lr/2
            print('new lr',lr)
            param_group['lr'] = lr
    observation = env.reset()
    #observation = convert_to_grey_tensor(env.reset())
    obs_window = []
    for pr in range(prev):
      obs_window.append(convert_to_grey_tensor(observation))
    state = torch.cat(obs_window).unsqueeze(0)
    #print(state.shape)
    observ_list = []
    action_list = []
    reward_list = []
    total_reward = 0.0
    for t in count():
        action = select_action(state)
        #print(action.item())
        next_observation, reward, done, info = env.step(action.item())
        total_reward += reward
        #reward = torch.tensor([reward], device=device,dtype=torch.float).unsqueeze(0)
        
        if not done:
            #next_observation = convert_to_grey_tensor(next_observation)
            obs_window.append(convert_to_grey_tensor(next_observation))
            obs_window = obs_window[1:]
            next_state = torch.cat(obs_window).unsqueeze(0)
        else:
            #next_observation = None
            next_state = None

        observ_list.append(state)
        action_list.append(action)
        reward_list.append(reward)
        state = next_state
        #print(state.shape)
        if done:
            optimize_model(observ_list, action_list, reward_list)
            if max_time < total_reward:
                max_time = total_reward
            episode_durations.append(total_reward)
            means = plot_durations()
            if max_mean < means:
                max_mean = means
            print('episode',i_episode)
            print('score:',total_reward,'max:',max_time)
            print('mean:',means,'max:',max_mean)
            break

    test_reward = '-'
    if i_episode % 20 == 0 and i_episode != 0:
        model_name='ReinforcementBreakoutBaseline'+str(i_episode)+'.pth'
        torch.save({'model':policy_net.state_dict(),'optimizer':optimizer.state_dict()}, model_name)
        policy_net.eval()
        observation = env.reset()
        #observation = convert_to_grey_tensor(env.reset())
        #env.render()
        obs_window = []
        for pr in range(prev):
          obs_window.append(convert_to_grey_tensor(observation))
        state = torch.cat(obs_window).unsqueeze(0)
        test_reward = 0.0
        for t in count():
            #print(state.shape)
            with torch.no_grad():
                action = policy_net(state)[0].max(1)[1].view(1, 1)
            # For some reason doing in eval mode always gives nan values
            next_observation, reward, done, info = env.step(action.item())
            print('Step',t,'State:',policy_net(state))
            print('Action:',action.item())
            #env.render()
            #print(action.item(),reward)
            #print(done)
            test_reward += reward
            #reward = torch.tensor([reward], device=device,dtype=torch.float).unsqueeze(0)
            if not done:
                #next_observation = convert_to_grey_tensor(next_observation)
                obs_window.append(convert_to_grey_tensor(next_observation))
                obs_window = obs_window[1:]
                next_state = torch.cat(obs_window).unsqueeze(0)
            else:
                break
            print('Picture Different:',1-torch.eq(state,next_state).all().item())
            state = next_state
        policy_net.train()
        print('Test Score:',test_reward)
    readingfile_ascsv.writerow([i_episode,total_reward,means,test_reward])
    readingfile.flush()
    i_episode += 1


readingfile.close()
print('Complete')
#env.render()
env.close()
plt.ioff()
plt.show()