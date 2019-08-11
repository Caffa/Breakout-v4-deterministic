import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import csv
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch import autograd

#env = gym.make('MsPacmanDeterministic-v4')
#env = gym.make('Breakout-ram-v0')
env = gym.make('BreakoutDeterministic-v4')
action_meaning = env.unwrapped.get_action_meanings()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")

class PicturePred(nn.Module):
    def __init__(self,in_channel,outputs):
        super(PicturePred,self).__init__()
        self.pi1 = nn.Sequential(
                nn.Conv2d(in_channels = in_channel,out_channels=7,kernel_size=5),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels = 7, out_channels = 5, kernel_size = 4, stride = 2),
                nn.MaxPool2d(kernel_size = 3),
                nn.Conv2d(in_channels = 5, out_channels = outputs, kernel_size = 5, stride = 3),
                nn.LeakyReLU()
                )
        self.b1 = nn.Sequential(
                nn.Conv2d(in_channels = in_channel,out_channels=7,kernel_size=5),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels = 7, out_channels = 5, kernel_size = 4, stride = 2),
                nn.MaxPool2d(kernel_size = 3),
                nn.Conv2d(in_channels = 5, out_channels = outputs, kernel_size = 5, stride = 3),
                nn.LeakyReLU()
                )
        self.softmax = nn.Sequential(nn.Linear(outputs*10*7, outputs, bias=True),nn.Softmax())
        self.fcb = nn.Linear(outputs*10*7, 1, bias=True)
        self.out = outputs
    def forward(self,x):
        pi = self.pi1(x)
        #print(pi.shape)
        b = self.b1(x)
        #print(b.shape)
        pi = pi.view(-1,self.out*10*7)
        b = b.view(-1,self.out*10*7)
        pi = self.softmax(pi).clamp(0.0001,1.0)
        b = self.fcb(b)
        return(pi,b)
        
class RAMPred(nn.Module):
    def __init__(self,in_channel,outputs):
        super(RAMPred,self).__init__()
        self.pi1 = nn.Sequential(
                nn.Linear(128,128,bias=True),
                nn.LeakyReLU(),
                nn.Linear(128,64,bias=True),
                nn.LeakyReLU(),
                nn.Linear(64,32,bias=True),
                nn.LeakyReLU()
                )
        self.b1 = nn.Sequential(
                nn.Linear(128,128,bias=True),
                nn.LeakyReLU(),
                nn.Linear(128,64,bias=True),
                nn.LeakyReLU(),
                nn.Linear(64,32,bias=True),
                nn.LeakyReLU()
                )
        self.softmax = nn.Sequential(nn.Linear(in_channel*32, outputs, bias=True),nn.Softmax())
        self.fcb = nn.Linear(in_channel*32, 1, bias=True)
        self.inputs = in_channel
    def forward(self,x):
        pi = self.pi1(x)
        #print(pi.shape)
        b = self.b1(x)
        #print(b.shape)
        pi = pi.view(-1,self.inputs*32)
        b = b.view(-1,self.inputs*32)
        pi = self.softmax(pi).clamp(0.0001,1.0)
        b = self.fcb(b)
        return(pi,b)
        
class MyFunc(autograd.Function):
  @staticmethod
  def forward(ctx, inp):
    return inp.clone()
  @staticmethod
  def backward(ctx, gO):
    raise RuntimeError("Some error in backward")
    return gO.clone()

def run_fn(a):
  out = MyFunc.apply(a)
  return out.sum()
  
"""Training"""
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400
#TARGET_UPDATE = 10


def convert_to_grey_tensor(obs,device):
  obs = torch.from_numpy(obs).float()
  obs = obs.transpose(1,2)
  obs = obs.transpose(0,1)
  obs = T.ToPILImage()(obs)
  obs = obs.convert('P')
  obs = T.ToTensor()(obs)
  obs = obs.to(device)
  return(obs)

def convert_to_ram_tensor(obs,device):
  obs = torch.tensor([obs], device=device).float().unsqueeze(0).to(device)
  return(obs)

# Get number of actions from gym action space
observation = env.reset()
#print(observation)
#print(type(observation))
#observation = convert_to_ram_tensor(observation,device)
observation = convert_to_grey_tensor(env.reset(),device)
obs_window = []
prev = 4
for i in range(prev):
  obs_window.append(observation)
observation = torch.cat(obs_window).unsqueeze(0)
#observation = torch.cat(obs_window,1)
n_inputs = list(observation.shape)
observation = observation.float()
n_actions = env.action_space.n
print(n_inputs,n_actions)

policy_net = PicturePred(prev,n_actions).to(device)

print(policy_net(observation))
#1/0

lr = 8e-3
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
            #action = pi.max(1)[1].view(1, 1)
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

def optimize_model(obs_list, act_list, rew_list, gamma=GAMMA, N = 1):
    Time = len(obs_list)
    act_list = torch.cat(act_list)
    obs_list = torch.cat(obs_list)
    rew_list = torch.tensor(rew_list, device=device).float().unsqueeze(1)

    prb_list, b_list = policy_net(obs_list)
    
    anc_list = torch.zeros([Time, 1], device=device).float()
    #print(Time)
    for t in range(Time):
        for n in range(min(N, Time-t)):
            anc_list[t] += rew_list[t+n]*gamma**n
        if t + N < Time:
            anc_list[t] += b_list[t+N]*gamma**N
    anc_list = rew_list.detach()
    #print(anc_list.sum())
    L2 = F.smooth_l1_loss(b_list,anc_list)
    prb_list = torch.gather(prb_list,1,act_list)
    L1 = torch.mean(-torch.log(prb_list)*(anc_list - b_list.detach()))
    L = L1 + L2
    optimizer.zero_grad()
    #with autograd.detect_anomaly():
      #out = run_fn(L)
      #out.backward()
    L.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

max_num_episodes = 500 # Original is 50
i_episode = 0
max_time = float('-inf')
max_mean = float('-inf')
prev_round = {}
best_round = {}

# Create the csv file name where the values will be stored
filename='Breakout_Pic_A2C_8Forward_4Behind_Rand'

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
    if i_episode % 50 == 0 and i_episode != 0:
        for param_group in optimizer.param_groups:
            lr = lr/2
            print('new lr',lr)
            param_group['lr'] = lr
    observation = env.reset()
    obs_window = []
    for pr in range(prev):
      obs_window.append(convert_to_grey_tensor(observation,device))
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
            obs_window.append(convert_to_grey_tensor(next_observation,device))
            obs_window = obs_window[1:]
            next_state = torch.cat(obs_window).unsqueeze(0)
        else:
            next_observation = None
            next_state = None

        observ_list.append(state)
        action_list.append(action)
        reward_list.append(reward)
        #optimize_model(observ_list, action_list, reward_list, gamma=GAMMA, N = 8)
        state = next_state
        #print(state.shape)
        if done:
            optimize_model(observ_list, action_list, reward_list, gamma=GAMMA, N = 8)
            if max_time <= total_reward:
                max_time = total_reward
                best_round['observations'] = observ_list[:]
                best_round['actions'] = action_list[:]
                best_round['rewards'] = reward_list[:]
            else:
                optimize_model(best_round['observations'], best_round['actions'], best_round['rewards'], gamma=GAMMA, N = 8)
            if i_episode > 0:
                optimize_model(prev_round['observations'], prev_round['actions'], prev_round['rewards'], gamma=GAMMA, N = 8)
            prev_round['observations'] = observ_list[:]
            prev_round['actions'] = action_list[:]
            prev_round['rewards'] = reward_list[:]
            episode_durations.append(total_reward)
            means = plot_durations()
            if max_mean < means:
                max_mean = means
            print('episode',i_episode)
            print('score:',total_reward,'max:',max_time)
            print('mean:',means,'max:',max_mean)
            break

    test_reward = '-'
    if i_episode % 50 == 0 and i_episode > 0:
        model_name='Breakout_Pic_8Forward_4Behind_Rand_'+str(i_episode)+'.pth'
        torch.save({'model':policy_net.state_dict(),'optimizer':optimizer.state_dict()}, model_name)
        policy_net.eval()
        observation = env.reset()
        #env.render()
        obs_window = []
        for pr in range(prev):
          obs_window.append(convert_to_grey_tensor(observation,device))
        state = torch.cat(obs_window).unsqueeze(0)
        test_reward = 0.0
        for t in count():
            #print(t)
            #print(state.shape)
            #print(policy_net(state))
            action = policy_net(state)[0].max(1)[1].view(1, 1)
            next_observation, reward, done, info = env.step(action.item())
            #env.render()
            #print(action_meaning[action.item()])
            #print('Reward Gain:',reward)
            #print(done)
            test_reward += reward
            #print('Reward:',test_reward)
            if not done:
                obs_window.append(convert_to_grey_tensor(next_observation,device))
                obs_window = obs_window[1:]
                state = torch.cat(obs_window).unsqueeze(0)
            else:
                break
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