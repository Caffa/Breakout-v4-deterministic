import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

import csv
import os

torch.manual_seed(30)
np.random.seed(6)

env = gym.make('Breakout-v0')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Replay Memory"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

"""three fc"""
class three_fc(nn.Module):
    def __init__(self,inputs,outputs):
        super(three_fc,self).__init__()
        C = inputs[0]
        #self.fc1 = nn.Linear(inputs, 24, bias=True)
        self.conv1 = nn.Conv2d(in_channels = C, out_channels = int(C*1.5), kernel_size = 10,
                               stride = 5, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = int(C*1.5), out_channels = int(C/2), kernel_size = 5)
        self.conv3 = nn.Conv2d(in_channels = int(C/2), out_channels = 1, kernel_size = 3, stride = 2)
        self.fc = nn.Linear(1*18*13, outputs, bias=True)
    def forward(self,x):
        x = self.conv1(x)
        x = F.selu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1,1*18*13)
        x = self.fc(x)
        return(x)

"""Training"""
BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
#init_screen = get_screen()
#_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
# Get number of actions from gym action space
def convert_to_grey_tensor(obs):
  obs = torch.tensor(obs).float()
  obs = obs.transpose(1,2)
  obs = obs.transpose(0,1)
  obs = T.ToPILImage()(obs)
  obs = obs.convert('P')
  obs = T.ToTensor()(obs).to(device)
  return(obs)

observation = convert_to_grey_tensor(env.reset())
prev = 4
prev = max(prev, 1)
obs_list = []
for i in range(prev):
  obs_list.append(observation)
observation = torch.cat(obs_list)
print(observation.shape)
n_input = list(observation.shape)
observation = observation.unsqueeze(0).float()
n_actions = env.action_space.n

policy_net = three_fc(n_input,n_actions).to(device)
target_net = three_fc(n_input,n_actions).to(device)
#policy_net = models.resnet18(num_classes = n_actions).to(device)
#target_net = models.resnet18(num_classes = n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

lr = 0.01
optimizer = optim.Adam(policy_net.parameters(),lr=lr) # default lr is 0.01 
memory = ReplayMemory(10000) # default is 10000

#observation = env.reset()
#observation = torch.tensor([observation], device=device).float().unsqueeze(0).to(device)
#state = torch.cat((observation,observation),1)
#policy_net(state)

steps_done = 0

# Create cnn class

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_score = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_score, dtype=torch.float)
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


"""Training loop"""
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    non_final_next_action = torch.cat([s for s in batch.next_action if s is not None])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = torch.gather(target_net(non_final_next_states),1,non_final_next_action).view(-1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #print('Loss: ',loss.item())
    #loss = F.l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

max_num_episodes = 500 # Original is 50
i_episode = 0
max_score = float('-inf')
max_mean = float('-inf')
update_time = 50
frame_skip = 3

# Create the csv file name where the values will be stored
filename='Breakout_pass3_channel4_newdoubleq'

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

while i_episode <= max_num_episodes:
    # Initialize the environment and state
    prev_obs = []
    R_tot = 0.0
    observation = env.reset()
    observation = convert_to_grey_tensor(observation).unsqueeze(0)
    for pr in range(prev):
        prev_obs.append(observation)
    state = torch.cat(prev_obs,1)
    if i_episode % update_time == 0 and i_episode != 0:
        lr = lr/2
        print('new learning rate:',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    for t in count():
        # Loop this to pass a certain amount of frames
        done = False
        f = 0.0
        p_reward = 0.0
        while f < frame_skip and not done:
          if f == 0:
            action = select_action(state)
            next_observation, reward, done, info = env.step(action.item())
          else:
            next_observation, reward, done, info = env.step(0)
          p_reward += reward
          f += 1
        R_tot += p_reward
        p_reward = torch.tensor([p_reward], device=device,dtype=torch.float)

        # Observe new state
        if not done:
            for pr in range(prev - 1):
                prev_obs[pr] = prev_obs[pr + 1]
            next_observation = convert_to_grey_tensor(next_observation).unsqueeze(0)
            prev_obs[-1] = next_observation
            next_state = torch.cat(prev_obs,1)
            with torch.no_grad():
                next_action = policy_net(next_state).max(1)[1].view(1, 1)
        else:
            next_observation = None
            next_state = None
            next_action = None

        # Store the transition in memory
        #print(type(observation),type(next_observation))
        memory.push(state, action, next_state, p_reward, next_action)

        # Move to the next state
        observation = next_observation
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            if max_score < R_tot:
                max_score = R_tot
            episode_score.append(R_tot)
            means = plot_durations()
            if max_mean < means:
                max_mean = means
            print('episode',i_episode)
            print('score:',R_tot,'max:',max_score)
            print('mean:',means,'max:',max_mean)
            break
    best_R = '-'
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
      target_net.load_state_dict(policy_net.state_dict())
    if i_episode % 50 == 0:
      model_name='Breakout_pass3_channel4_newdoubleq_'+str(i_episode)+'.pth'
      torch.save({'model':target_net.state_dict(),'optimizer':optimizer.state_dict()}, model_name)
      # do testing
      prev_obs = []
      best_R = 0.0
      observation = env.reset()
      observation = convert_to_grey_tensor(observation).unsqueeze(0)
      for pr in range(prev):
          prev_obs.append(observation)
      state = torch.cat(prev_obs,1)
      for t in count():
        # Loop this to pass a certain amount of frames
        done = False
        f = 0.0
        p_reward = 0.0
        while f < frame_skip and not done:
          if f == 0:
            action = select_action(state)
            next_observation, reward, done, info = env.step(action.item())
          else:
            next_observation, reward, done, info = env.step(0)
          p_reward += reward
          f += 1
        best_R += p_reward
        if not done:
          for pr in range(prev - 1):
              prev_obs[pr] = prev_obs[pr + 1]
          next_observation = convert_to_grey_tensor(next_observation).unsqueeze(0)
          prev_obs[-1] = next_observation
          next_state = torch.cat(prev_obs,1)
        else:
          next_observation = None
          next_state = None
        observation = next_observation
        state = next_state
        if done:
          print('test score:',best_R)
          break
    readingfile_ascsv.writerow([i_episode,R_tot,means,best_R])
    readingfile.flush()
    i_episode+=1

readingfile.close();
print('Complete')
#env.render()
env.close()
plt.ioff()
plt.show()