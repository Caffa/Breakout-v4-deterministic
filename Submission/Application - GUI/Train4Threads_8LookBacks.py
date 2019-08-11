import sys
from skimage.transform import resize
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import gym
import numpy as np
import h5py
import argparse
from pathlib import Path
import os
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import csv
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
import time
from keras import backend as K

def build_network(input_shape, output_shape):

    print("Building Network")



    state = Input(shape=input_shape)
    h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(state)
    h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)

    value = Dense(1, activation='linear', name='value')(h)
    policy = Dense(output_shape, activation='softmax', name='policy')(h)
    print("Value Model Setting")
    value_network = Model(inputs=state, outputs=value)
    print("Policy Model Setting")
    policy_network = Model(inputs=state, outputs=policy)

    advantage = Input(shape=(1,))
    print("Train Model Setting")
    train_network = Model(inputs=[state, advantage], outputs=[value, policy])
    print("Finish Building Model")
    return value_network, policy_network, train_network, advantage


def policy_loss(advantage=0., beta=0.01):


    def loss(y_true, y_pred):
        # return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
        #        beta * K.sum(y_pred * K.log(y_pred + K.epsilon())) #+ noOpsPenalty
        differencePredict = K.sum(y_true * y_pred, axis=-1)
        logDiff = K.log(differencePredict + K.epsilon())
        logPrediction = K.log(y_pred + K.epsilon())
        loss = -K.sum(logDiff * K.flatten(advantage)) + \
               beta * K.sum(y_pred * logPrediction)

        return loss

    return loss


def value_loss():

    def loss(y_true, y_pred):
        squaredDiff = K.square(y_true - y_pred)
        return 0.5 * K.sum(squaredDiff)

    return loss





class Brain(object):
    global arguments
    def __init__(self, action_space, batch_size=32, screen=(84, 84), swap_freq=200):
        self.screen = screen
        self.input_depth = 1
        self.past_range = arguments["LookBackFramesNumber"]
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen
        self.batch_size = batch_size

        _, _, self.train_net, advantage = build_network(self.observation_shape, action_space.n)
        print("Compiling Train Net")
        # self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
        #                        loss=[value_loss(), policy_loss(advantage, arguments["beta"])])
        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99), loss=[value_loss(), policy_loss(advantage, arguments["beta"])])

        print("Setting Misc Vars")
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.n))
        self.policyLossMem = deque(maxlen=25)
        self.validationLossMem = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.entropy = deque(maxlen=25)
        self.counter = 0
        print("Done: Agent Setup")

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        # print("Learning Begins")
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames

        values, policy = self.train_net.predict([last_observations, self.unroll])

        self.targets.fill(0.)
        advantage = rewards - values.flatten()
        self.targets[self.unroll, actions] = 1.

        loss = self.train_net.train_on_batch([last_observations, advantage], [rewards, self.targets])
        entropy = np.mean(-policy * np.log(policy + 0.00000001))
        self.policyLossMem.append(loss[2])
        self.validationLossMem.append(loss[1])
        self.entropy.append(entropy)
        self.values.append(np.mean(values))
        min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)
        print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
              '--- Value-Loss: %10.6f; Avg: %10.6f '
              '--- Entropy: %7.6f; Avg: %7.6f '
              '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f' % (
                  self.counter,
                  loss[2], np.mean(self.policyLossMem),
                  loss[1], np.mean(self.validationLossMem),
                  entropy, np.mean(self.entropy),
                  min_val, max_val, avg_val), end='')


        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False


def Learning(MemoryQueue, weight_dict):
    global arguments
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0.3,' + \
                                 'compiledir=th_comp_learn'


    print(' %5d> Learning process' % (pid,))


    save_freq = arguments["SavePolicyEvery"]
    learning_rate = arguments["LearningRate"]
    batch_size = arguments["BatchSize"]
    checkpoint = arguments["UseSavedCheckpoint"]
    steps = arguments["LRDecayRate"]


    env = gym.make('BreakoutDeterministic-v4')
    agent = Brain(env.action_space, batch_size = arguments["BatchSize"], swap_freq=arguments["SwapRate"])
    print("Done Creating Agent")


    if checkpoint > 0:
        maindir = str(sys.path[0])
        pathForSavedWeights = os.path.join(maindir, "saved_weights", 'model-BreakoutDeterministic-v4-%d.h5' % (checkpoint,))
        print(' %5d> Loading weights from file %s' % (pid, pathForSavedWeights))
        agent.train_net.load_weights(pathForSavedWeights)


    print(' %5d> Setting weights in dict' % (pid,))
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    print(' %5d> Finish Setting Weights' % (pid,))


    last_obs = np.zeros((batch_size,) + agent.observation_shape)
    actions = np.zeros(batch_size, dtype=np.int32)
    rewards = np.zeros(batch_size)


    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:


        last_obs[idx, ...], actions[idx], rewards[idx] = MemoryQueue.get()
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(0.00000001, (steps - agent.counter) / steps * learning_rate)
            updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
            if updated:
                # print(' %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.train_net.get_weights()
                weight_dict['update'] += 1


        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            agent.train_net.save_weights(os.path.join(saveWeightFolder, 'model-BreakoutDeterministic-v4-%d.h5' % (agent.counter,)), overwrite=True)

def makeMyDir(myPath):
    if not os.path.exists(myPath):
        os.makedirs(myPath)

class ActorCritic(object):
    global arguments
    def __init__(self, action_space, screen=(84, 84), n_step=8, discount=0.99):
        self.screen = screen
        self.input_depth = 1
        self.past_range = arguments["LookBackFramesNumber"]
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.value_net, self.policy_net, self.load_net, adv = build_network(self.observation_shape, action_space.n)

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)
        self.last_observations = np.zeros_like(self.observations)


        self.n_step_observations = deque(maxlen=n_step)
        self.n_step_actions = deque(maxlen=n_step)
        self.n_step_rewards = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount
        self.counter = 0

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def reset(self):
        self.counter = 0
        self.n_step_observations.clear()
        self.n_step_actions.clear()
        self.n_step_rewards.clear()

    def SaveToMemory(self, action, reward, observation, terminal, MemoryQueue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        # reward /= 1


        self.n_step_observations.appendleft(self.last_observations)
        self.n_step_actions.appendleft(action)
        self.n_step_rewards.appendleft(reward)


        self.counter += 1
        if terminal or self.counter >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(self.counter):
                r = self.n_step_rewards[i] + self.discount * r
                MemoryQueue.put((self.n_step_observations[i], self.n_step_actions[i], r))
            self.reset()

    # def choose_epsilon(self): #TODO make eps for getAction


    def getAction(self):
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(resize(data, self.screen))[None, ...]


def Training(MemoryQueue, weight_dict, no):
    global arguments
    # name = datetime.now()
    # date_time = now.strftime("%m_%d%_Y%-H_%M_%S")
    # targetPath = os.path.join(rootdir, date_time + " Episode " + str(i_episode)+  "Target.pth")
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,' + \
                                 'compiledir=th_comp_act_' + str(no)

    print(' %5d> Process started' % (pid,))

    env = gym.make('BreakoutDeterministic-v4')
    frames = 0
    batch_size = arguments["BatchSize"]
    agent = ActorCritic(env.action_space, n_step=arguments["N-Steps"])

    if frames > 0:
        print(' %5d> Loaded weights from file' % (pid,))
        agent.load_net.load_weights('model-BreakoutDeterministic-v4-%d.h5' % ( frames))
    else:
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        print(' %5d> Loaded weights from dict' % (pid,))

    best_score = 0
    avg_score = deque([0], maxlen=25)
    all_rewards = []

    last_update = 0
    print("Starting Program")
    csvPath = os.path.join(rootdir, str(no) + "Rewards.csv")
    episode_count = 0
    # global noOpsPenalty
    while True:
        # noOpsPenalty = 0
        numberTimeRun = 0
        done = False
        episode_reward = 0
        noops = 0
        op_last, op_count = 0, 0
        observation = env.reset()
        agent.init_episode(observation)


        print(" %5d> , Thread %5d Episode "% (pid, no,) + str(episode_count) + " is starting")
        while not done:
            numberTimeRun += 1

            if episode_count % arguments["RenderEvery"] == 0 and no == 0: #only first thread every 50 eps
                env.render()

            frames += 1
            action = agent.getAction()
            #check for noops
            if action == 0:
                noops += 1
            else:
                noops = 0
            if noops > 100:
                print("########## Didn't Fire (probably) or just noOps over 100 times ##########")
                # noOpsPenalty = 1
                break

            if arguments["AutoFire"] and numberTimeRun == 0:
                #auto fire for first shot
                action = 1


            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            best_score = max(best_score, episode_reward)


            agent.SaveToMemory(action, reward, observation, done, MemoryQueue)


            op_count = 0 if op_last != action else op_count + 1
            done = done or op_count >= 100 #this is also noOps
            op_last = action



            if frames % 2000 == 0:
                print(' %5d> Best: %4d; Avg: %6.2f; Max: %4d' % (
                    pid, best_score, np.mean(avg_score), np.max(avg_score)))

            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    # print(' %5d> Getting weights from dict' % (pid,))
                    agent.load_net.set_weights(weight_dict['weights'])


        avg_score.append(episode_reward)
        all_rewards.append(episode_reward)
        episode_count += 1
        #CHANGED
        msg = 'Episode %5d at Frame %8d> Reward: %4d; NoOps: %8d/%8d  Average Score: %6.2f; Max Score: %4d \n' % (episode_count, frames, episode_reward, noops,numberTimeRun, np.mean(avg_score), np.max(avg_score))


        writeOneLiner(msg, no)
        print()

        rewardList = appendCSV(no, all_rewards, csvPath)
        all_rewards = [] #flush rubbish

        if episode_count % 50 == 0:
            print("Plotting Reward Curve")
            renderRewardGraphFromDF(no, rewardList, frames, episode_count)

            # renderRewardGraph(name, all_rewards)
            # saveCSV(name, rewards_list)


            # renderRewardGraphFromDF(no, rewardList, frames , episode_count)


            # print("Doing LRP Analysis")
            # checkLRP(agent.policy_net)
        elif episode_count < 50:
            print("Plotting Reward Curve < 50")
            renderRewardGraphFromDF(no, rewardList, frames, episode_count)



def writeOneLiner(msg, name):
    f=open(os.path.join(rootdir, str(name) +"Thread_ContinuousReporting.txt"), "a+")
    f.write(msg)
    f.close()

flattenList = lambda l: [item for sublist in l for item in sublist]


def appendCSV(name, rewards_list, csvPath):
    print(rewards_list)
    if os.path.isfile(csvPath):
        print("CSV File Exists - Append")
        data = pd.read_csv(csvPath, index_col=0)
        # print(data.head())
        # print("read val")
        readVals = data.values.tolist()
        # print(readVals)
        if any(isinstance(i, list) for i in readVals):
            oldRewards = flattenList(readVals)
        else:
            oldRewards = readVals
        # print("OLD REWARDS#########################")
        # print(oldRewards)
        fullRewardList = oldRewards + rewards_list
    else:
        fullRewardList = rewards_list
    print("Saving CSV")


    df_new = pd.DataFrame(fullRewardList)
    df_new.to_csv(csvPath)

    return fullRewardList


def saveCSV(name, rewards_list):
    print("Printing CSV")
    df = pd.DataFrame(rewards_list)
    csvPath = os.path.join(rootdir, str(name) + "Rewards.csv")
    df.to_csv(csvPath)



def renderRewardGraphFromDF(name, rewardList, frames, episode_count):
    # rewardList = rewardListDF.values.tolist()[0]

    print("Plotting Reward Curve for Episode: " + str(len(rewardList)))
    plt.figure(2)
    plt.clf()
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(list(range(0, len(rewardList))), rewardList)

    print("Saving Figure")

    if episode_count > 49:
        imgPath = os.path.join(rootdir, "RewardsGraph_Episode" + str(len(rewardList)) + "_Thread " + str(name) +".PNG")
        plt.savefig(imgPath)



    GeneralPath = os.path.join(rootdir, "General Rewards Graph for Thread " + str(name) +".PNG")

    plt.savefig(GeneralPath)

    print("\nCalculate Average Reward")
    if len(rewardList) > 100:
        rwdsList= rewardList[-100:]
        averageLast100Eps = statistics.mean(rwdsList)
        msg = str(episode_count) + " Episodes_" + str(frames) + " Frames: Average reward over last 100 Episodes: " + str(averageLast100Eps) +"\n"
    else:
        rwdsList = rewardList
        averageLast100Eps = statistics.mean(rwdsList)
        msg = str(episode_count) + " Episodes_" + str(frames) + " Frames: Average reward over all Episodes: " + str(averageLast100Eps) +"\n"
    # print(msg)
    print("Write to file")
    f=open(os.path.join(rootdir, str(name) +"Thread_averageRewards.txt"), "a+")
    f.write(msg)
    f.close()

    return


def renderRewardGraph(name, rewardList):
    print("Plotting Reward Curve for Episode: " + str(len(rewardList)))
    plt.figure(2)
    plt.clf()
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(list(range(0, len(rewardList))), rewardList)

    print("Saving Figure")

    imgPath = os.path.join(rootdir, "RewardsGraph_Episode" + str(len(rewardList)) + "_Thread " + str(name) +".PNG")
    plt.savefig(imgPath)

    print("Calculate Average Reward")
    if len(rewardList) > 100:
        rwdsList= rewardList[-100:]
        msg = "\nAverage reward over last 100 Episodes: "
    else:
        rwdsList = rewardList
        msg = "\nAverage reward over all Episodes: "

    averageLast100Eps = statistics.mean(rwdsList)
    print("Write to file")
    f=open(os.path.join(rootdir, str(name) +"Thread_averageRewards.txt"), "a+")
    f.write(msg + str(averageLast100Eps))
    f.close()

    return


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():

    global arguments
    arguments = {}
    arguments["NumberProcesses"] = 4
    arguments["LearningRate"] = 0.001
    arguments["LRDecayRate"] = 80000000
    arguments["BatchSize"] = 64
    arguments["SwapRate"] = 100
    arguments["SavePolicyEvery"] = 100000
    arguments["UseSavedCheckpoint"] = 0
    arguments["ExperienceQueueSize"] = 64
    arguments["N-Steps"] = 10
    arguments["beta"] = 0.01
    arguments["AutoFire"] = True
    arguments["RenderEvery"] = 50
    arguments["LookBackFramesNumber"] = 8


    global rootdir
    maindir = str(sys.path[0])
    rootdir = os.path.join(maindir,"Results")
    makeMyDir(rootdir)
    global saveWeightFolder
    saveWeightFolder = os.path.join(rootdir, "saved_weights")
    makeMyDir(saveWeightFolder)
    manager = Manager()
    weight_dict = manager.dict()
    MemoryQueue = manager.Queue(arguments["ExperienceQueueSize"])

    pool = Pool(arguments["NumberProcesses"] + 1, init_worker)

    try:
        for i in range(arguments["NumberProcesses"]):
            pool.apply_async(Training, (MemoryQueue, weight_dict, i))
        pool.apply_async(Learning, (MemoryQueue, weight_dict))
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()







if __name__ == "__main__":
    main()
