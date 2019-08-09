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



# -----
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent',
                    dest='processes', type=int)
parser.add_argument('--lr', default=0.001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--steps', default=80000000, help='Number of frames to decay learning rate', dest='steps', type=int)
parser.add_argument('--batch_size', default=64, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--swap_freq', default=100, help='Number of frames before swapping network weights',
                    dest='swap_freq', type=int)
parser.add_argument('--checkpoint', default=0, help='Frame to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_freq', default=100000, help='Number of frames before saving weights', dest='save_freq',
                    type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--n_step', default=5, help='Number of steps', dest='n_step', type=int)
parser.add_argument('--reward_scale', default=1., dest='reward_scale', type=float)
parser.add_argument('--beta', default=0.01, dest='beta', type=float)
# -----
args = parser.parse_args()


# -----


def build_network(input_shape, output_shape):
    print("Building Network")
    from keras.models import Model
    from keras.layers import Input, Conv2D, Flatten, Dense
    # -----
    state = Input(shape=input_shape)
    h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(state)
    h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)

    value = Dense(1, activation='linear', name='value')(h)
    policy = Dense(output_shape, activation='softmax', name='policy')(h)

    value_network = Model(inputs=state, outputs=value)
    policy_network = Model(inputs=state, outputs=policy)

    advantage = Input(shape=(1,))
    train_network = Model(inputs=[state, advantage], outputs=[value, policy])

    return value_network, policy_network, train_network, advantage


def policy_loss(advantage=0., beta=0.01):
    from keras import backend as K

    def loss(y_true, y_pred):
        # return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
        #        beta * K.sum(y_pred * K.log(y_pred + K.epsilon())) #+ noOpsPenalty
        loss = -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))

        return loss

    return loss


def value_loss():
    from keras import backend as K

    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))

    return loss


# -----

class LearningAgent(object):
    def __init__(self, action_space, batch_size=32, screen=(84, 84), swap_freq=200):
        from keras.optimizers import RMSprop
        # -----
        self.screen = screen
        self.input_depth = 1
        self.past_range = 8
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen
        self.batch_size = batch_size

        _, _, self.train_net, advantage = build_network(self.observation_shape, action_space.n)

        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
                               loss=[value_loss(), policy_loss(advantage, args.beta)])

        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.entropy = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.n))
        self.counter = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        # print("Learning Begins")
        import keras.backend as K
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames
        # -----
        values, policy = self.train_net.predict([last_observations, self.unroll])
        # -----
        self.targets.fill(0.)
        advantage = rewards - values.flatten()
        self.targets[self.unroll, actions] = 1.
        # -----
        loss = self.train_net.train_on_batch([last_observations, advantage], [rewards, self.targets])
        entropy = np.mean(-policy * np.log(policy + 0.00000001))
        self.pol_loss.append(loss[2])
        self.val_loss.append(loss[1])
        self.entropy.append(entropy)
        self.values.append(np.mean(values))
        min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)
        print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
              '--- Value-Loss: %10.6f; Avg: %10.6f '
              '--- Entropy: %7.6f; Avg: %7.6f '
              '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f' % (
                  self.counter,
                  loss[2], np.mean(self.pol_loss),
                  loss[1], np.mean(self.val_loss),
                  entropy, np.mean(self.entropy),
                  min_val, max_val, avg_val), end='')
        # -----
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False


def learn_proc(mem_queue, weight_dict):
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0.3,' + \
                                 'compiledir=th_comp_learn'
    # -----
    print(' %5d> Learning process' % (pid,))
    # -----
    save_freq = args.save_freq
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    steps = args.steps
    # -----
    env = gym.make('BreakoutDeterministic-v4')
    agent = LearningAgent(env.action_space, batch_size=args.batch_size, swap_freq=args.swap_freq)
    # -----
    if checkpoint > 0:
        maindir = str(Path().absolute())
        pathForSavedWeights = os.path.join(maindir, "saved_weights", 'model-BreakoutDeterministic-v4-%d.h5' % (checkpoint,))
        print(' %5d> Loading weights from file %s' % (pid, pathForSavedWeights))
        agent.train_net.load_weights(pathForSavedWeights)
        # -----
    print(' %5d> Setting weights in dict' % (pid,))
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    print(' %5d> Finish Setting Weights' % (pid,))
    # -----
    last_obs = np.zeros((batch_size,) + agent.observation_shape)
    actions = np.zeros(batch_size, dtype=np.int32)
    rewards = np.zeros(batch_size)
    # -----
    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:
        # -----
        last_obs[idx, ...], actions[idx], rewards[idx] = mem_queue.get()
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(0.00000001, (steps - agent.counter) / steps * learning_rate)
            updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
            if updated:
                # print(' %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.train_net.get_weights()
                weight_dict['update'] += 1
        # -----
        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            agent.train_net.save_weights(os.path.join(saveWeightFolder, 'model-BreakoutDeterministic-v4-%d.h5' % (agent.counter,)), overwrite=True)

def makeMyDir(myPath):
    if not os.path.exists(myPath):
        os.makedirs(myPath)

class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84), n_step=8, discount=0.99):
        self.screen = screen
        self.input_depth = 1
        self.past_range = 8
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.value_net, self.policy_net, self.load_net, adv = build_network(self.observation_shape, action_space.n)

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)
        self.last_observations = np.zeros_like(self.observations)
        # -----
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

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        # reward /= args.reward_scale
        # -----
        self.n_step_observations.appendleft(self.last_observations)
        self.n_step_actions.appendleft(action)
        self.n_step_rewards.appendleft(reward)
        # -----
        self.counter += 1
        if terminal or self.counter >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(self.counter):
                r = self.n_step_rewards[i] + self.discount * r
                mem_queue.put((self.n_step_observations[i], self.n_step_actions[i], r))
            self.reset()

    def choose_action(self):
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(resize(data, self.screen))[None, ...]


def generate_experience_proc(mem_queue, weight_dict, no):
    # name = datetime.now()
    # date_time = now.strftime("%m_%d%_Y%-H_%M_%S")
    # targetPath = os.path.join(rootdir, date_time + " Episode " + str(i_episode)+  "Target.pth")
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,' + \
                                 'compiledir=th_comp_act_' + str(no)
    # -----
    print(' %5d> Process started' % (pid,))
    # -----
    frames = 0
    batch_size = args.batch_size
    # -----
    env = gym.make('BreakoutDeterministic-v4')
    agent = ActingAgent(env.action_space, n_step=args.n_step)

    if frames > 0:
        print(' %5d> Loaded weights from file' % (pid,))
        agent.load_net.load_weights('model-BreakoutDeterministic-v4-%d.h5' % ( frames))
    else:
        import time
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

        # -----
        print(" %5d> Episode "% (pid,) + str(len(all_rewards)) + " is starting")
        while not done:
            numberTimeRun += 1

            # if len(all_rewards) % 25 == 0:
            #     env.render()

            frames += 1
            action = agent.choose_action()
            #check for noops
            if action == 0:
                noops += 1
            else:
                noops = 0
            if noops > 100:
                print("########## Didn't Fire (probably) or just noOps over 100 times ##########")
                # noOpsPenalty = 1
                break

            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            best_score = max(best_score, episode_reward)
            # -----
            agent.sars_data(action, reward, observation, done, mem_queue)
            # -----
            op_count = 0 if op_last != action else op_count + 1
            done = done or op_count >= 100 #this is also noOps
            op_last = action
            # -----

            if frames % 2000 == 0:
                print(' %5d> Best: %4d; Avg: %6.2f; Max: %4d' % (
                    pid, best_score, np.mean(avg_score), np.max(avg_score)))

            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    # print(' %5d> Getting weights from dict' % (pid,))
                    agent.load_net.set_weights(weight_dict['weights'])
        # -----
        avg_score.append(episode_reward)
        all_rewards.append(episode_reward)
        episode_count = len(all_rewards)
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


        #test
        # print("Plotting Reward Curve")
        # name = no
        #
        # csvPath = os.path.join(rootdir, str(name) + "Rewards.csv")
        # rewardListDF = appendCSV(name, all_rewards, csvPath)
        # renderRewardGraphFromDF(name, rewardListDF, frames , episode_count)
        # print("finish render")


def writeOneLiner(msg, name):
    f=open(os.path.join(rootdir, str(name) +"Thread_ContinuousReporting.txt"), "a+")
    f.write(msg)
    f.close()

def appendCSV(name, rewards_list, csvPath):
    if os.path.isfile(csvPath):
        print("CSV File Exists - Append")
        data = pd.read_csv(csvPath)
        oldRewards = data.values.tolist()[0]
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
    global rootdir
    maindir = str(Path().absolute())
    rootdir = os.path.join(maindir,"Results")
    makeMyDir(rootdir)
    global saveWeightFolder
    saveWeightFolder = os.path.join(rootdir, "saved_weights")
    makeMyDir(saveWeightFolder)
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)

    pool = Pool(args.processes + 1, init_worker)

    try:
        for i in range(args.processes):
            pool.apply_async(generate_experience_proc, (mem_queue, weight_dict, i))

        pool.apply_async(learn_proc, (mem_queue, weight_dict))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
