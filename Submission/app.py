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
        maindir = str(Path().absolute())
        global FourThreadModelCheckpoint
        pathForSavedWeights = os.path.join(maindir, "trained_weights", 'model-BreakoutDeterministic-v4-%d.h5' % (checkpoint,))
        if arguments["UseSavedCheckpoint"] == FourThreadModelCheckpoint:
            pathForSavedWeights = os.path.join(str(Path().absolute()), "trained_weights", 'model-4threads.h5')
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


class TestingActor(object):
    global arguments
    def __init__(self, action_space, screen=(84, 84)):
        self.screen = screen
        self.input_depth = 1
        self.past_range = arguments["LookBackFramesNumber"]
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        _, self.policy_net, self.load_net, adv = build_network(self.observation_shape, action_space.n)

        # self.value_net.compile(optimizer='rmsprop', loss='mse')
        # self.policy_net.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.load_net.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.
        # self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss

        self.action_space = action_space
        self.observations = np.zeros((self.input_depth * self.past_range,) + screen)
        # self.observations = np.zeros(self.observation_shape)
        # self.last_observations = np.zeros_like(self.observations)
        #
        #
        # self.n_step_observations = deque(maxlen=n_step)
        # self.n_step_actions = deque(maxlen=n_step)
        # self.n_step_rewards = deque(maxlen=n_step)
        # self.n_step = n_step
        # self.discount = discount
        # self.counter = 0

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def getAction(self, observation):
        self.save_observation(observation)
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        policy /= np.sum(policy)  # numpy, why?
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
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


def main(NumberProcesses = 4, LearningRate = 0.001, LRDecayRate = 80000000, BatchSize = 64, SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = 0, ExperienceQueueSize = 64, N_steps = 10, beta = 0.01, autoFire = True, RenderEvery = 50, LookBackFrames = 8):

    global arguments
    arguments = {}
    arguments["NumberProcesses"] = NumberProcesses
    arguments["LearningRate"] = LearningRate
    arguments["LRDecayRate"] = LRDecayRate
    arguments["BatchSize"] = BatchSize
    arguments["SwapRate"] = SwapRate
    arguments["SavePolicyEvery"] = SavePolicyEvery
    if UseSavedCheckpoint > 0:
        arguments["UseSavedCheckpoint"] = UseSavedCheckpoint
    else:
        arguments["UseSavedCheckpoint"] = 0
    arguments["ExperienceQueueSize"] = ExperienceQueueSize
    arguments["N-Steps"] = N_steps
    arguments["beta"] = beta
    arguments["AutoFire"] = autoFire
    if RenderEvery > 0:
        arguments["RenderEvery"] = RenderEvery
    else:
        arguments["RenderEvery"] = 50
    arguments["LookBackFramesNumber"] = LookBackFrames


    global rootdir
    maindir = str(Path().absolute())
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

# if __name__ == "__main__":
#     main()



###HERE UPDATE 4 Thread Model
def play(NumberProcesses = 4, LearningRate = 0.001, LRDecayRate = 80000000, BatchSize = 64, SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = 10000000, ExperienceQueueSize = 64, N_steps = 10, beta = 0.01, autoFire = True, RenderEvery = 50, LookBackFrames = 8):
    # framesToTest = int(framesToTestInput.value)
    global FourThreadModelCheckpoint
    global arguments
    arguments = {}
    arguments["NumberProcesses"] = NumberProcesses
    arguments["LearningRate"] = LearningRate
    arguments["LRDecayRate"] = LRDecayRate
    arguments["BatchSize"] = BatchSize
    arguments["SwapRate"] = SwapRate
    arguments["SavePolicyEvery"] = SavePolicyEvery
    # arguments["UseSavedCheckpoint"] = UseSavedCheckpoint
    arguments["UseSavedCheckpoint"] = FourThreadModelCheckpoint
    arguments["ExperienceQueueSize"] = ExperienceQueueSize
    arguments["N-Steps"] = N_steps
    arguments["beta"] = beta
    arguments["AutoFire"] = autoFire
    if RenderEvery > 0:
        arguments["RenderEvery"] = RenderEvery
    else:
        arguments["RenderEvery"] = 50
    arguments["LookBackFramesNumber"] = LookBackFrames
    global rootdir
    rootdir = os.path.join(str(Path().absolute()), "Results", "EvaluationDir")
    makeMyDir(rootdir)
    env = gym.make('BreakoutDeterministic-v4')
    agent = TestingActor(env.action_space)
    # model_file = arguments["UseSavedCheckpoint"]
    agent.load_net.load_weights(os.path.join(str(Path().absolute()), "trained_weights", 'model-4threads.h5'))
    game = 1
    framesToPlay = int(framesToTestInput.value)
    frames = 0
    for _ in range(framesToPlay):
        all_rewards =[]
        done = False
        episode_reward = 0
        noops = 0
        # init game
        observation = env.reset()
        agent.init_episode(observation)
        # play one game
        print('Game #%8d; ' % (game,), end='')
        while not done:
            frames += 1
            env.render()
            action = agent.getAction(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # ----
            if action == 0:
                noops += 1
            else:
                noops = 0
            if noops > 100:
                break

        print('Reward %5d; ' % (episode_reward,))
        all_rewards.append(episode_reward)
        msg = 'Episode %5d has Reward: %4d; \n' % (game, episode_reward,)
        writeOneLiner(msg, 1)
        csvPath = os.path.join(rootdir, "Rewards.csv")
        rewardList = appendCSV(1, all_rewards, csvPath)
        all_rewards = [] #flush rubbish
        if game >= framesToPlay:
            print("Plot Reward Curve")
            renderRewardGraphFromDF(1, rewardList, frames, game)

        game += 1



def trainAppliedParams():

    global FourThreadModelCheckpoint
    if bool(UseSavedCheckpointInput.value): #should be true or false:
        if int(NumberProcessesInput.value) == 4:
            checkpoint = FourThreadModelCheckpoint #TODO change according to saved_weights for 4 threads  ###HERE UPDATE 4 Thread Model
            main(NumberProcesses = 4, LearningRate = float(LearningRateInput.value), LRDecayRate = 80000000, BatchSize = 64, SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = checkpoint, ExperienceQueueSize = 64, N_steps = 10, beta = 0.01, autoFire = True, RenderEvery = int(renderEveryXEpisodesEntry.value), LookBackFrames = 8)
        elif int(NumberProcessesInput.value) == 16:
            checkpoint = 18300000 # according to saved_weights
            main(NumberProcesses = 16, LearningRate = float(LearningRateInput.value), LRDecayRate = 80000000, BatchSize = 20, SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = checkpoint, ExperienceQueueSize = 256, N_steps = 5, beta = 0.01, autoFire = bool(autoFireInput.value), RenderEvery = int(renderEveryXEpisodesEntry.value), LookBackFrames = 3)
        else:
            checkpoint = 0
            main(NumberProcesses = int(NumberProcessesInput.value), LearningRate = float(LearningRateInput.value), LRDecayRate = 80000000, BatchSize = int(batchSizeInput.value), SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = checkpoint, ExperienceQueueSize = int(ExperienceQueueSizeInput.value), N_steps = int(N_stepsInput.value), beta = 0.01, autoFire = bool(autoFireInput.value), RenderEvery = int(renderEveryXEpisodesEntry.value), LookBackFrames = int(LookBackFramesInput.value))
    else:
        checkpoint = 0
        main(NumberProcesses = int(NumberProcessesInput.value), LearningRate = float(LearningRateInput.value), LRDecayRate = 80000000, BatchSize = int(batchSizeInput.value), SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = checkpoint, ExperienceQueueSize = int(ExperienceQueueSizeInput.value), N_steps = int(N_stepsInput.value), beta = 0.01, autoFire = bool(autoFireInput.value), RenderEvery = int(renderEveryXEpisodesEntry.value), LookBackFrames = int(LookBackFramesInput.value))


        # setParameters(batch_size = float(batchSizeInput.value), gamma = float(gammaInput.value), eps_start = float(eps_startInput.value), eps_end = float(eps_endInput.value), eps_decay = float(eps_decayInput.value), learning_rate = float(learning_rateInput.value))

from guizero import App,Window, PushButton, Slider, Text, TextBox, Picture, Box, CheckBox

global FourThreadModelCheckpoint
FourThreadModelCheckpoint = 10000000


def open_window():
    window.show(wait=True)

def close_window():
    window.hide()
    trainAppliedParams()
##### App stuff Start
app = App(title="Breakout", layout="grid")

window = Window(app, title="Set Parameters for Training")
window.hide()

# text = Text(window, text="This text will show up in the second window")

open_button = PushButton(app, text="Set Parameter", command=open_window, grid = [0,0])
close_button = PushButton(window, text="Train", command=close_window)


# app = App(title="Breakout A3C")
build_a_snowman = app.yesno("Training", "Do you want to use the default parameters?")
if build_a_snowman == False:
    app.info("Setting Parameters Info", "Default values are shown in entry box, if checkpoint is used then default values for that checkpoint will be used as necessary.")
else:
    app.info("Info", "Default Parameters as specificed in Report: NumberProcesses = 4, LearningRate = 0.001, LRDecayRate = 80000000, BatchSize = 64, SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = 0, ExperienceQueueSize = 64, N_steps = 10, beta = 0.01, autoFire = True, RenderEvery = 50, LookBackFrames = 8")

### prompt if run with default (main()) or set parameters main(put params here)
### Run Train (put warning that gui will freeze + button launches main())


#Set parameters
# main(NumberProcesses = 4, LearningRate = 0.001, LRDecayRate = 80000000, BatchSize = 64, SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = 0, ExperienceQueueSize = 64, N_steps = 10, beta = 0.01, autoFire = True, RenderEvery = 50, LookBackFrames = 8):



col1 = Box(window, layout="grid", grid=[0,1], align="top")

buttonTrain = PushButton(col1, text= "Train", command=main, grid=[0,0])

NumberProcessesLabel = Text(col1, text="Number of Threads:", grid=[0,1]) #NumberProcesses
LearningRateLabel = Text(col1, text="Learning Rate: ", grid=[0,2]) #LearningRate
BatchSizeLabel = Text(col1, text="Batch Size", grid=[0,3]) #batch_size
ExperienceQueueSizeLabel = Text(col1, text="Experience Queue Size", grid=[0,4]) #experience Queue Size
N_stepsLabel = Text(col1, text="N Steps", grid=[0,5]) #n steps
LookBackFramesLabel = Text(col1, text="Look at the past _ frames", grid=[0,6]) #n steps
renderEveryXEpisodesLabel = Text(col1, text="Render Every _ Episode:", grid=[0,7]) #RenderEvery
autoFireLabel = Text(col1, text="Auto Fire at Start", grid=[0,8]) #auto fire
UseSavedCheckpointLabel = Text(col1, text="Use Saved Weights (Pre-trained)", grid=[0,9]) #used trained weight

NumberProcessesInput = TextBox(col1, text="4", grid=[1,1]) #NumberProcesses
LearningRateInput = TextBox(col1, text="0.001", grid=[1,2]) #LearningRate
BatchSizeInput = TextBox(col1, text="64", grid=[1,3]) #batch_size
ExperienceQueueSizeInput = TextBox(col1, text="64", grid=[1,4]) #experience Queue Size
N_stepsInput = TextBox(col1, text="20", grid=[1,5]) #n steps
LookBackFramesInput = TextBox(col1, text="8", grid=[1,6]) #n steps
renderEveryXEpisodesEntry = TextBox(col1, text="50", grid=[1,7]) #RenderEvery
autoFireInput = CheckBox(col1, text="Auto Fire" , grid=[1,8]) #auto fire
UseSavedCheckpointInput = CheckBox(col1, text="Only for processes = 4 or processes = 16" , grid=[1,9]) #used trained weight

# renderEveryXEpisodesEntry.repeat(2000, setRenderEpisodes)
# buttonSetTrain = PushButton(col1, text= "Train", command=trainAppliedParams, grid=[1,10])

# CheckBox(app, text="Add extra glitter")

#fourth column
col4 = Box(app, layout="grid", grid=[0,0], align="top")
buttonTrain = PushButton(col4, text= "Train with Default Parameters - 4 Threads, Look back 8 Frames", grid=[0,0], command=main) #starts training with default params
# buttonTest = PushButton(col4, text= "Train with Default Parameters", command=StartTesting, grid=[0,0], command=main) #starts training with default params

buttonTest = PushButton(col4, text= "Test for X frames - 4 Threads, Look back 8 Frames", grid=[0,1], command=play) #test for
framesToTestLabel = Text(col4, text="Test for _ Episodes:", grid=[0,2])
framesToTestInput = TextBox(col4, text="10", grid=[1,2])

app.display()
