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
import signal
from PIL import Image

from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

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
    train_network = Model(inputs=state, outputs=[value, policy])
    print("Finish Building Model")
    return value_network, policy_network, train_network, advantage

def makeMyDir(myPath):
    if not os.path.exists(myPath):
        os.makedirs(myPath)

class TestingActor(object):
    global arguments
    def __init__(self, action_space, screen=(84, 84)):
        self.screen = screen
        self.input_depth = 1
        self.past_range = arguments["LookBackFramesNumber"]
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        _, self.policy_net, self.load_net, adv = build_network(self.observation_shape, action_space.n)

        self.load_net.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.


        self.action_space = action_space
        self.observations = np.zeros((self.input_depth * self.past_range,) + screen)

    def init_episode(self, observation):
        # print("Init Eps")
        for _ in range(self.past_range):
            self.save_observation(observation)

    def getAction(self, observation):
        self.save_observation(observation)
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        policy /= np.sum(policy)
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        # print("save observation")
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        img = resize(data, self.screen)
        gray = rgb2gray(img)[None, ...]
        # print(img)
        return gray

def writeOneLinerBug(name):
    f=open(os.path.join(rootdir, "Bug_Thread.txt"), "a+")
    f.write(name + " doesn't act")
    f.close()


def writeOneLiner(msg, name):
    f=open(os.path.join(rootdir, str(name) +"Thread_ContinuousReporting.txt"), "a+")
    f.write(msg)
    f.close()

def writeOneLinerAverageRewards(rewardList, name):
    f=open(os.path.join(rootdir, "Average Rewards.txt"), "a+")
    msg = str(name) + " has average rewards over 10 episodes: " + str(statistics.mean(rewardList)) + "\n"
    f.write(msg)
    f.close()

flattenList = lambda l: [item for sublist in l for item in sublist]


def appendCSV(name, rewards_list, csvPath):
    # print(rewards_list)
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



def renderRewardGraphFromDF(name, rewardList):
    # rewardList = rewardListDF.values.tolist()[0]

    print("Plotting Reward Curve for Episode: " + str(len(rewardList)))
    plt.figure(2)
    plt.clf()
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(list(range(0, len(rewardList))), rewardList)

    print("Saving Figure")

    # if episode_count > 49:
    #     imgPath = os.path.join(rootdir, "RewardsGraph_Episode" + str(len(rewardList)) + "_Thread " + str(name) +".PNG")
    #     plt.savefig(imgPath)



    GeneralPath = os.path.join(rootdir, "General Rewards Graph for Thread " + str(name) +".PNG")

    plt.savefig(GeneralPath)

    # print("\nCalculate Average Reward")
    # if len(rewardList) > 100:
    #     rwdsList= rewardList[-100:]
    #     averageLast100Eps = statistics.mean(rwdsList)
    #     msg = str(episode_count) + " Episodes_" + str(frames) + " Frames: Average reward over last 100 Episodes: " + str(averageLast100Eps) +"\n"
    # else:
    #     rwdsList = rewardList
    #     averageLast100Eps = statistics.mean(rwdsList)
    #     msg = str(episode_count) + " Episodes_" + str(frames) + " Frames: Average reward over all Episodes: " + str(averageLast100Eps) +"\n"
    # # print(msg)
    # print("Write to file")
    # f=open(os.path.join(rootdir, "Thread_averageRewards.txt"), "a+")
    # f.write(msg)
    # f.close()

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
    if len(rewardList) >= 5:
        rwdsList= rewardList[-5:]
        msg = "\nAverage reward over last 5 Episodes: "
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

    signal.signal(signal.SIGINT, signal.SIG_IGN)



def main(NumberProcesses = 4, LearningRate = 0.001, LRDecayRate = 80000000, BatchSize = 64, SwapRate = 100, SavePolicyEvery = 100000, UseSavedCheckpoint = 10000000, ExperienceQueueSize = 64, N_steps = 10, beta = 0.01, autoFire = True, RenderEvery = 50, LookBackFrames = 8):
    # framesToTest = int(framesToTestInput.value)
    # global FourThreadModelCheckpoint
    global arguments
    arguments = {}

    arguments["NumberProcesses"] = 16
    arguments["LearningRate"] = 0.001
    arguments["LRDecayRate"] = 80000000
    arguments["BatchSize"] = 20
    arguments["SwapRate"] = 100
    arguments["SavePolicyEvery"] = 100000
    arguments["UseSavedCheckpoint"] = 0
    arguments["ExperienceQueueSize"] = 256
    arguments["N-Steps"] = 5
    arguments["beta"] = 0.01
    arguments["AutoFire"] = False
    arguments["RenderEvery"] = 50
    arguments["LookBackFramesNumber"] = 3

    # arguments["NumberProcesses"] = NumberProcesses
    # arguments["LearningRate"] = LearningRate
    # arguments["LRDecayRate"] = LRDecayRate
    # arguments["BatchSize"] = BatchSize
    # arguments["SwapRate"] = SwapRate
    # arguments["SavePolicyEvery"] = SavePolicyEvery
    # # arguments["UseSavedCheckpoint"] = UseSavedCheckpoint
    # # arguments["UseSavedCheckpoint"] = FourThreadModelCheckpoint
    # arguments["ExperienceQueueSize"] = ExperienceQueueSize
    # arguments["N-Steps"] = N_steps
    # arguments["beta"] = beta
    # arguments["AutoFire"] = autoFire
    # if RenderEvery > 0:
    #     arguments["RenderEvery"] = RenderEvery
    # else:
    #     arguments["RenderEvery"] = 50
    # arguments["LookBackFramesNumber"] = LookBackFrames
    global rootdir
    rootdir = os.path.join(str(Path().absolute()), "Results", "EvaluationDir")

    makeMyDir(rootdir)
    global env
    env = gym.make('BreakoutDeterministic-v4')


def testCheckGrey(data, type, msg):
    img = Image.fromarray(data, type)
    img.show()
    img.save(msg + ".JPEG")



def testCheck(data):
    img = Image.fromarray(data, 'RGB').convert('L')
    img.save("TestCheck.PNG")
    img.show()

def testMatCheck(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()
    plt.savefig("TestMatCheck.PNG")

def convertGrey(data):
    img = Image.fromarray(data, 'RGB')
    # img.save("Loaded.PNG")
    img = img.convert('L')
    # img.save("Convert Grey.PNG")
    # img = img.convert('1')
    # img.save("Convert BW.PNG")
    data = np.array(img)

    return data


def preprocess(img):
    #downsample
    img = img[::2, ::2]
    # print(img.shape)

    if greyScale:
        img = convertGrey(img)
        img = FloatTensor(img).unsqueeze(2)
    else:
        img = FloatTensor(img)

    # print(img.shape)
    return img
    # env.close()

def saveThisImg(observation, imgPath):
    img = Image.fromarray(observation,'RGB')
    img.save(imgPath)


def testPlay(WeightsName):
    print("Test Playing begins")
    global rootdir
    global env
    global arguments
    global framesToPlay

    agent = TestingActor(env.action_space)
    # model_file = arguments["UseSavedCheckpoint"]
    print("Loading Weights")
    agent.load_net.load_weights(os.path.join(str(Path().absolute()), "trained_weights", WeightsName))
    print("Finish Loading Weights")
    # agent.load_net.load_weights(WeightsName)
    game = 1

    frames = 0
    global DoRender
    for _ in range(framesToPlay):
        actionList = []
        numberTimeRun = 0
        all_rewards =[]
        done = False
        episode_reward = 0
        noops = 0
        Acting = True
        # init game
        observation = env.reset()
        # testCheckGrey(observation, "RGB", "RGB")
        # testCheckGrey(agent.transform_screen(observation), "RGB", "RGB")
        # testCheckGrey(observation, "HSV", "HSV")
        # print("First Observation")
        agent.init_episode(observation)
        # play one game
        print('Game #%8d; ' % (game,), end='')
        while not done:
            frames += 1
            if DoRender:
                env.render()
            action = agent.getAction(observation)
            if arguments["AutoFire"] and numberTimeRun == 0:
                numberTimeRun += 1
                print("Auto-fire first shot")
                #auto fire for first shot
                action = 1
                if Acting:
                    #if twice in a row didn't do anything, quit
                    msg = WeightsName + " is bad - doesn't fire for ep twice in a row"
                    # print(msg)
                    writeOneLinerBug(msg)
                    done = True

                Acting = False
            elif frames % 50 == 0:
                print("Do action " + str(env.unwrapped.get_action_meanings()[action]))
                Acting = True

                if len(actionList) > 100:
                    if len(set(actionList))==1:
                        #all the actions are the same
                        msg = WeightsName + " is bad - always the same action: " + str(env.unwrapped.get_action_meanings()[action])
                        print(msg)
                        writeOneLinerBug(msg)
                        done = True
                        framesToPlay = 1
                        break;
            else:
                actionList.append(action)


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
        msg = str(WeightsName) +' Episode %5d has Reward: %4d with Frames %5d; \n' % (game, episode_reward, frames)
        writeOneLiner(msg, WeightsName)
        csvPath = os.path.join(rootdir, WeightsName +"_Rewards.csv")
        imgPath = os.path.join(rootdir, WeightsName + "_Episode" + str(game) +"_End.png")
        saveThisImg(observation, imgPath)
        # all_rewards = [] #flush rubbish
        if game >= framesToPlay:
            print("Plot Reward Curve")
            saveCSV(WeightsName, all_rewards)
            # rewardList = appendCSV(1, all_rewards, csvPath)
            # renderRewardGraphFromDF(WeightsName, all_rewards)
            writeOneLinerAverageRewards(all_rewards, WeightsName)

        game += 1



def runForAllWeights():
    global DoRender
    DoRender = True
    global framesToPlay
    framesToPlay = 5

    mypath = os.path.join(str(Path().absolute()), "trained_weights")
    main() #set params
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".h5")]
    # print(onlyfiles)
    onlyfiles.sort()
    print(onlyfiles)
    for WeightsName in onlyfiles:
        # if WeightsName.endswith(".h5"):
        print("######################## Playing ####################################")
        # print(WeightsName[WeightsName.find("v4-")+3:])
        print(WeightsName)
        testPlay(WeightsName)



if __name__ == "__main__":
    runForAllWeights()
