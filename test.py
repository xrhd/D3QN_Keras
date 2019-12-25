import os
# for keras the CUDA commands must come before importing the keras libraries
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import gym
from gym import wrappers
import numpy as np
from src.d3qn_keras import D3QNAgent
# from keras_radam import RAdam
from utils.utils import plotLearning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    d3qn_agent = D3QNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=0.0,
                           batch_size=64, input_dims=8, fname='model/d3qn_model_radam.h5')
    d3qn_agent.load_model()
    score = 0
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = d3qn_agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        observation = observation_
        env.render()
    print('socore:', score)