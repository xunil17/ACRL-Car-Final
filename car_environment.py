import gym
import numpy as np
import random
from image_process import process_image
import cv2
import sys
import time

class CarEnvironment:

    # Inputs:
    # stacked_frames is how many frames in a row to use - atari used four
    # seed is list of seeds to sample from during training, input a list of numbers ex: [103, 104, 105]

    def __init__(self, seed=None, stacked_frames=4, flip=False):
        self.name = "ACRL-Car-Racing"
        self.env = gym.make('CarRacing-v0')
        self.stacked_frames = stacked_frames
        self.image_dimension = [96,96]

        self.state_space_size = stacked_frames * np.prod(self.image_dimension)
        self.action_space_size = 5
        self.state_shape = [None, self.stacked_frames] + list(self.image_dimension)

        self.history = []
        self.action_dict = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
        self.seed = seed
        self.flip = flip
        self.flip_episode = False

    def render(self):
        self.env.render()

    def step(self, action):
        action = self.map_action(action)
        total_reward = 0
        n = random.choice([2,3,4])
        for _ in range(n):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            info = {'true_done': done}
            if done:
                break

        processed_next_state = self.process(next_state)
        return processed_next_state, total_reward, done, info

    def process(self, state):
        self.add_history(state)
        if len(self.history) < self.stacked_frames:
            zeros = np.zeros(self.image_dimension)
            result = np.tile(zeros, ((self.stacked_frames - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result


    def add_history(self, state):
        if len(self.history) >= self.stacked_frames:
            self.history.pop(0)
        processed_image = process_image(state, flip=self.flip_episode)
        print (processed_image.shape)
        cv2.imshow('image', processed_image)
        self.history.append(processed_image)
        

    def reset(self):
        if self.seed:
            self.env.seed(random.choice(self.seed))
        self.flip_episode = random.random() > 0.5 and self.flip
        # self.flip_episode = True
        return self.process(self.env.reset())

    # returns single integer representing action (1-5)
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    # return control u list based on action integer
    def map_action(self,action):
        if self.flip_episode and action <= 1:
            action = 1 - action
        return self.action_dict[action]

    def close(self):
        self.env.close()

    def __str__(self):
        return self.name + '\nseed: {0}'.format(self.seed)

if __name__ == '__main__':
    env = CarEnvironment(flip = True)
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.sample_action_space())
        cv2.waitKey()
    # cv2.waitKey()
    env.close()
    # print ('hello')


