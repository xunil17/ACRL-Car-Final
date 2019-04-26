import gym
import numpy as np
import random
from image_process import process_image
from collections import deque
import cv2
import car_racing
# import sys
# import time

class CarEnvironment:

    # Inputs:
    # stacked_frames is how many frames in a row to use - atari used four
    # seed is list of seeds to sample from during training, input a list of numbers ex: [103, 104, 105]

    def __init__(self, seed=None, stacked_frames=4, flip=False, domain_seed = None):
        self.name = "ACRL-Car-Racing"
        #self.env = gym.make('CarRacing-v0')
        self.env = car_racing.CarRacing()

        self.stacked_frames = stacked_frames
        self.image_dimension = [96,96]

        self.state_space_size = stacked_frames * np.prod(self.image_dimension)
        self.action_space_size = 5
        self.state_shape = [None, self.stacked_frames] + list(self.image_dimension)

        self.frames = deque([], maxlen = stacked_frames)
        #prepopulate frames with zeros
        for _ in range(0,self.stacked_frames):
            zeros = np.zeros(self.image_dimension)
            self.frames.append(zeros)

        self.action_dict = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
        self.seed = seed
        self.flip = flip
        self.flip_episode = False
        self.domain_seed = domain_seed

    # shows windows for each image in lazy frame (usually processed next state)
    def show_images_lazy(self, lazyframe):
        frame_np = np.array(lazyframe)
        for num in range(0,self.stacked_frames):
            cv2.imshow('image' + str(num), frame_np[num])

    #renders environment
    def render(self):
        self.env.render()

    # takes one step with action as input (u), returns processed next state which is LazyFrame with four frames in it
    def step(self, action, test=False):
        action = self.map_action(action)
        total_reward = 0
        # n = random.choice([2,3,4])
        n = 1 if test else random.choice([2,3,4])
        for j in range(n):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            info = {'true_done': done}
            if done:
                break

        #processed_next_state is lazy frame with numpy array of 4x96x96
        processed_next_state = self.process(next_state)
        # self.show_images_lazy(processed_next_state)
        return processed_next_state, total_reward, done, info

    # adds last state to history array
    def process(self, state):
        processed_image = process_image(state, flip=self.flip_episode)
        self.frames.append(processed_image)

        frame = LazyFrames(list(self.frames))
        # frame_np = np.array(frame)
        # print (frame_np.shape)

        return frame
      

    def reset(self, test=False):
        if self.seed:
            self.env.seed(random.choice(self.seed))

        if test:
            self.env.seed() #set to random seed at test time

        self.flip_episode = random.random() > 0.5 and not test and self.flip
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

    def get_total_reward(self):
        return self.env.reward

    def __str__(self):
        return self.name + '\nseed: {0}'.format(self.seed)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            
            # self._out = np.concatenate(self._frames, axis=-1)
            self._out = np.stack(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]

if __name__ == '__main__':
    env = CarEnvironment(flip = True)
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.sample_action_space())
        cv2.waitKey()
    # cv2.waitKey()
    env.close()
    # print ('hello')


