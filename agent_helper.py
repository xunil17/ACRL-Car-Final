import sys
sys.dont_write_bytecode = True

import numpy as np
import random

class Training_Metadata:

    def __init__(self, frame=0, frame_limit=1000000, episode=0, num_episodes=10000):
        self.frame = frame
        self.frame_limit = frame_limit
        self.episode = episode
        self.num_episodes = num_episodes

    def increment_frame(self):
        self.frame += 1

    def increment_episode(self):
        self.episode += 1


# The explore rate decays from 0.6 to 0.1 linearly over the half of the
# number of episodes defined in the training_metadata and stays at 0.1
# thereafter
class Basic_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, 0.5* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))

    def __str__(self):
        return 'max(0.1, 0.5* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))'


class Decay_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))'

# The learning rate is fixed to 0.00025 constantly
class Basic_Learning_Rate:

    def get(self, training_metadata):
        return 0.00025

    def __str__(self):
        return '0.00025'

class Replay_Memory:

    # To initialize the replay memory
    # memory_capacity defines the limit for storage
    # batch_size defines the size for the minibatch
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []

    # To output the filled size of memory
    def length(self):
        return len(self.memory)
        
    # To randomly sample a minibatch for update
    def get_mini_batch(self, *args, **kwargs):
        mini_batch = random.sample(self.memory, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]
        done_batch = [data[4] for data in mini_batch]
        state_batch = [np.array(item) for item in state_batch]
        next_state_batch = [np.array(item) for item in next_state_batch]
        # print (done_batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    # To add an episode into the memory
    # agent defines the environment
    # state defines the past stack of images
    # action defines the action taken
    # reward defines the reward returned based on the state and action
    # next_state defines the current stack of images
    def add(self, agent, state_lazy, action, reward, next_state_lazy, done):
        # convert the action to one hot action for easier computation
        one_hot_action = np.zeros(agent.action_size)
        one_hot_action[action] = 1
        self.memory.append((state_lazy, one_hot_action, reward, next_state_lazy, done))
        # delete the earliest episode if memory is full
        if (len(self.memory) > self.memory_capacity):
            self.memory.pop(0)

    # To randomly sample states for evaluating average Q value during 
    # training.
    # size defines the size to sample.
    def get_q_grid(self, size, *args, **kwargs):
        return [data[0] for data in random.sample(self.memory, size)]


# Creates a txt file storing model parameters
# Parameters:
# - agent: An object of type DQN_Agent
def document_parameters(agent):
    # document parameters
    with open(agent.model_path + '/params.txt', 'w') as file:
        file.write('Architecture: ' + str(agent.architecture) + '\n')
        file.write('Explore Rate: ' + str(agent.explore_rate) + '\n')
        file.write('Learning Rate: ' + str(agent.learning_rate) + '\n')
        file.write('Discount: ' + str(agent.discount) + '\n')
        file.write('Batch Size: ' + str(agent.replay_memory.batch_size) + '\n')
        file.write('Memory Capacity: ' + str(agent.replay_memory.memory_capacity) + '\n')
        file.write('Num Episodes: ' + str(agent.training_metadata.num_episodes) + '\n')
        file.write('Learning Rate Drop Frame Limit: ' + str(agent.training_metadata.frame_limit) + '\n')