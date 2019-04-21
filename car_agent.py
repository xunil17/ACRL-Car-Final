import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import tensorflow as tf
import random
import os
import subprocess
import time
from tensorflow.python.saved_model import tag_constants

from neural_net import NeuralNet
from agent_helper import Training_Metadata, Decay_Explore_Rate, Basic_Learning_Rate, Replay_Memory, document_parameters, Basic_Explore_Rate
from car_environment import CarEnvironment
import argparse


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class CarAgent:

    def __init__(self, batch_size, memory_capacity, num_episodes, learning_rate_drop_frame_limit,
            target_update_frequency, discount = 0.99, delta = 1, model_name = None, visualize = False):
    
        self.env = CarEnvironment(seed = [104, 106, 108])
        self.architecture = NeuralNet()
        self.explore_rate = Basic_Explore_Rate()
        self.learning_rate = Basic_Learning_Rate()
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name
        self.log_path = self.model_path + '/log'
        self.visualize = visualize


        self.initialize_tf_variables()

        self.target_update_frequency = target_update_frequency
        self.discount = discount
        self.replay_memory = Replay_Memory(memory_capacity, batch_size)
        self.training_metadata = Training_Metadata(frame=0, frame_limit = learning_rate_drop_frame_limit, 
                                                    episode = 0, num_episodes=num_episodes)

        self.delta = delta
        document_parameters(self)

    # Description: Sets up tensorflow graph and other variables, only called internally
    # Parameters: None
    # Output: None
    def initialize_tf_variables(self):
        # Setting up game specific variables
        self.state_size = self.env.state_space_size
        self.action_size = self.env.action_space_size
        self.state_shape = self.env.state_shape
        self.q_grid = None

        # Tf placeholders - feeds data into neural net from outside
        self.state_tf = tf.placeholder(shape=self.state_shape, dtype=tf.float32, name='state_tf')
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='action_tf')
        self.y_tf = tf.placeholder(dtype=tf.float32, name='y_tf')
        self.alpha = tf.placeholder(dtype=tf.float32, name='alpha')
        self.test_score = tf.placeholder(dtype=tf.float32, name='test_score')
        self.avg_q = tf.placeholder(dtype=tf.float32, name='avg_q')

        # Keep track of episode and frames
        # Variables are used to store information about neural net
        self.episode = tf.Variable(initial_value=0, trainable=False, name='episode')
        self.frames = tf.Variable(initial_value=0, trainable=False, name='frames')
        self.increment_frames_op = tf.assign(self.frames, self.frames + 1, name='increment_frames_op')
        self.increment_episode_op = tf.assign(self.episode, self.episode + 1, name='increment_episode_op')

        # Operations
        # NAME                      DESCRIPTION                                         FEED DEPENDENCIES
        # Q_value                   Value of Q at given state(s)                        state_tf
        # Q_argmax                  Action(s) maximizing Q at given state(s)            state_tf
        # Q_amax                    Maximal action value(s) at given state(s)           state_tf
        # Q_value_at_action         Q value at specific (action, state) pair(s)         state_tf, action_tf
        # onehot_greedy_action      One-hot encodes greedy action(s) at given state(s)  state_tf
        self.Q_value = self.architecture.evaluate(self.state_tf, self.action_size)
        self.Q_argmax = tf.argmax(self.Q_value, axis=1, name='Q_argmax')
        self.Q_amax = tf.reduce_max(self.Q_value, axis=1, name='Q_max')
        self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1, name='Q_value_at_action')
        self.onehot_greedy_action = tf.one_hot(self.Q_argmax, depth=self.action_size)

        # Training related
        # NAME                          FEED DEPENDENCIES
        # loss                          y_tf, state_tf, action_tf
        # train_op                      y_tf, state_tf, action_tf, alpha

        # self.loss = tf.losses.mean_squared_error(self.y_tf, self.Q_value_at_action)
        self.loss = tf.losses.huber_loss(self.y_tf, self.Q_value_at_action)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.train_op = self.optimizer.minimize(self.loss, name='train_minimize')

        # Tensorflow session setup
        self.saver = tf.train.Saver(max_to_keep=None)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = False
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        self.trainable_variables = tf.trainable_variables()
        print (self.trainable_variables)

        # Tensorboard setup
        self.writer = tf.summary.FileWriter(self.log_path)
        self.writer.add_graph(self.sess.graph)
        test_score = tf.summary.scalar("Training score", self.test_score, collections=None, family=None)
        avg_q = tf.summary.scalar("Average Q-value", self.avg_q, collections=None, family=None)
        self.training_summary = tf.summary.merge([avg_q])
        self.test_summary = tf.summary.merge([test_score])
        # subprocess.Popen(['tensorboard', '--logdir', self.log_path])

        # Initialising variables and finalising graph
        self.sess.run(tf.global_variables_initializer())
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

        self.sess.graph.finalize()

    # Description: Performs one step of batch gradient descent on the DDQN loss function. 
    # Parameters:
    # - alpha: Number, the learning rate 
    # Output: None
    def experience_replay(self, alpha):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.get_mini_batch(self.training_metadata)
        y_batch = [None] * self.replay_memory.batch_size
        fixed_feed_dict = {self.state_tf: next_state_batch}
        fixed_feed_dict.update(zip(self.trainable_variables, self.fixed_target_weights))

        greedy_actions = self.sess.run(self.onehot_greedy_action, feed_dict={self.state_tf: next_state_batch})
        fixed_feed_dict.update({self.action_tf: greedy_actions})
        Q_batch = self.sess.run(self.Q_value_at_action, feed_dict=fixed_feed_dict)

        y_batch = reward_batch + self.discount * np.multiply(np.invert(done_batch), Q_batch)

        feed = {self.state_tf: state_batch, self.action_tf: action_batch, self.y_tf: y_batch, self.alpha: alpha}
        self.sess.run(self.train_op, feed_dict=feed)

    # Description: Updates the weights of the target network
    # Parameters:   None
    # Output:       None
    def update_fixed_target_weights(self):
        self.fixed_target_weights = self.sess.run(self.trainable_variables)


    # Trains the model
    def train(self):
        while self.sess.run(self.episode) < self.training_metadata.num_episodes:

            #basically grapb the episode number from the neural net
            episode = self.sess.run(self.episode)
            self.training_metadata.increment_episode()
            # increments the episode in the neural net
            self.sess.run(self.increment_episode_op)

            # set up car environment
            state_lazy = self.env.reset()
            self.env.render()

            done = False
            epsilon = self.explore_rate.get(self.training_metadata)
            alpha = self.learning_rate.get(self.training_metadata)

            print("Episode {0}/{1} \t Epsilon: {2} \t Alpha: {3}".format(episode, self.training_metadata.num_episodes, epsilon, alpha))
            print ("Replay Memory: %d" % self.replay_memory.length())
            episode_frame = 0

            max_reward = float('-inf')

            while not done:

                # Update target weights every update frequency
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                # Choose and perform action and update replay memory
                action = self.get_action(np.array(state_lazy), epsilon)
                next_state_lazy, reward, done, info = self.env.step(action)

                if self.visualize:
                    self.env.render()

                episode_frame += 1

                self.replay_memory.add(self, state_lazy, action, reward, next_state_lazy, done)

                # Train with replay memory if populated
                if self.replay_memory.length() > 10 * self.replay_memory.batch_size:
                    self.sess.run(self.increment_frames_op)
                    self.training_metadata.increment_frame()
                    self.experience_replay(alpha)

                state_lazy = next_state_lazy
                done = info['true_done']

                abs_reward = self.env.get_total_reward()
                max_reward = max(max_reward, abs_reward)
                if max_reward - abs_reward > 5:
                    done = True

                if done:
                    print("Episode reward:", abs_reward)





    # Description: Chooses action wrt an e-greedy policy. 
    # Parameters:
    # - state:      Tensor representing a single state
    # - epsilon:    Number in (0,1)
    # Output:       Integer in the range 0...self.action_size-1 representing an action
    def get_action(self, state, epsilon):
        # Performing epsilon-greedy action selection
        if random.random() < epsilon:
            return self.env.sample_action_space()
        else:
            return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: [state]})[0]

    # Description: Tests the model
    # Parameters:
    # - num_test_episodes:  Integer, giving the number of episodes to be tested over
    # - visualize:          Boolean, gives whether should render the testing gameplay
    def test(self, num_test_episodes, visualize):
        rewards = []
        for episode in range(num_test_episodes):
            done = False
            state_lazy = self.env.reset(test=True)
            state = np.array(state_lazy)
            episode_reward = 0
            if not visualize:
                self.test_env.render()
            while not done:
                if visualize:
                    self.env.render()
                action = self.get_action(state, epsilon=0)
                next_state_lazy, reward, done, info = self.env.step(action, test=True)
                state = np.array(next_state_lazy)
                episode_reward += reward
                done = info['true_done']
            rewards.append(episode_reward)
        return np.mean(rewards), np.std(rewards), rewards

    # Description: Returns average Q-value over some number of fixed tracks
    # Parameters:   None
    # Output:       None
    def estimate_avg_q(self):
        if not self.q_grid:
            return 0
        return np.average(np.amax(self.sess.run(self.Q_value, feed_dict={self.state_tf: self.q_grid}), axis=1))

    # Description: Loads a model trained in a previous session
    # Parameters:
    # - path:   String, giving the path to the checkpoint file to be loaded
    # Output:   None
    def load(self, path):
        self.saver.restore(self.sess, path)


if __name__ == '__main__':
    parameters = {
    'target_update_frequency': 1000, # number of frames between each target Q update
    'batch_size': 32, # define size of mini-batch
    'memory_capacity': 50000,  #capacity of replay memory
    'num_episodes': 3000, # number of training episodes
    'learning_rate_drop_frame_limit': 250000 #number of frames exploration rate decays over
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="model store folder")
    parser.add_argument("--vis", help="do visualization", action="store_true")
    args = parser.parse_args()

    car_agent = CarAgent(model_name=args.model_name, **parameters, visualize = args.vis)
    ########################### Train Model ##########################

    car_agent.train()

    ########################### Test Model ##########################3
    # car_agent.load("/home/sean/RL-2018/src/DQN_Agent/models/no_conv/data.chkp-1471")
    # car_agent.test(5, True)