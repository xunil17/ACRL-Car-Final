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
from agent_helper import Training_Metadata, Decay_Explore_Rate, Basic_Learning_Rate, Replay_Memory, document_parameters
from car_environment import CarEnvironment


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class CarAgent:

    def __init__(self, batch_size, memory_capacity, num_episodes, learning_rate_drop_frame_limit,
            target_update_frequency, discount = 0.99, delta = 1, model_name = None):
    
        self.env = CarEnvironment()
        self.architecture = NeuralNet()
        self.explore_rate = Decay_Explore_Rate()
        self.learning_rate = Basic_Learning_Rate()
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name
        self.log_path = self.model_path + '/log'


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

        # Tf placeholders
        self.state_tf = tf.placeholder(shape=self.state_shape, dtype=tf.float32, name='state_tf')
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='action_tf')
        self.y_tf = tf.placeholder(dtype=tf.float32, name='y_tf')
        self.alpha = tf.placeholder(dtype=tf.float32, name='alpha')
        self.test_score = tf.placeholder(dtype=tf.float32, name='test_score')
        self.avg_q = tf.placeholder(dtype=tf.float32, name='avg_q')

        # Keep track of episode and frames
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



if __name__ == '__main__':
    parameters = {
    'target_update_frequency': 1000,
    'batch_size': 32, 
    'memory_capacity': 50000, 
    'num_episodes': 3000,
    'learning_rate_drop_frame_limit': 250000}

    try:
        name = sys.argv[1]
    except IndexError as err:
        raise Exception("Did you include model name when running? Ex: python3 car_agent.py testone")


    car_agent = CarAgent(model_name=name, **parameters)
