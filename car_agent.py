import sys
sys.dont_write_bytecode = True

import numpy as np
import random
import os
import argparse
import glob

import gym
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from neural_net import NeuralNet
from agent_helper import Training_Metadata, Decay_Explore_Rate, Basic_Learning_Rate, Replay_Memory, document_parameters, Basic_Explore_Rate
from car_environment import CarEnvironment


class CarAgent:

    def __init__(self, batch_size, memory_capacity, num_episodes, learning_rate_drop_frame_limit,
            target_update_frequency, seeds = [104, 106, 108], discount = 0.99, delta = 1, model_name = None, visualize = False):
    
        self.env = CarEnvironment(seed = seeds)
        self.architecture = NeuralNet(self.env.state_space_size, self.env.action_space_size, self.env.state_shape)
        self.explore_rate = Basic_Explore_Rate()
        self.learning_rate = Basic_Learning_Rate()
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name
        self.log_path = self.model_path + '/log'
        self.visualize = visualize
        self.damping_mult = 1

        self.initialize_tf_variables()

        self.target_update_frequency = target_update_frequency
        self.discount = discount
        self.replay_memory = Replay_Memory(memory_capacity, batch_size)
        self.training_metadata = Training_Metadata(frame=0, frame_limit = learning_rate_drop_frame_limit, episode = 0, num_episodes=num_episodes)

        self.delta = delta
        document_parameters(self)

    # sets up tensorflow graph - called in setup
    def initialize_tf_variables(self):
        # Tensorflow session setup
        self.architecture.build()
        self.q_grid = None
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
        test_score = tf.summary.scalar("Training score", self.architecture.test_score, collections=None, family=None)
        # avg_q = tf.summary.scalar("Average Q-value", self.architecture.avg_q, collections=None, family=None)
        # self.training_summary = tf.summary.merge([avg_q])
        self.test_summary = tf.summary.merge([test_score])

        # Initialising variables and finalising graph
        self.sess.run(tf.global_variables_initializer())
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

        self.sess.graph.finalize()

    # Performs one step of batch gradient descent on the DDQN loss function. 
    # alpha = learning rate 
    def experience_replay(self, alpha):

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.get_mini_batch(self.training_metadata)
        
        # get argmax of q-network
        greedy_actions = self.sess.run(self.architecture.onehot_greedy_action, feed_dict={self.architecture.state_tf: next_state_batch})

        y_batch = [None] * self.replay_memory.batch_size
        fixed_feed_dict = {self.architecture.state_tf: next_state_batch, self.architecture.action_tf: greedy_actions}
        fixed_feed_dict.update(zip(self.trainable_variables, self.fixed_target_weights))


        Q_batch = self.sess.run(self.architecture.Q_value_at_action, feed_dict=fixed_feed_dict)

        y_batch = reward_batch + self.discount * np.multiply(np.invert(done_batch), Q_batch)

        feed = {self.architecture.state_tf: state_batch, self.architecture.action_tf: action_batch, self.architecture.y_tf: y_batch, self.architecture.alpha: alpha}
        self.sess.run(self.architecture.train_op, feed_dict=feed)

    # Updates weights of target network
    def update_fixed_target_weights(self):
        self.fixed_target_weights = self.sess.run(self.trainable_variables)


    # Trains the model
    def train(self, imitation = False):
        while self.sess.run(self.architecture.episode) < self.training_metadata.num_episodes:

            #basically grapb the episode number from the neural net
            episode = self.sess.run(self.architecture.episode)
            self.training_metadata.increment_episode()
            # increments the episode in the neural net
            self.sess.run(self.architecture.increment_episode_op)

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

            while True:

                # Update target weights every update frequency
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                # Choose and perform action and update replay memory

                if random.random() < epsilon:
                    if imitation:
                        action = self.get_oracle_action(self.env)
                    else:
                        action = self.env.sample_action_space()
                else:
                    action = self.get_action(np.array(state_lazy), 0)


                next_state_lazy, reward, done, info = self.env.step(action)

                if self.visualize:
                    self.env.render()

                episode_frame += 1

                self.replay_memory.add(self, state_lazy, action, reward, next_state_lazy, done)

                # Train with replay memory if populated
                if self.replay_memory.length() > 10 * self.replay_memory.batch_size:
                    self.sess.run(self.architecture.increment_frames_op)
                    self.training_metadata.increment_frame()
                    self.experience_replay(alpha)
                    self.q_grid = self.replay_memory.get_q_grid(size=200, training_metadata=self.training_metadata)

                # avg_q = self.estimate_avg_q()

                state_lazy = next_state_lazy
                done = info['true_done']

                abs_reward = self.env.get_total_reward()
                max_reward = max(max_reward, abs_reward)
                
                if max_reward - abs_reward > 5 or done:
                    print("Episode reward:", abs_reward)
                    break

            # Saving tensorboard data and model weights
            if (episode % 30 == 0) and (episode != 0):
                score, std, rewards = self.test(num_test_episodes=5, visualize=True)
                print('{0} +- {1}'.format(score, std))
                self.writer.add_summary(self.sess.run(self.test_summary,
                                                      feed_dict={self.architecture.test_score: score}), episode / 30)
                self.saver.save(self.sess, self.model_path + '/data.chkp', global_step=self.training_metadata.episode)

                file = open(self.model_path + '/trainlog.txt', "a+")
                printstr = '%f %f %f %f %f \n' % (score, std, episode, alpha, epsilon)
                file.write(printstr)
                file.close()

            # self.writer.add_summary(self.sess.run(self.training_summary, feed_dict={self.architecture.avg_q: avg_q}), episode)
            # self.writer.add_summary(self.sess.run(self.training_summary), episode)

    # Chooses action wrt an e-greedy policy. 
    # - state      Tensor representing a single state
    # - epsilon    Number in (0,1)
    # Output       Integer in the range 0...self.action_size-1 representing an action
    def get_action(self, state, epsilon):
        # Performing epsilon-greedy action selection
        if random.random() < epsilon:
            return self.env.sample_action_space()
        else:
            return self.sess.run(self.architecture.Q_argmax, feed_dict={self.architecture.state_tf: [state]})[0]

    def get_oracle_action(self, env):
        env = env.env
        a = 4

        car_x = env.car.hull.position[0]
        car_y = env.car.hull.position[1]
        car_angle = -env.car.hull.angle
        car_vel = np.linalg.norm(env.car.hull.linearVelocity)

        target_seg = 0
        for i in range(len(env.road)):
            if not env.road[i].road_visited:
                target_seg = min(i + 3, len(env.road) - 1)
                break

        target_loc = env.nav_tiles[target_seg]
        #env.highlight_loc = target_loc
        angle_to = np.arctan2(target_loc[0] - car_x, target_loc[1] - car_y) - car_angle
        angle_to = (angle_to + 2 * np.pi) % (2 * np.pi)

        if angle_to > np.pi:
            angle_to -= 2*np.pi

        vel_err = 35 - car_vel
        if vel_err > 2:
            a = 2
        
        if angle_to < -0.15 * self.damping_mult:
            a = 0

        if angle_to > 0.15 * self.damping_mult:
            a = 1

        if a == 4:
            self.damping_mult /= 1.5
            self.damping_mult = max(self.damping_mult, 1)
        else:
            self.damping_mult *= 1.2

        return a

    # Tests the model
    def test(self, num_test_episodes, visualize):
        rewards = []
        for episode in range(num_test_episodes):
            done = False
            state_lazy = self.env.reset(test=True)
            input()
            state = np.array(state_lazy)
            episode_reward = 0
            max_reward = float('-inf')
            while not done:
                if visualize:
                    self.env.render()
                action = self.get_action(state, epsilon=0)
                next_state_lazy, reward, done, info = self.env.step(action, test=True)
                state = np.array(next_state_lazy)
                episode_reward += reward

                if(self.env.env.t > 30):
                    print("Ended due to time limit")
                    done = True

            rewards.append(episode_reward)
            print(episode_reward)
        return np.mean(rewards), np.std(rewards), rewards

    # # average Q-value over some number of fixed tracks
    # def estimate_avg_q(self):
    #     if self.q_grid:
    #         return np.average(np.amax(self.sess.run(self.architecture.Q_value, feed_dict={self.architecture.state_tf: self.q_grid}), axis=1))
    #     else:
    #         return None

    # loads a model trained in a previous session
    # - path:   String, giving the path to the checkpoint file to be loaded
    def load(self, path):
        self.saver.restore(self.sess, path)


if __name__ == '__main__':
    parameters = {
    'target_update_frequency': 1000, # number of frames between each target Q update
    'batch_size': 32, # define size of mini-batch
    'memory_capacity': 50000,  #capacity of replay memory
    'num_episodes': 3000, # number of training episodes
    'learning_rate_drop_frame_limit': 250000, #number of frames exploration rate decays over
    # 'seeds': random.sample(range(1,200),50)
    # 'seeds': [108]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="model store folder")
    parser.add_argument("--vis", help="do visualization", action="store_true")
    parser.add_argument("--test", help="do testing", action="store_true")
    parser.add_argument("--load", help="load previous model file", action="store_true")
    parser.add_argument("--im", help="use imitation learning", action="store_true")
    args = parser.parse_args()

    car_agent = CarAgent(model_name=args.model_name, **parameters, visualize = args.vis)
    ########################### Train Model ##########################

    if not args.test:
        if args.load:
                list_of_files = glob.glob(car_agent.model_path + '/*data-*') # find all files in the model folder
                latest_file = max(list_of_files, key=os.path.getctime) #sort by newest
                k = latest_file.rfind(".")
                chkp_file = latest_file[:k]
                print("---------Loading file---------------", chkp_file)
                car_agent.load(chkp_file)

        car_agent.train(imitation = args.im)
    else:
        list_of_files = glob.glob(car_agent.model_path + '/*data-*') # find all files in the model folder
        latest_file = max(list_of_files, key=os.path.getctime) #sort by newest
        k = latest_file.rfind(".")
        chkp_file = latest_file[:k]

        print("---------Loading file---------------", chkp_file)

        car_agent.load(chkp_file)
        score, std, rewards = car_agent.test(50, args.vis)
        print('{0} +- {1}'.format(score, std))