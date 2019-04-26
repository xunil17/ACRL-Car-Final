import sys
sys.dont_write_bytecode = True

import tensorflow as tf

class NeuralNet:
    def __init__(self, state_size, action_size, state_shape):
        # Setting up game specific variables
        self.state_size = state_size
        self.action_size = action_size
        self.state_shape = state_shape
	# input and action_size are tf.placeholders
    def build(self):
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

        layer1_out = tf.layers.conv2d(self.state_tf, filters=32, kernel_size=[8,8],
            strides=[4,4], padding='same', activation=tf.nn.relu,  
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1_out')
        layer2_out = tf.layers.conv2d(layer1_out, filters=64, kernel_size=[4,4],
            strides=[2,2], padding='same', activation=tf.nn.relu,  
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out')
        layer3_out = tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu,  
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out')
        layer4_out = tf.nn.dropout(tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu), .7, name='layer4_out')
        self.Q_value =  tf.layers.dense(layer4_out, self.action_size, activation=None, name='output')

        # Operations
        # Q_value                   Value of Q at given state(s)                        state_tf
        # Q_argmax                  Action(s) maximizing Q at given state(s)            state_tf
        # Q_amax                    Maximal action value(s) at given state(s)           state_tf
        # Q_value_at_action         Q value at specific (action, state) pair(s)         state_tf, action_tf
        # onehot_greedy_action      One-hot encodes greedy action(s) at given state(s)  state_tf
        # self.Q_value = self.architecture.build()
        self.Q_argmax = tf.argmax(self.Q_value, axis=1, name='Q_argmax')
        self.Q_amax = tf.reduce_max(self.Q_value, axis=1, name='Q_max')
        self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1, name='Q_value_at_action')
        self.onehot_greedy_action = tf.one_hot(self.Q_argmax, depth=self.action_size)

        # Training related
        # loss                          y_tf, state_tf, action_tf
        # train_op                      y_tf, state_tf, action_tf, alpha
        self.loss = tf.losses.huber_loss(self.y_tf, self.Q_value_at_action)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.train_op = self.optimizer.minimize(self.loss, name='train_minimize')

        return self.Q_value



    def __str__(self):
        return "nature 2015, dropout, 0.7 prob"