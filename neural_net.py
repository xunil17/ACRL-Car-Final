import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np

class NeuralNet:
	# input and action_size are tf.placeholders
    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=32, kernel_size=[8,8],
            strides=[4,4], padding='same', activation=tf.nn.relu,  
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1_out')
        layer2_out = tf.layers.conv2d(layer1_out, filters=64, kernel_size=[4,4],
            strides=[2,2], padding='same', activation=tf.nn.relu,  
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out')
        layer3_out = tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu,  
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out')
        layer4_out = tf.nn.dropout(tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu), .7, name='layer4_out')
        output =  tf.layers.dense(layer4_out, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "Architecture used in the nature paper in 2015 with dropout on dense layers, 0.7 keep prob."