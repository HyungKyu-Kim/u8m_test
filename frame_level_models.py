# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

from tensorflow.python.ops import init_ops
from rcn import dynamic_rcn, GruRcnCell

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    
    isp = model_input.get_shape().as_list()
    print model_input
    input = tf.reshape(model_input, shape=[isp[1], -1, isp[2]])
    print input
    print "hkkk num_frames ", num_frames
    print "hkkk input.shape", input.shape

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

VGG_MEAN = [103.939, 116.779, 123.68]

class GruRcn:

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # Convert RGB to BGR
#         red, green, blue = tf.unstack(self.data, axis=4)
#         bgr = tf.stack([
#             blue - VGG_MEAN[0],
#             green - VGG_MEAN[1],
#             red - VGG_MEAN[2],
#         ], 4)
        self.data_dict = None
        self.seq_length = num_frames
        train_mode = True
                
        isp = model_input.get_shape().as_list()
        print "isp: ", isp
        print "num_frames: ",num_frames
        input = tf.reshape(model_input, shape=[isp[1], -1, 1, isp[2], 1])
        print input
        
        self.conv1 = self.conv_layer(input, 1, 8, "conv1")
#         slim.conv2d(inputs=input, num_outputs=8, kernel_size=[1, 3])
        print ("hkkkk self.conv1: ",self.conv1.shape)
        self.pool1 = self.max_pool(self.conv1, 'pool1')
        
        self.conv2 = self.conv_layer(self.pool1, 8, 16, "conv2")
        self.pool2 = self.max_pool(self.conv2, 'pool2')

        self.conv3 = self.conv_layer(self.pool2, 16, 32, "conv3")
        self.pool3 = self.max_pool(self.conv3, 'pool3')
        print ("hkkkk self.pool3: ",self.pool3.shape)
        self.rcn4 = self.rcn_layer(self.pool3, 32, 32, "rcn4")
        print ("hkkkk self.rcn4: ",self.rcn4.shape)
        self.pool4 = self.max_pool(self.rcn4, 'pool4')
        print "hkkk self.pool4: ",self.pool4
        self.rcn5 = self.rcn_layer(self.pool4, 32, 32, "rcn5")
        self.rcn5_lastframe = self.last_frame_layer(self.rcn5, "rcn5_lastframe")
        self.pool5 = self.max_single_pool(self.rcn5_lastframe, 'pool5')
        print "hkkk self.pool5: ",self.pool5
        
        fc_isp = self.pool5.get_shape().as_list()
        print fc_isp
        
        x = tf.reshape(self.pool5, [-1, fc_isp[1]*fc_isp[2]*fc_isp[3]])
        output = slim.fully_connected(
        x, 1024, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(1e-8))
#         self.fc6 = self.fc_layer(self.pool5, fc_isp[1]*fc_isp[2]*fc_isp[3], 512, "fc6")
#         self.relu6 = tf.nn.relu(self.fc6)
        print "hkkkkk output: ",output
        aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=output,
            vocab_size=vocab_size,
            **unused_params)
#         self.relu6 = tf.cond(1, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)

#         self.fc_7 = self.fc_layer(self.relu6, 512, 4716, "fc_7")

#         self.prob = tf.nn.softmax(self.fc_7, name="prob")

#         del self.data_dict
#         print "self.prob: ", self.prob
#         return {"predictions": self.prob}

    def last_frame_layer(self, bottom, name):
        number = tf.range(0, tf.shape(self.seq_length)[0])
        indexs = tf.stack([self.seq_length - 1, number], axis=1)
        return tf.gather_nd(bottom, indexs, name)

    def max_pool(self, bottom, name):
        with tf.variable_scope(name):
            def _inner_max_pool(bott):
                return tf.nn.max_pool(bott,
                                      ksize=[1, 1, 2, 1],
                                      strides=[1, 1, 2, 1],
                                      padding='SAME',
                                      name=name)

            bottoms = tf.unstack(bottom, axis=0)
            output = tf.stack([_inner_max_pool(bott) for bott in bottoms], axis=0)

            return output

    def max_single_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
#             filt, conv_biases = self.get_conv_var(1, 3, in_channels, out_channels, name)
#             filt = tf.truncated_normal([1, 3, in_channels, out_channels], 0.0, 0.01)
            filt = [1, 3]
            def _inner_conv(bott):
                return slim.conv2d(inputs=bott, num_outputs=out_channels, kernel_size=filt)
#                 conv = tf.nn.conv2d(bott, filt, [1, 1, 1, 1], padding='SAME')
#                 bias = tf.nn.bias_add(conv, conv_biases)
#                 relu = tf.nn.relu(bias)
#                 return relu

            bottoms = tf.unstack(bottom, axis=0)
            output = tf.stack([_inner_conv(bott) for bott in bottoms], axis=0)

            return output

    def conv_single_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(1, 3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def rcn_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            _, _, N, H, C = bottom.get_shape().as_list()
            input_size = (N, H, C)
            nb_filter = out_channels
            dict_name = name.replace("rcn", "conv")
            weight_initializers = {}
            if self.data_dict is not None and dict_name in self.data_dict:
                filters = self.data_dict[dict_name][0]
                biases = self.data_dict[dict_name][1]
                weight_initializers['WConv'] = init_ops.constant_initializer(filters)
                weight_initializers['c_biases'] = init_ops.constant_initializer(biases)
            cell = GruRcnCell(input_size, nb_filter, 1, 3, [1, 1, 1, 1], "SAME", 1, 3, weight_initializers=weight_initializers)
            output, _ = dynamic_rcn(cell, bottom, sequence_length=self.seq_length, dtype=tf.float32)
            #output, _ =  tf.nn.dynamic_rnn(cell, bottom, sequence_length=self.seq_length, dtype=tf.float32)
            return output

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size_h, filter_size_w, in_channels, out_channels, name):
        if self.data_dict is not None and name in self.data_dict:
            filters = self.get_var(self.data_dict[name][0], name + "_filters", False)
            biases = self.get_var(self.data_dict[name][1], name + "_biases", False)
        else:
            initial_filter = tf.truncated_normal([filter_size_h, filter_size_w, in_channels, out_channels], 0.0, 0.01)
            initial_bias = tf.ones([out_channels], dtype=tf.float32)
            filters = self.get_var(initial_filter, name + "_filters", True)
            biases = self.get_var(initial_bias, name + "_biases", True)
        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        if self.data_dict is not None and name in self.data_dict:
            weights = self.get_var(self.data_dict[name][0], name + "_weights", True)
            biases = self.get_var(self.data_dict[name][1], name + "_biases", True)
        else:
            initial_weight = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
            weights = self.get_var(initial_weight, name + "_weights", True)
            initial_bias = tf.ones([out_size], dtype=tf.float32)
            biases = self.get_var(initial_bias, name + "_biases", True)
        return weights, biases

    def get_var(self, initial_value, var_name, trainable):
        if trainable:
            var = tf.Variable(initial_value, name=var_name)
        else:
            var = tf.constant(initial_value, dtype=tf.float32, name=var_name)
        return var

'''
    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += functools.reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
'''