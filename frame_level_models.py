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
from rcn import dynamic_rcn, GruRcnCell, StackedGruRcnCell, dynamic_stacked_rcn

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
    print ("number_of_layers ",number_of_layers)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    print ("stacked_lstm ",stacked_lstm)
    print ("model_input ",model_input)
    print ("num_frames ",num_frames)
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)
    
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    print "state", state
    print "state[-1].h", state[-1].h
    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)

VGG_MEAN = [103.939, 116.779, 123.68]

class GruRcn:

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a GruRcn to represent the video.

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

        self.data_dict = None
        self.seq_length = num_frames
        self.__convKernelSize = None
        self.__poolKernelSize = None
        self.__poolStrideSize = None
        
        max_frames = model_input.get_shape().as_list()[1]   # 300
        inputWidth = model_input.get_shape().as_list()[2]   # 1024
        inputHeight = 1 
        inputChannels = 1

        # model_input [batch ,max frames, features] -> reCombinedInput [max frames, batch ,features]
        reCombinedInput = tf.stack(tf.unstack(model_input, axis=1))
        
        # reshape [max frames, batch ,features] -> [max frames, batch, 1, features, 1]
        reshapedInput = tf.reshape(reCombinedInput, shape=[max_frames, -1, inputHeight, inputWidth, inputChannels])

        if inputHeight == 1:
            self.__convKernelSize = [1, 3]
            self.__poolKernelSize = [1, 2]
            self.__poolStrideSize = [1, 2]
        else:
            self.__convKernelSize = [3, 3]
            self.__poolkernelSize = [2, 2]
            self.__poolStrideSize = [2, 2]
        
        outFilterSize = 4
         
        rcn0, state0 = self.rcn_layer(reshapedInput, outFilterSize, "rcn0")
#         avgpool3 = self.avg_pool_to_size1(state3, "avg0")
        fc0 = self.fc_layer(state0, vocab_size, "fc0")
        
        conv0 = self.conv_layer(reshapedInput, 1, outFilterSize, "conv0")
        pool0 = self.max_pool(conv0, 'pool0')
        rcnpool0 = self.max_pool(rcn0, 'rcnpool0')
#  
#         rcn1, state1 = self.rcn_layer(pool0, outFilterSize*2, "rcn1")
        rcn1, state1 = self.stacked_rcn_layer(pool0, rcnpool0, outFilterSize*2, "rcn1")
        fc1 = self.fc_layer(state1, vocab_size, "fc1")
        
#         conv1 = self.conv_layer(pool0, outFilterSize, outFilterSize*2, "conv1")
#         pool1 = self.max_pool(conv1, 'pool1')
#            
#         rcn2, state2 = self.rcn_layer(pool1, outFilterSize*4, "rcn2")
#         fc2 = self.fc_layer(state2, vocab_size, "fc2")

#         fcSum = tf.add_n(inputs=[fc0, fc1], name="fc_sum")
#         divSum = tf.div(fcSum, 2, name="divide")

        predict = tf.nn.softmax(fc0, name="softmax") 
        return {"predictions": predict}
#         aggregated_model = getattr(video_level_models,
#                                FLAGS.video_level_classifier_model)
#          
#         return aggregated_model().create_model(
#             model_input=x,
#             vocab_size=vocab_size,
#             **unused_params)
    
    def last_frame_layer(self, bottom, name):
        number = tf.range(0, tf.shape(self.seq_length)[0])
        indexs = tf.stack([self.seq_length - 1, number], axis=1)
        return tf.gather_nd(bottom, indexs, name)
    
    def avg_pool_to_size1(self, bottom, name):
        with tf.variable_scope(name):
            _, _bottomHeight, _bottomWidth, _ = bottom.get_shape().as_list()
            return tf.nn.avg_pool(value=bottom, 
                                  ksize=[1, _bottomHeight, _bottomWidth, 1], 
                                  strides=[1, _bottomHeight, _bottomWidth, 1], 
                                  padding='SAME',
                                  name=name)
            
    def max_pool(self, bottom, name):
        with tf.variable_scope(name):
            _kH, _kW = self.__poolKernelSize
            _sH, _sW = self.__poolStrideSize
            def _inner_max_pool(bott):
                return tf.nn.max_pool(bott,
                                      ksize=[1, _kH, _kW, 1],
                                      strides=[1, _sH, _sW, 1],
                                      padding='SAME',
                                      name=name)

            _bottoms = tf.unstack(bottom, axis=0)
            output = tf.stack([_inner_max_pool(bott) for bott in _bottoms], axis=0)

            return output

    def max_single_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
#             filt, conv_biases = self.get_conv_var(self.__convKernelSize, in_channels, out_channels, name)
            filter_size_h, filter_size_w = self.__convKernelSize
            
            filt = tf.get_variable(name=name + "_filters", shape=[filter_size_h, filter_size_w, in_channels, out_channels], initializer=init_ops.random_normal_initializer(stddev=0.01))
            conv_biases = tf.get_variable(name=name + "_biases", shape=[out_channels], initializer=init_ops.random_normal_initializer(stddev=0.01)) 

            def _inner_conv(bott):
                conv = tf.nn.conv2d(bott, filt, [1, 1, 1, 1], padding='SAME')
                bias = tf.nn.bias_add(conv, conv_biases)
                relu = tf.nn.relu(bias)
                return relu

            _bottoms = tf.unstack(bottom, axis=0)
            output = tf.stack([_inner_conv(bott) for bott in _bottoms], axis=0)

            return output

    def conv_single_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(self.__convKernelSize, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def rcn_layer(self, bottom, out_channels, name):
        with tf.variable_scope(name):
            _, _, N, H, C = bottom.get_shape().as_list()
            input_size = (N, H, C)
            num_outputs = out_channels
            dict_name = name.replace("rcn", "conv")
#             if self.data_dict is not None and dict_name in self.data_dict:
#                 filters = self.data_dict[dict_name][0]
#                 biases = self.data_dict[dict_name][1]
#                 weight_initializers['WConv'] = init_ops.constant_initializer(filters)
#                 weight_initializers['c_biases'] = init_ops.constant_initializer(biases)
            cell = GruRcnCell(input_size, num_outputs, self.__convKernelSize, [1, 1, 1, 1], "SAME", self.__convKernelSize)
#             cell = GruRcnCell(input_size, num_outputs, 1, 3, [1, 1, 1, 1], "SAME", 1, 3)
            output, state = dynamic_rcn(cell, bottom, sequence_length=self.seq_length, dtype=tf.float32)
            return output, state
        
    def stacked_rcn_layer(self, bottom, prevState, out_channels, name):
        with tf.variable_scope(name):
            _, _, N, H, C = bottom.get_shape().as_list()
            input_size = (N, H, C)
            num_outputs = out_channels
            dict_name = name.replace("rcn", "conv")
#             if self.data_dict is not None and dict_name in self.data_dict:
#                 filters = self.data_dict[dict_name][0]
#                 biases = self.data_dict[dict_name][1]
#                 weight_initializers['WConv'] = init_ops.constant_initializer(filters)
#                 weight_initializers['c_biases'] = init_ops.constant_initializer(biases)
            cell = StackedGruRcnCell(input_size, num_outputs, self.__convKernelSize, [1, 1, 1, 1], "SAME", self.__convKernelSize)
#             cell = GruRcnCell(input_size, num_outputs, 1, 3, [1, 1, 1, 1], "SAME", 1, 3)
            output, state = dynamic_stacked_rcn(cell, bottom, prevState, sequence_length=self.seq_length, dtype=tf.float32)
            return output, state

    def fc_layer(self, bottom, out_size, name):
        with tf.variable_scope(name):
            _, _height, _width, _channel = bottom.get_shape().as_list() 
            size = _height*_width*_channel
            weights = tf.get_variable(name=name + "_weights", shape = [size, out_size], initializer=init_ops.random_normal_initializer(stddev=0.01))
            biases = tf.get_variable(name=name + "_biases", shape=[out_size], initializer=init_ops.random_normal_initializer(stddev=0.01)) 
             
            x = tf.reshape(bottom, [-1, size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

        
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        if self.data_dict is not None and name in self.data_dict:
            filters = tf.constant(value=self.data_dict[name][0], name = name + "_filters")
            biases = tf.constant(value=self.data_dict[name][1], name = name + "_biases")
        else:
            filter_size_h, filter_size_w = filter_size
            initial_filter = tf.truncated_normal([filter_size_h, filter_size_w, in_channels, out_channels], 0.0, 0.01)
#             initial_bias = tf.ones([out_channels], dtype=tf.float32)
            filters = tf.get_variable(name=name + "_filters", initializer=initial_filter)
            biases = tf.get_variable(name=name + "_biases", shape=[out_channels], initializer=tf.constant_initializer(0.0))
        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        if self.data_dict is not None and name in self.data_dict:
            weights = tf.constant(value=self.data_dict[name][0], name=name + "_weights")
            biases = tf.constant(value=self.data_dict[name][1], name=name + "_biases")
        else:
            initial_weight = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
            weights = tf.get_variable(name=name + "_weights", initializer=initial_weight)
#             initial_bias = tf.ones([out_size], dtype=tf.float32)
            biases = tf.get_variable(name=name + "_biases", shape=[out_size], initializer=tf.constant_initializer(0.0))
        return weights, biases
    