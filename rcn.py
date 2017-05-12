"""Something I have to write here

"""

from __future__ import division

import tensorflow as tf
#from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
#from tensorflow.nn.rnn_cell import RNNCell

def dynamic_rcn(cell, inputs, **rnn_args):
    """Creates a recurrent neural network specified by RCNCell `cell`.

    Args:
        cell: An instance of RCNCell.
        inputs:
            A `Tensor` of shape: `[max_time, batch_size, image_height,
            image_width, chanel_size]`.
        other args:
            The same as `dynamic_rnn` function.

    Returns:

    """
    if not isinstance(cell, RCNCell):
        raise TypeError("cell must be an instance of RCNCell")
    rnn_args['time_major'] = True 
    isp = inputs.get_shape().as_list()
    seq_input = tf.reshape(inputs, shape=[isp[0], -1, isp[2] * isp[3] * isp[4]])
    output, state = tf.nn.dynamic_rnn(cell, seq_input, **rnn_args)
    return output, state

def dynamic_stacked_rcn(cell, inputs, prevState, **rnn_args):
    """Creates a recurrent neural network specified by RCNCell `cell`.

    Args:
        cell: An instance of RCNCell.
        inputs:
            A `Tensor` of shape: `[max_time, batch_size, image_height,
            image_width, chanel_size]`.
        other args:
            The same as `dynamic_rnn` function.

    Returns:

    """
    if not isinstance(cell, RCNCell):
        raise TypeError("cell must be an instance of RCNCell")
    rnn_args['time_major'] = True 
    mergedInputs = tf.stack([inputs, prevState], axis=2)
    isp = mergedInputs.get_shape().as_list()
    print "mergedInputs", mergedInputs
    seq_input = tf.reshape(mergedInputs, shape=[isp[0], -1, isp[2], isp[3] * isp[4] * isp[5]])
    print "seq_input",seq_input 
    output, state = tf.nn.dynamic_rnn(cell, seq_input, **rnn_args)
    return output, state

class RCNCell(RNNCell):
    """To be completed
    """
    @property
    def input_size(self):
        """Abstract function
        """
        raise NotImplementedError("Abstract method")

    def __call__(self, inputs, state, scope=None):
        isp = inputs.get_shape().as_list()
        H, W, C = self.input_size
        assert isp[-1] == H * W * C
        input2 = tf.reshape(inputs, shape=(-1, H, W, C))
        return self.call(input2, state, scope)

    def call(self, inputs, state, scope=None):
        """A fake function ...
        """
        raise NotImplementedError("Abstract method")

class GruRcnCell(RCNCell):
    """To be completed.

    """

    def __init__(self, input_size, num_outputs,
                 ih_filter_size, ih_strides, ih_pandding,
                 hh_filter_size,
                 data_format=None):
        """To be completed.

        """
        self._input_size = input_size
        self._num_outputs = num_outputs
        self._ih_filter_h_length, self._ih_filter_w_length = ih_filter_size
        self._ih_strides = ih_strides
        self._ih_pandding = ih_pandding
        self._hh_filter_h_length, self._hh_filter_w_length = hh_filter_size
        self._data_format = data_format
        
        if data_format == 'NCHW':
            _, H, W = input_size
        else:
            H, W, _ = input_size
#             hS, wS = ih_strides
        _, hS, wS, _ = ih_strides
        if ih_pandding == "SAME":
            oH = (H - 1) / hS + 1
            oW = (W - 1) / wS + 1
        else:
            oH = (H - self._ih_filter_h_length) / hS + 1
            oW = (W - self._ih_filter_w_length) / wS + 1
        if oH % 1 == 0 and oH % 1 == 0:
            if data_format == 'NCHW':
                self._state_size = tf.TensorShape([num_outputs, oH, oW])
            else:
                self._state_size = tf.TensorShape([oH, oW, num_outputs])
        else:
            raise ValueError("The setting of convolutional op doesn't match the input_size")

    @property
    def input_size(self):
        """Return a 3-D list representing the shape of input_bak_old
        """
        return self._input_size

    @property
    def state_size(self):
        """Return a TensorShape representing the shape of hidden state
        """
        return self._state_size

    @property
    def output_size(self):
        """Get the shape of output
        """
        return tf.TensorShape(self._state_size)

    def call(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):  # "GruRcnCell"
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0.
                w_zrw = self._conv(inputs, self._num_outputs*3, self._ih_filter_h_length, self._ih_filter_w_length,
                                 self._ih_strides, self._ih_pandding, init_ops.truncated_normal_initializer(stddev=0.01), scope="WzrwConv")
 
                u_zr = self._conv(state, self._num_outputs*2, self._hh_filter_h_length, self._hh_filter_w_length, [1, 1, 1, 1],
                                 "SAME", init_ops.truncated_normal_initializer(stddev=0.01), scope="UzrConv")
                
                w_z, w_r, w =tf.split(value=w_zrw, num_or_size_splits=3, axis=3, name="w_split")
                u_z, u_r =tf.split(value=u_zr, num_or_size_splits=2, axis=3, name="u_split")

                z_bias = tf.get_variable(
                    name="z_biases",
                    shape=[self._num_outputs],
                    initializer=init_ops.ones_initializer()
                )
                z_gate = math_ops.sigmoid(tf.nn.bias_add(w_z + u_z, z_bias))

                r_bias = tf.get_variable(
                    name="r_biases",
                    shape=[self._num_outputs],
                    initializer=init_ops.ones_initializer())
                r_gate = math_ops.sigmoid(tf.nn.bias_add(w_r + u_r, r_bias))

            with vs.variable_scope("Candidate"):
#                 w = self._conv(inputs, self._num_outputs, self._ih_filter_h_length, self._ih_filter_w_length,
#                                self._ih_strides, self._ih_pandding, init_ops.truncated_normal_initializer(stddev=0.01), scope="WConv")
                u = self._conv(r_gate * state, self._num_outputs, self._hh_filter_h_length, self._hh_filter_w_length,
                               [1, 1, 1, 1], "SAME", init_ops.truncated_normal_initializer(stddev=0.01), scope="UConv")
                c_bias = tf.get_variable(
                    name="c_biases",
                    shape=[self._num_outputs],
                    initializer=init_ops.ones_initializer())
                c = math_ops.tanh(tf.nn.bias_add(w + u, c_bias))
            new_h = z_gate * state + (1 - z_gate) * c
        return new_h, new_h

    def _conv(self, inputs, nb_filter, filter_h_length, filter_w_length, strides,
              padding, weight_initializer, scope=None):
        with tf.variable_scope(scope or 'Convolutional'):
            in_channels = inputs.get_shape().as_list()[-1]
            kernel = tf.get_variable(
                initializer=weight_initializer,
                shape=[filter_h_length, filter_w_length, in_channels, nb_filter],
                name='weight')
            conv = tf.nn.conv2d(inputs, kernel, strides, padding,
                                data_format=self._data_format)
        return conv

class StackedGruRcnCell(RCNCell):
    """To be completed.

    """

    def __init__(self, input_size, num_outputs,
                 ih_filter_size, ih_strides, ih_pandding,
                 hh_filter_size,
                 data_format=None):
        """To be completed.

        """
        self._input_size = input_size
        self._num_outputs = num_outputs
        self._ih_filter_h_length, self._ih_filter_w_length = ih_filter_size
        self._ih_strides = ih_strides
        self._ih_pandding = ih_pandding
        self._hh_filter_h_length, self._hh_filter_w_length = hh_filter_size
        self._data_format = data_format
        
        if data_format == 'NCHW':
            _, H, W = input_size
        else:
            H, W, _ = input_size
#             hS, wS = ih_strides
        _, hS, wS, _ = ih_strides
        if ih_pandding == "SAME":
            oH = (H - 1) / hS + 1
            oW = (W - 1) / wS + 1
        else:
            oH = (H - self._ih_filter_h_length) / hS + 1
            oW = (W - self._ih_filter_w_length) / wS + 1
        if oH % 1 == 0 and oH % 1 == 0:
            if data_format == 'NCHW':
                self._state_size = tf.TensorShape([num_outputs, oH, oW])
            else:
                self._state_size = tf.TensorShape([oH, oW, num_outputs])
        else:
            raise ValueError("The setting of convolutional op doesn't match the input_size")

    @property
    def input_size(self):
        """Return a 3-D list representing the shape of input_bak_old
        """
        return self._input_size

    @property
    def state_size(self):
        """Return a TensorShape representing the shape of hidden state
        """
        return self._state_size

    @property
    def output_size(self):
        """Get the shape of output
        """
        return tf.TensorShape(self._state_size)

    def call(self, mergedInputs, state, scope=None):
        print "call",mergedInputs
        inputs, prevState = tf.split(mergedInputs, num_or_size_splits=2, axis=2, name="splitMergedInput")
        with vs.variable_scope(scope or type(self).__name__):  # "GruRcnCell"
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0.
                w_zrw = self._conv(inputs, self._num_outputs*3, self._ih_filter_h_length, self._ih_filter_w_length,
                                 self._ih_strides, self._ih_pandding, init_ops.truncated_normal_initializer(stddev=0.01), scope="WzrwConv")
 
                u_zr = self._conv(state, self._num_outputs*2, self._hh_filter_h_length, self._hh_filter_w_length, [1, 1, 1, 1],
                                 "SAME", init_ops.truncated_normal_initializer(stddev=0.01), scope="UzrConv")
                
                pervU_zr = self._conv(prevState, self._num_outputs*2, self._hh_filter_h_length, self._hh_filter_w_length, [1, 1, 1, 1],
                                 "SAME", init_ops.truncated_normal_initializer(stddev=0.01), scope="UzrConv")
                
                w_z, w_r, w =tf.split(value=w_zrw, num_or_size_splits=3, axis=3, name="w_split")
                u_z, u_r =tf.split(value=u_zr, num_or_size_splits=2, axis=3, name="u_split")
                prevU_z, prevU_r =tf.split(value=pervU_zr, num_or_size_splits=2, axis=3, name="prevU_split")

                z_bias = tf.get_variable(
                    name="z_biases",
                    shape=[self._num_outputs],
                    initializer=init_ops.ones_initializer()
                )
                z_gate = math_ops.sigmoid(tf.nn.bias_add(w_z + u_z + prevU_z, z_bias))

                r_bias = tf.get_variable(
                    name="r_biases",
                    shape=[self._num_outputs],
                    initializer=init_ops.ones_initializer())
                r_gate = math_ops.sigmoid(tf.nn.bias_add(w_r + u_r + prevU_r, r_bias))

            with vs.variable_scope("Candidate"):
#                 w = self._conv(inputs, self._num_outputs, self._ih_filter_h_length, self._ih_filter_w_length,
#                                self._ih_strides, self._ih_pandding, init_ops.truncated_normal_initializer(stddev=0.01), scope="WConv")
                u = self._conv(r_gate * state, self._num_outputs, self._hh_filter_h_length, self._hh_filter_w_length,
                               [1, 1, 1, 1], "SAME", init_ops.truncated_normal_initializer(stddev=0.01), scope="UConv")
                c_bias = tf.get_variable(
                    name="c_biases",
                    shape=[self._num_outputs],
                    initializer=init_ops.ones_initializer())
                c = math_ops.tanh(tf.nn.bias_add(w + u, c_bias))
            new_h = z_gate * state + (1 - z_gate) * c
        return new_h, new_h

    def _conv(self, inputs, nb_filter, filter_h_length, filter_w_length, strides,
              padding, weight_initializer, scope=None):
        with tf.variable_scope(scope or 'Convolutional'):
            in_channels = inputs.get_shape().as_list()[-1]
            kernel = tf.get_variable(
                initializer=weight_initializer,
                shape=[filter_h_length, filter_w_length, in_channels, nb_filter],
                name='weight')
            conv = tf.nn.conv2d(inputs, kernel, strides, padding,
                                data_format=self._data_format)
        return conv
