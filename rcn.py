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
    print rnn_args 
    isp = inputs.get_shape().as_list()
    seq_input = tf.reshape(inputs, shape=[isp[0], -1, isp[2] * isp[3] * isp[4]])
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

    def __init__(self, input_size, nb_filter,
                 ih_filter_h_length, ih_filter_w_length, ih_strides, ih_pandding,
                 hh_filter_h_length,
                 hh_filter_w_length,
                 weight_initializers=None,
                 data_format=None):
        """To be completed.

        """
        self._input_size = input_size
        self._nb_filter = nb_filter
        self._ih_filter_h_length = ih_filter_h_length
        self._ih_filter_w_length = ih_filter_w_length
        self._ih_strides = ih_strides
        self._ih_pandding = ih_pandding
        self._hh_filter_h_length = hh_filter_h_length
        self._hh_filter_w_length = hh_filter_w_length
        self.weight_initializers=weight_initializers
        self._data_format = data_format

        if data_format == 'NCHW':
            _, H, W = input_size
        else:
            H, W, _ = input_size
        _, hS, wS, _ = ih_strides
        if ih_pandding == "SAME":
            oH = (H - 1) / hS + 1
            oW = (W - 1) / wS + 1
        else:
            oH = (H - ih_filter_h_length) / hS + 1
            oW = (W - ih_filter_w_length) / wS + 1
        if oH % 1 == 0 and oH % 1 == 0:
            if data_format == 'NCHW':
                self._state_size = tf.TensorShape([nb_filter, oH, oW])
            else:
                self._state_size = tf.TensorShape([oH, oW, nb_filter])
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
                init = self._get_initializer("WzConv", init_ops.truncated_normal_initializer(stddev=0.01))
                w_z = self._conv(inputs, self._nb_filter, self._ih_filter_h_length, self._ih_filter_w_length,
                                 self._ih_strides, self._ih_pandding, init, scope="WzConv")

                init = self._get_initializer("UzConv", init_ops.truncated_normal_initializer(stddev=0.01))
                u_z = self._conv(state, self._nb_filter, self._hh_filter_h_length, self._hh_filter_w_length, [1, 1, 1, 1],
                                 "SAME", init, scope="UzConv")

                init = self._get_initializer("z_biases", init_ops.ones_initializer())
                z_bias = tf.get_variable(
                    name="z_biases",
                    shape=[self._nb_filter],
                    initializer=init
                )
                z_gate = math_ops.sigmoid(tf.nn.bias_add(w_z + u_z, z_bias))

                init = self._get_initializer("WrConv", init_ops.truncated_normal_initializer(stddev=0.01))
                w_r = self._conv(inputs, self._nb_filter, self._ih_filter_h_length, self._ih_filter_w_length,
                                 self._ih_strides, self._ih_pandding, init, scope="WrConv")

                init = self._get_initializer("UrConv", init_ops.truncated_normal_initializer(stddev=0.01))
                u_r = self._conv(state, self._nb_filter, self._hh_filter_h_length, self._hh_filter_w_length, [1, 1, 1, 1],
                                 "SAME", init, scope="UrConv")

                init = self._get_initializer("r_biases", init_ops.ones_initializer())
                r_bias = tf.get_variable(
                    name="r_biases",
                    shape=[self._nb_filter],
                    initializer=init_ops.ones_initializer())
                r_gate = math_ops.sigmoid(tf.nn.bias_add(w_r + u_r, r_bias))

            with vs.variable_scope("Candidate"):
                init = self._get_initializer("WConv", init_ops.truncated_normal_initializer(stddev=0.01))
                w = self._conv(inputs, self._nb_filter, self._ih_filter_h_length, self._ih_filter_w_length,
                               self._ih_strides, self._ih_pandding, init, scope="WConv")
                init = self._get_initializer("UConv", init_ops.truncated_normal_initializer(stddev=0.01))
                u = self._conv(r_gate * state, self._nb_filter, self._hh_filter_h_length, self._hh_filter_w_length,
                               [1, 1, 1, 1], "SAME", init, scope="UConv")
                init = self._get_initializer("c_biases", init_ops.ones_initializer())
                c_bias = tf.get_variable(
                    name="c_biases",
                    shape=[self._nb_filter],
                    initializer=init_ops.ones_initializer())
                c = math_ops.tanh(tf.nn.bias_add(w + u, c_bias))
            new_h = z_gate * state + (1 - z_gate) * c
        return new_h, new_h

    def _get_initializer(self, name, default=None):
        if self.weight_initializers is not None and name in self.weight_initializers:
            init = self.weight_initializers[name]
        else:
            init = default
        return init

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
