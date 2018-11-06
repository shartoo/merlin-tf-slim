import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops, math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs


class LayerNormGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(LayerNormGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = rnn_cell_impl._linear([inputs, state], 2 * self._num_units, True, bias_ones, \
                                          self._kernel_initializer)
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
            r, u = layer_normalization(r, scope="r/"), layer_normalization(u, scope="u/")
            r, u = math_ops.sigmoid(r), math_ops.sigmoid(u)
        with vs.variable_scope("candidate"):
            c = self._activation(
                rnn_cell_impl._linear([inputs, r * state], self._num_units, True, self._bias_initializer,
                                      self._kernel_initializer))
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def layer_normalization(inputs, epsilon=1e-5, scope=None):
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + "LN", reuse=None):
        scale = tf.get_variable(name="scale", shape=[inputs.get_shape()[1]], initializer=tf.constant_initializer(1))
        shift = tf.get_variable(name="shift", shape=[inputs.get_shape()[1]], initializer=tf.constant_initializer(0))
    LN_output = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
    return LN_output
