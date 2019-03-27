# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Popnn RNN operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging

POPNN_LSTM = "lstm"

POPNN_LSTM_NUM_GATES = 4

__all__ = ["PopnnLSTM"]


class PopnnLSTM(base_layer.Layer):
  # pylint:disable=line-too-long
  """XLA compatible, time-major Popnn implementation of an LSTM layer.

  Below is a typical workflow:

  ```python
  with tf.Graph().as_default():
    lstm = PopnnLSTM(num_units, ...)

    outputs, output_states = lstm(inputs, initial_states, training=True)
  ```
  """
  # pylint:enable=line-too-long
  _rnn_mode = POPNN_LSTM
  _num_gates_per_layer = POPNN_LSTM_NUM_GATES

  # _saveable_cls = popnn_rnn_ops.PopnnLSTMSaveable

  def __init__(self,
               num_units,
               dtype=dtypes.float32,
               partials_dtype=dtypes.float32,
               weights_initializer=None,
               bias_initializer=None,
               name=None):
    """Creates a PopnnLSTM model from model spec.

    Args:
      num_units: the number of units within the LSTM model.
      dtype: tf.float16 or tf.float32
      partials_dtype: the type used by Popnn to perform partial calculations.
        Either tf.float16 or tf.float32.
      weights_initializer: starting value to initialize the weight
        (default is all zeros).
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      name: VariableScope for the created subgraph; defaults to class name.
        This only serves the default scope if later no scope is specified when
        invoking __call__().
    """
    super(PopnnLSTM, self).__init__(dtype=dtype, name=name)

    if dtype not in [dtypes.float16, dtypes.float32]:
      raise ValueError("Only support float16, float32, provided %s" % dtype)
    # Layer self.dtype is type name, the original DType object is kept here.
    self._plain_dtype = dtype
    self._partials_dtype = partials_dtype
    self._num_layers = 1
    self._num_units = num_units
    self._weights_initializer = weights_initializer
    self._bias_initializer = bias_initializer
    # Init input_size to None, which will be set after build().
    self._input_size = None
    self._saveable = None

  @property
  def num_layers(self):
    return self._num_layers

  @property
  def num_units(self):
    return self._num_units

  @property
  def input_size(self):
    if not self._input_size:
      raise ValueError(
          "\'input_size\' is unknown since layer has not been built.")
    return self._input_size

  @property
  def saveable(self):
    raise NotImplementedError(
        "This cell does not yet support object-based saving. File a feature "
        "request if this limitation bothers you.")

  @property
  def canonical_weight_shape(self):
    """Shapes of Popnn canonical weight tensors."""
    if not self._input_size:
      raise RuntimeError(
          "%s.canonical_weight_shape invoked before input shape is known" %
          type(self).__name__)

    return self._canonical_weight_shape(0)

  @property
  def canonical_bias_shapes(self):
    """Shapes of Popnn canonical bias tensors."""
    return self._canonical_bias_shape(0)

  def build(self, input_shape):
    """Create variables of the Popnn LSTM.

    It can be called manually before `__call__()` or automatically through
    `__call__()`. In the former case, subsequent `__call__()`s will skip
    creating variables.
    Args:
      input_shape: a TensorShape object with 3 dimensions.
    Raises:
      ValueError: if input_shape has wrong dimension or unknown 3rd dimension.
    """
    if self.built:
      return

    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape.ndims != 3:
      raise ValueError(
          "Expecting input_shape with 3 dims, got %d" % input_shape.ndims)
    if input_shape[-1].value is None:
      raise ValueError("The last dimension of the inputs to `PopnnLSTM` "
                       "should be defined. Found `None`.")
    self._input_size = input_shape[-1].value
    self.input_spec = base_layer.InputSpec(ndim=3, axes={-1: self._input_size})

    # Create the variables
    with vs.variable_scope(self._scope, reuse=self.built):
      if self._weights_initializer is None:
        self._weights_initializer = init_ops.constant_initializer(
            0.0, dtype=self._plain_dtype)
      if self._bias_initializer is None:
        self._bias_initializer = init_ops.constant_initializer(
            0.0, dtype=self._plain_dtype)
      self.kernel = vs.get_variable(
          "kernel",
          dtype=self._plain_dtype,
          initializer=self._weights_initializer,
          shape=self.canonical_weight_shape)
      self.biases = vs.get_variable(
          "biases",
          dtype=self._plain_dtype,
          initializer=self._bias_initializer,
          shape=self.canonical_bias_shapes)

    self.built = True

  def call(self, inputs, initial_state=None, training=True):
    """Runs the forward step for the LSTM model.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple of tensor of shape `[batch_size, num_units]`. If
        not provided, the state is initialized to zeros.
      training: whether this operation will be used in training or inference.
    Returns:
      output: a tensor of shape `[time_len, batch_size, num_units]`.
      output_states: a tuple of tensor of the same shape and structure as
        `initial_state`.
    Raises:
      ValueError: initial_state is not a tuple.
    """
    if initial_state is not None and not isinstance(initial_state, tuple):
      raise ValueError("Invalid initial_state type: %s, expecting tuple.",
                       type(initial_state))
    dtype = self.dtype
    inputs = ops.convert_to_tensor(inputs, dtype=dtype)

    batch_size = array_ops.shape(inputs)[1]
    if initial_state is None:
      initial_state = self._zero_state(batch_size)
    h, c = initial_state
    h = ops.convert_to_tensor(h, dtype=dtype)
    c = ops.convert_to_tensor(c, dtype=dtype)
    outputs, state = self._forward(inputs, h, c, self.kernel, self.biases,
                                   training)
    return outputs, state

  def state_shape(self, batch_size):
    """Shape of Popnn LSTM states.

    Shape is a 2-element tuple. Each is
    [batch_size, num_units]
    Args:
      batch_size: an int
    Returns:
      a tuple of python arrays.
    """
    return ([batch_size, self.num_units], [batch_size, self.num_units])

  def _zero_state(self, batch_size):
    res = []
    for sp in self.state_shape(batch_size):
      res.append(array_ops.zeros(sp, dtype=self.dtype))
    return tuple(res)

  def _canonical_weight_shape(self, layer):
    """Shapes of Popnn canonical weight tensors for given layer."""
    if layer < 0 or layer >= self._num_layers:
      raise ValueError("\'layer\' is not valid, got %s, expecting [%d, %d]" %
                       (layer, 0, self._num_layers - 1))
    if not self._input_size:
      raise RuntimeError(
          "%s._canonical_weight_shape invoked before input shape is known" %
          type(self).__name__)

    input_size = self._input_size
    num_units = self._num_units
    num_gates = self._num_gates_per_layer

    if layer == 0:
      tf_wts = [input_size, num_units * num_gates]
    else:
      #TODO we only support one layer.
      tf_wts = [num_units, num_units * num_gates]
    tf_wts[0] += num_units
    return tf_wts

  def _canonical_bias_shape(self, unused_layer):
    """Shapes of Popnn canonical bias tensors for given layer."""
    return [self._num_gates_per_layer, self._num_units]

  def _forward(self, inputs, h, c, kernel, biases, training):
    output, output_h, output_c, _ = gen_popnn_ops.popnn_lstm_layer(
        inputs=inputs,
        num_channels=self._num_units,
        kernel=kernel,
        biases=biases,
        input_h_state=h,
        input_c_state=c,
        is_training=training,
        partials_dtype=self._partials_dtype,
        name=self._name)
    return output, (output_h, output_c)
