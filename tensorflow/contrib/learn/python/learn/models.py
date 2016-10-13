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

"""Various high level TF models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib.learn.python.learn.ops import losses_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs


def linear_regression_zero_init(x, y):
  """Linear regression subgraph with zero-value initial weights and bias.

  Args:
    x: tensor or placeholder for input features.
    y: tensor or placeholder for target.

  Returns:
    Predictions and loss tensors.
  """
  return linear_regression(x, y, init_mean=0.0, init_stddev=0.0)


def logistic_regression_zero_init(x, y):
  """Logistic regression subgraph with zero-value initial weights and bias.

  Args:
    x: tensor or placeholder for input features.
    y: tensor or placeholder for target.

  Returns:
    Predictions and loss tensors.
  """
  return logistic_regression(x, y, init_mean=0.0, init_stddev=0.0)


def linear_regression(x, y, init_mean=None, init_stddev=1.0):
  """Creates linear regression TensorFlow subgraph.

  Args:
    x: tensor or placeholder for input features.
    y: tensor or placeholder for target.
    init_mean: the mean value to use for initialization.
    init_stddev: the standard devation to use for initialization.

  Returns:
    Predictions and loss tensors.

  Side effects:
    The variables linear_regression.weights and linear_regression.bias are
    initialized as follows.  If init_mean is not None, then initialization
    will be done using a random normal initializer with the given init_mean
    and init_stddv.  (These may be set to 0.0 each if a zero initialization
    is desirable for convex use cases.)  If init_mean is None, then the
    uniform_unit_scaling_initialzer will be used.
  """
  with vs.variable_scope('linear_regression'):
    scope_name = vs.get_variable_scope().name
    logging_ops.histogram_summary('%s.x' % scope_name, x)
    logging_ops.histogram_summary('%s.y' % scope_name, y)
    dtype = x.dtype.base_dtype
    y_shape = y.get_shape()
    if len(y_shape) == 1:
      output_shape = 1
    else:
      output_shape = y_shape[1]
    # Set up the requested initialization.
    if init_mean is None:
      weights = vs.get_variable(
          'weights', [x.get_shape()[1], output_shape], dtype=dtype)
      bias = vs.get_variable('bias', [output_shape], dtype=dtype)
    else:
      weights = vs.get_variable('weights', [x.get_shape()[1], output_shape],
                                initializer=init_ops.random_normal_initializer(
                                    init_mean, init_stddev, dtype=dtype),
                                dtype=dtype)
      bias = vs.get_variable('bias', [output_shape],
                             initializer=init_ops.random_normal_initializer(
                                 init_mean, init_stddev, dtype=dtype),
                             dtype=dtype)
    logging_ops.histogram_summary('%s.weights' % scope_name, weights)
    logging_ops.histogram_summary('%s.bias' % scope_name, bias)
    return losses_ops.mean_squared_error_regressor(x, y, weights, bias)


def logistic_regression(x,
                        y,
                        class_weight=None,
                        init_mean=None,
                        init_stddev=1.0):
  """Creates logistic regression TensorFlow subgraph.

  Args:
    x: tensor or placeholder for input features,
       shape should be [batch_size, n_features].
    y: tensor or placeholder for target,
       shape should be [batch_size, n_classes].
    class_weight: tensor, [n_classes], where for each class
                  it has weight of the class. If not provided
                  will check if graph contains tensor `class_weight:0`.
                  If that is not provided either all ones are used.
    init_mean: the mean value to use for initialization.
    init_stddev: the standard devation to use for initialization.

  Returns:
    Predictions and loss tensors.

  Side effects:
    The variables linear_regression.weights and linear_regression.bias are
    initialized as follows.  If init_mean is not None, then initialization
    will be done using a random normal initializer with the given init_mean
    and init_stddv.  (These may be set to 0.0 each if a zero initialization
    is desirable for convex use cases.)  If init_mean is None, then the
    uniform_unit_scaling_initialzer will be used.
  """
  with vs.variable_scope('logistic_regression'):
    scope_name = vs.get_variable_scope().name
    logging_ops.histogram_summary('%s.x' % scope_name, x)
    logging_ops.histogram_summary('%s.y' % scope_name, y)
    dtype = x.dtype.base_dtype
    # Set up the requested initialization.
    if init_mean is None:
      weights = vs.get_variable(
          'weights', [x.get_shape()[1], y.get_shape()[-1]], dtype=dtype)
      bias = vs.get_variable('bias', [y.get_shape()[-1]], dtype=dtype)
    else:
      weights = vs.get_variable('weights',
                                [x.get_shape()[1], y.get_shape()[-1]],
                                initializer=init_ops.random_normal_initializer(
                                    init_mean, init_stddev, dtype=dtype),
                                dtype=dtype)
      bias = vs.get_variable('bias', [y.get_shape()[-1]],
                             initializer=init_ops.random_normal_initializer(
                                 init_mean, init_stddev, dtype=dtype),
                             dtype=dtype)
    logging_ops.histogram_summary('%s.weights' % scope_name, weights)
    logging_ops.histogram_summary('%s.bias' % scope_name, bias)
    # If no class weight provided, try to retrieve one from pre-defined
    # tensor name in the graph.
    if not class_weight:
      try:
        class_weight = ops.get_default_graph().get_tensor_by_name(
            'class_weight:0')
      except KeyError:
        pass

    return losses_ops.softmax_classifier(x,
                                         y,
                                         weights,
                                         bias,
                                         class_weight=class_weight)


## This will be in TensorFlow 0.7.
## TODO(ilblackdragon): Clean this up when it's released
def _reverse_seq(input_seq, lengths):
  """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
    lengths:   A tensor of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply
               reverses the list.

  Returns:
    time-reversed sequence
  """
  if lengths is None:
    return list(reversed(input_seq))

  for input_ in input_seq:
    input_.set_shape(input_.get_shape().with_rank(2))

  # Join into (time, batch_size, depth)
  s_joined = array_ops_.pack(input_seq)

  # Reverse along dimension 0
  s_reversed = array_ops_.reverse_sequence(s_joined, lengths, 0, 1)
  # Split again into list
  result = array_ops_.unpack(s_reversed)
  return result


def bidirectional_rnn(cell_fw,
                      cell_bw,
                      inputs,
                      initial_state_fw=None,
                      initial_state_bw=None,
                      dtype=None,
                      sequence_length=None,
                      scope=None):
  """Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states
  are ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not
  given.
  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      [batch_size x cell.state_size].
    initial_state_bw: (optional) Same as for initial_state_fw.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int64 vector (tensor) of size
      [batch_size],
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"

  Returns:
    A pair (outputs, state) where:
      outputs is a length T list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs
      state is the concatenated final state of the forward and backward RNN

  Raises:
    TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell_fw, nn.rnn_cell.RNNCell):
    raise TypeError('cell_fw must be an instance of RNNCell')
  if not isinstance(cell_bw, nn.rnn_cell.RNNCell):
    raise TypeError('cell_bw must be an instance of RNNCell')
  if not isinstance(inputs, list):
    raise TypeError('inputs must be a list')
  if not inputs:
    raise ValueError('inputs must not be empty')

  name = scope or 'BiRNN'
  # Forward direction
  with vs.variable_scope(name + '_FW'):
    output_fw, state_fw = nn.rnn(cell_fw, inputs, initial_state_fw, dtype,
                                 sequence_length)

  # Backward direction
  with vs.variable_scope(name + '_BW'):
    tmp, state_bw = nn.rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                           initial_state_bw, dtype, sequence_length)
  output_bw = _reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  outputs = [array_ops_.concat(1, [fw, bw])
             for fw, bw in zip(output_fw, output_bw)]

  return outputs, array_ops_.concat(1, [state_fw, state_bw])

# End of TensorFlow 0.7


def get_rnn_model(rnn_size, cell_type, num_layers, input_op_fn, bidirectional,
                  target_predictor_fn, sequence_length, initial_state,
                  attn_length, attn_size, attn_vec_size):
  """Returns a function that creates a RNN TensorFlow subgraph.

  Args:
    rnn_size: The size for rnn cell, e.g. size of your word embeddings.
    cell_type: The type of rnn cell, including rnn, gru, and lstm.
    num_layers: The number of layers of the rnn model.
    input_op_fn: Function that will transform the input tensor, such as
                 creating word embeddings, byte list, etc. This takes
                 an argument `x` for input and returns transformed `x`.
    bidirectional: boolean, Whether this is a bidirectional rnn.
    target_predictor_fn: Function that will predict target from input
                         features. This can be logistic regression,
                         linear regression or any other model,
                         that takes `x`, `y` and returns predictions and loss
                         tensors.
    sequence_length: If sequence_length is provided, dynamic calculation is
      performed. This saves computational time when unrolling past max sequence
      length. Required for bidirectional RNNs.
    initial_state: An initial state for the RNN. This must be a tensor of
      appropriate type and shape [batch_size x cell.state_size].
    attn_length: integer, the size of attention vector attached to rnn cells.
    attn_size: integer, the size of an attention window attached to rnn cells.
    attn_vec_size: integer, the number of convolutional features calculated on
      attention state and the size of the hidden layer built from base cell state.

  Returns:
    A function that creates the subgraph.
  """

  def rnn_estimator(x, y):
    """RNN estimator with target predictor function on top."""
    x = input_op_fn(x)
    if cell_type == 'rnn':
      cell_fn = nn.rnn_cell.BasicRNNCell
    elif cell_type == 'gru':
      cell_fn = nn.rnn_cell.GRUCell
    elif cell_type == 'lstm':
      cell_fn = functools.partial(
          nn.rnn_cell.BasicLSTMCell, state_is_tuple=False)
    else:
      raise ValueError('cell_type {} is not supported. '.format(cell_type))
    # TODO: state_is_tuple=False is deprecated
    if bidirectional:
      # forward direction cell
      fw_cell = cell_fn(rnn_size)
      bw_cell = cell_fn(rnn_size)
      # attach attention cells if specified
      if attn_length is not None:
        fw_cell = contrib_rnn.AttentionCellWrapper(
          fw_cell, attn_length=attn_length, attn_size=attn_size,
          attn_vec_size=attn_vec_size, state_is_tuple=False)
        bw_cell = contrib_rnn.AttentionCellWrapper(
          bw_cell, attn_length=attn_length, attn_size=attn_size,
          attn_vec_size=attn_vec_size, state_is_tuple=False)
      rnn_fw_cell = nn.rnn_cell.MultiRNNCell([fw_cell] * num_layers,
                                             state_is_tuple=False)
      # backward direction cell
      rnn_bw_cell = nn.rnn_cell.MultiRNNCell([bw_cell] * num_layers,
                                             state_is_tuple=False)
      # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
      _, encoding = bidirectional_rnn(rnn_fw_cell,
                                      rnn_bw_cell,
                                      x,
                                      dtype=dtypes.float32,
                                      sequence_length=sequence_length,
                                      initial_state_fw=initial_state,
                                      initial_state_bw=initial_state)
    else:
      rnn_cell = cell_fn(rnn_size)
      if attn_length is not None:
        rnn_cell = contrib_rnn.AttentionCellWrapper(
            rnn_cell, attn_length=attn_length, attn_size=attn_size,
            attn_vec_size=attn_vec_size, state_is_tuple=False)
      cell = nn.rnn_cell.MultiRNNCell([rnn_cell] * num_layers,
                                      state_is_tuple=False)
      _, encoding = nn.rnn(cell,
                           x,
                           dtype=dtypes.float32,
                           sequence_length=sequence_length,
                           initial_state=initial_state)
    return target_predictor_fn(encoding, y)

  return rnn_estimator
