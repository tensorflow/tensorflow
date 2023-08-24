# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Module implementing RNN Cells.

This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

# Remove caller that rely on private symbol in future.
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

BasicLSTMCell = rnn_cell_impl.BasicLSTMCell
BasicRNNCell = rnn_cell_impl.BasicRNNCell
DeviceWrapper = rnn_cell_impl.DeviceWrapper
DropoutWrapper = rnn_cell_impl.DropoutWrapper
GRUCell = rnn_cell_impl.GRUCell
LayerRNNCell = rnn_cell_impl.LayerRNNCell
LSTMCell = rnn_cell_impl.LSTMCell
LSTMStateTuple = rnn_cell_impl.LSTMStateTuple
MultiRNNCell = rnn_cell_impl.MultiRNNCell
ResidualWrapper = rnn_cell_impl.ResidualWrapper
RNNCell = rnn_cell_impl.RNNCell


def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""

  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size

  return nest.map_structure(get_state_shape, state_size)


def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.

  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).

  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.

  Returns:
    shape: the concatenation of prefix and suffix.

  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, tensor.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError(
          "prefix tensor must be either a scalar or vector, but saw tensor: %s"
          % p
      )
  else:
    p = tensor_shape.TensorShape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (
        constant_op.constant(p.as_list(), dtype=dtypes.int32)
        if p.is_fully_defined()
        else None
    )
  if isinstance(suffix, tensor.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError(
          "suffix tensor must be either a scalar or vector, but saw tensor: %s"
          % s
      )
  else:
    s = tensor_shape.TensorShape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (
        constant_op.constant(s.as_list(), dtype=dtypes.int32)
        if s.is_fully_defined()
        else None
    )

  if static:
    shape = tensor_shape.TensorShape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError(
          "Provided a prefix or suffix of None: %s and %s" % (prefix, suffix)
      )
    shape = array_ops.concat((p, s), 0)
  return shape


def _hasattr(obj, attr_name):
  try:
    getattr(obj, attr_name)
  except AttributeError:
    return False
  else:
    return True


def assert_like_rnncell(cell_name, cell):
  """Raises a TypeError if cell is not like an RNNCell.

  NOTE: Do not rely on the error message (in particular in tests) which can be
  subject to change to increase readability. Use
  ASSERT_LIKE_RNNCELL_ERROR_REGEXP.

  Args:
    cell_name: A string to give a meaningful error referencing to the name of
      the functionargument.
    cell: The object which should behave like an RNNCell.

  Raises:
    TypeError: A human-friendly exception.
  """
  conditions = [
      _hasattr(cell, "output_size"),
      _hasattr(cell, "state_size"),
      _hasattr(cell, "get_initial_state") or _hasattr(cell, "zero_state"),
      callable(cell),
  ]
  errors = [
      "'output_size' property is missing",
      "'state_size' property is missing",
      "either 'zero_state' or 'get_initial_state' method is required",
      "is not callable",
  ]

  if not all(conditions):
    errors = [error for error, cond in zip(errors, conditions) if not cond]
    raise TypeError(
        "The argument {!r} ({}) is not an RNNCell: {}.".format(
            cell_name, cell, ", ".join(errors)
        )
    )
