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
"""Module for constructing fused RNN cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn


@six.add_metaclass(abc.ABCMeta)
class FusedRNNCell(object):
  """Abstract object representing a fused RNN cell.

  A fused RNN cell represents the entire RNN expanded over the time
  dimension. In effect, this represents an entire recurrent network.

  Unlike RNN cells which are subclasses of `rnn_cell.RNNCell`, a `FusedRNNCell`
  operates on the entire time sequence at once, by putting the loop over time
  inside the cell. This usually leads to much more efficient, but more complex
  and less flexible implementations.

  Every `FusedRNNCell` must implement `__call__` with the following signature.
  """

  @abc.abstractmethod
  def __call__(self,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
    """Run this fused RNN on inputs, starting from the given state.

    Args:
      inputs: `3-D` tensor with shape `[time_len x batch_size x input_size]`
        or a list of `time_len` tensors of shape `[batch_size x input_size]`.
      initial_state: either a tensor with shape `[batch_size x state_size]`
        or a tuple with shapes `[batch_size x s] for s in state_size`, if the
        cell takes tuples. If this is not provided, the cell is expected to
        create a zero initial state of type `dtype`.
      dtype: The data type for the initial state and expected output. Required
        if `initial_state` is not provided or RNN state has a heterogeneous
          dtype.
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len)`.
        Defaults to `time_len` for each element.
      scope: `VariableScope` or `string` for the created subgraph; defaults to
        class name.

    Returns:
      A pair containing:

      - Output: A `3-D` tensor of shape `[time_len x batch_size x output_size]`
        or a list of `time_len` tensors of shape `[batch_size x output_size]`,
        to match the type of the `inputs`.
      - Final state: Either a single `2-D` tensor, or a tuple of tensors
        matching the arity and shapes of `initial_state`.
    """
    pass


class FusedRNNCellAdaptor(FusedRNNCell):
  """This is an adaptor for RNNCell classes to be used with `FusedRNNCell`."""

  def __init__(self, cell, use_dynamic_rnn=False):
    """Initialize the adaptor.

    Args:
      cell: an instance of a subclass of a `rnn_cell.RNNCell`.
      use_dynamic_rnn: whether to use dynamic (or static) RNN.
    """
    self._cell = cell
    self._use_dynamic_rnn = use_dynamic_rnn

  def __call__(self,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
    is_list = isinstance(inputs, list)
    if self._use_dynamic_rnn:
      if is_list:
        inputs = array_ops.stack(inputs)
      outputs, state = rnn.dynamic_rnn(
          self._cell,
          inputs,
          sequence_length=sequence_length,
          initial_state=initial_state,
          dtype=dtype,
          time_major=True,
          scope=scope)
      if is_list:
        # Convert outputs back to list
        outputs = array_ops.unstack(outputs)
    else:  # non-dynamic rnn
      if not is_list:
        inputs = array_ops.unstack(inputs)
      outputs, state = rnn.static_rnn(
          self._cell,
          inputs,
          initial_state=initial_state,
          dtype=dtype,
          sequence_length=sequence_length,
          scope=scope)
      if not is_list:
        # Convert outputs back to tensor
        outputs = array_ops.stack(outputs)

    return outputs, state


class TimeReversedFusedRNN(FusedRNNCell):
  """This is an adaptor to time-reverse a FusedRNNCell.

  For example,

  ```python
  cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(10)
  fw_lstm = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)
  bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)
  fw_out, fw_state = fw_lstm(inputs)
  bw_out, bw_state = bw_lstm(inputs)
  ```
  """

  def __init__(self, cell):
    self._cell = cell

  def _reverse(self, t, lengths):
    """Time reverse the provided tensor or list of tensors.

    Assumes the top dimension is the time dimension.

    Args:
      t: 3D tensor or list of 2D tensors to be reversed
      lengths: 1D tensor of lengths, or `None`

    Returns:
      A reversed tensor or list of tensors
    """
    if isinstance(t, list):
      return list(reversed(t))
    else:
      if lengths is None:
        return array_ops.reverse_v2(t, [0])
      else:
        return array_ops.reverse_sequence(t, lengths, 0, 1)

  def __call__(self,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
    inputs = self._reverse(inputs, sequence_length)
    outputs, state = self._cell(
        inputs,
        initial_state=initial_state,
        dtype=dtype,
        sequence_length=sequence_length,
        scope=scope)
    outputs = self._reverse(outputs, sequence_length)
    return outputs, state
