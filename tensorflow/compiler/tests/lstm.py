# Lint as: python2, python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A simple LSTM layer with benchmarks.

This sets up a simple LSTM (Long Short Term Memory) layer, unrolled to a fixed
length sequence.  The only deviation from standard LSTM cells is that
activations are clipped, inspired by the GNMT machine translation model.
The GNMT paper has more details: https://arxiv.org/abs/1609.08144
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables


def Clip(x):
  """Clips x to the range [-1., 1.]."""
  return math_ops.maximum(math_ops.minimum(x, 1.), -1.)


def LSTMCellWeightsShape(num_inputs, num_nodes):
  """Returns the shape of the weights for a single LSTM cell."""
  # Dimension 0 accounts for combining x with the previous m state.
  # Dimension 1 accounts for the in value and the (in, forget, out) gates.
  return [num_inputs + num_nodes, 4 * num_nodes]


def LSTMCell(weights, m_prev, c_prev, x, pad):
  """Unrolls a single LSTM cell with clipped activations forward by one step.

  Args:
    weights: Weight matrix with shape LSTMCellWeightsShape.
    m_prev: Previous m states with shape [batch_size, num_nodes].
    c_prev: Previous c states with shape [batch_size, num_nodes].
    x: Input with shape [batch_size, num_inputs].
    pad: Padding with shape [batch_size, 1].  Each padding value is either
        0 or 1, where 1 indicates padding; i.e. the input is shorter than the
        sequence length, and the (m, c) states should simply be passed through
        from the previous states.
  Returns:
    The next (m, c) states, each with shape [batch_size, num_nodes].
  """
  # Apply weights to the input and previous hidden state.
  # The matmul here is the "big" operation.
  xm = array_ops.concat([x, m_prev], 1)
  xmw = math_ops.matmul(xm, weights)

  # Element-wise ops for the standard LSTM cell, with clipped activations.
  # XLA can fuse these operations into a single loop.
  in_value, in_gate, forget_gate, out_gate = array_ops.split(
      value=xmw, num_or_size_splits=4, axis=1)
  in_value = math_ops.tanh(in_value)
  in_gate = math_ops.sigmoid(in_gate)
  forget_gate = math_ops.sigmoid(forget_gate)
  out_gate = math_ops.sigmoid(out_gate)
  c_next = Clip(Clip(forget_gate * c_prev) + Clip(in_gate * in_value))
  m_next = Clip(out_gate * c_next)

  # Account for padding.
  c_next = c_prev * pad + c_next * (1.0 - pad)
  m_next = m_prev * pad + m_next * (1.0 - pad)

  return m_next, c_next


def LSTMLayer(cell_name, weights, m, c, x_seq, pad_seq):
  """Unrolls a layer of LSTM cells forward by the sequence length.

  The sequence length is determined by the length of x_seq and pad_seq, which
  must be the same.

  Args:
    cell_name: Base name of each cell.
    weights: Weight matrix with shape LSTMCellWeightsShape.
    m: Initial m states with shape [batch_size, num_nodes].
    c: Initial c states with shape [batch_size, num_nodes].
    x_seq: List of inputs, each with shape [batch_size, num_inputs].
        The length of the list is the sequence length.
    pad_seq: List of paddings, each with shape [batch_size, 1].
        The length of the list is the sequence length.
        Each padding value is either 0 or 1, where 1 indicates padding;
        i.e. the input is shorter than the sequence length.
  Returns:
    List of per-sequence-step outputs, each with shape [batch_size, num_nodes].
  Raises:
    ValueError: If len(x_seq) != len(pad_seq).
  """
  if len(x_seq) != len(pad_seq):
    raise ValueError('length of x_seq(%d) != pad_seq(%d)' %
                     (len(x_seq), len(pad_seq)))
  out_seq = []
  for seq in range(len(x_seq)):
    with ops.name_scope('%s_%d' % (cell_name, seq)):
      m, c = LSTMCell(weights, m, c, x_seq[seq], pad_seq[seq])
      out_seq.append(array_ops.identity(m, name='out'))
  return out_seq


def RandomVar(shape, name=None):
  """Returns a variable of the given shape initialized to random values."""
  return variables.VariableV1(
      random_ops.random_uniform(shape), dtype=dtypes.float32, name=name)


def RandomInputs(batch_size, seq_length, num_inputs):
  """Returns randomly initialized (x_seq, pad_seq) sequences."""
  x_seq = []
  pad_seq = []
  with ops.name_scope('inputs'):
    for seq in range(seq_length):
      x_seq.append(RandomVar([batch_size, num_inputs], name='x_seq_%d' % seq))
      # Real padding values are always a sequence of 0 followed by a
      # sequence of 1, but random values are fine for benchmarking.
      pad_seq.append(RandomVar([batch_size, 1], name='pad_seq_%d' % seq))
  return x_seq, pad_seq


def BuildLSTMLayer(batch_size, seq_length, num_inputs, num_nodes):
  """Builds a single LSTM layer with random weights and inputs.

  Args:
    batch_size: Inputs are fed in batches of this size.
    seq_length: The sequence length to unroll the LSTM layer.
    num_inputs: Dimension of inputs that are fed into each LSTM cell.
    num_nodes: The number of nodes in each LSTM cell.

  Returns:
    (out_seq, weights) pair.  The out_seq is a list of per-sequence-step
    outputs, each with shape [batch_size, num_nodes].  The weights are a list of
    weight variables that may be trained.
  """
  weights = RandomVar(
      LSTMCellWeightsShape(num_inputs, num_nodes), name='weights')
  m = array_ops.zeros([batch_size, num_nodes], name='init_m')
  c = array_ops.zeros([batch_size, num_nodes], name='init_c')
  x_seq, pad_seq = RandomInputs(batch_size, seq_length, num_inputs)

  out_seq = LSTMLayer('lstm', weights, m, c, x_seq, pad_seq)
  return out_seq, [weights]
