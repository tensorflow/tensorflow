# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LSTM layers for sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope


def _shape(tensor):
  return tensor.get_shape().as_list()


def ndlstm_base_unrolled(inputs, noutput, scope=None, reverse=False):
  """Run an LSTM, either forward or backward.

  This is a 1D LSTM implementation using unrolling and the TensorFlow
  LSTM op.

  Args:
    inputs: input sequence (length, batch_size, ninput)
    noutput: depth of output
    scope: optional scope name
    reverse: run LSTM in reverse

  Returns:
    Output sequence (length, batch_size, noutput)

  """
  with variable_scope.variable_scope(scope, "SeqLstmUnrolled", [inputs]):
    length, batch_size, _ = _shape(inputs)
    lstm_cell = rnn_cell.BasicLSTMCell(noutput, state_is_tuple=False)
    state = array_ops.zeros([batch_size, lstm_cell.state_size])
    output_u = []
    inputs_u = array_ops.unstack(inputs)
    if reverse:
      inputs_u = list(reversed(inputs_u))
    for i in xrange(length):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = lstm_cell(inputs_u[i], state)
      output_u += [output]
    if reverse:
      output_u = list(reversed(output_u))
    outputs = array_ops.stack(output_u)
    return outputs


def ndlstm_base_dynamic(inputs, noutput, scope=None, reverse=False):
  """Run an LSTM, either forward or backward.

  This is a 1D LSTM implementation using dynamic_rnn and
  the TensorFlow LSTM op.

  Args:
    inputs: input sequence (length, batch_size, ninput)
    noutput: depth of output
    scope: optional scope name
    reverse: run LSTM in reverse

  Returns:
    Output sequence (length, batch_size, noutput)
  """
  with variable_scope.variable_scope(scope, "SeqLstm", [inputs]):
    lstm_cell = rnn_cell.BasicLSTMCell(noutput)
    if reverse:
      inputs = array_ops.reverse_v2(inputs, [0])
    outputs, _ = rnn.dynamic_rnn(
      lstm_cell, inputs, time_major=True, dtype=inputs.dtype)
    if reverse:
      outputs = array_ops.reverse_v2(outputs, [0])
    return outputs


def ndlstm_base(inputs, noutput, scope=None, reverse=False, dynamic=True):
  """Implements a 1D LSTM, either forward or backward.

  This is a base case for multidimensional LSTM implementations, which
  tend to be used differently from sequence-to-sequence
  implementations.  For general 1D sequence to sequence
  transformations, you may want to consider another implementation
  from TF slim.

  Args:
    inputs: input sequence (length, batch_size, ninput)
    noutput: depth of output
    scope: optional scope name
    reverse: run LSTM in reverse
    dynamic: use dynamic_rnn

  Returns:
    Output sequence (length, batch_size, noutput)

  """
  # TODO(tmb) maybe add option for other LSTM implementations, like
  # slim.rnn.basic_lstm_cell
  if dynamic:
    return ndlstm_base_dynamic(inputs, noutput, scope=scope, reverse=reverse)
  else:
    return ndlstm_base_unrolled(inputs, noutput, scope=scope, reverse=reverse)


def sequence_to_final(inputs, noutput, scope=None, name=None, reverse=False):
  """Run an LSTM across all steps and returns only the final state.

  Args:
    inputs: (length, batch_size, depth) tensor
    noutput: size of output vector
    scope: optional scope name
    name: optional name for output tensor
    reverse: run in reverse

  Returns:
    Batch of size (batch_size, noutput).
  """
  with variable_scope.variable_scope(scope, "SequenceToFinal", [inputs]):
    length, batch_size, _ = _shape(inputs)
    lstm = rnn_cell.BasicLSTMCell(noutput, state_is_tuple=False)
    state = array_ops.zeros([batch_size, lstm.state_size])
    inputs_u = array_ops.unstack(inputs)
    if reverse:
      inputs_u = list(reversed(inputs_u))
    for i in xrange(length):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = lstm(inputs_u[i], state)
    outputs = array_ops.reshape(output, [batch_size, noutput], name=name)
    return outputs


def sequence_softmax(inputs, noutput, scope=None, name=None, linear_name=None):
  """Run a softmax layer over all the time steps of an input sequence.

  Args:
    inputs: (length, batch_size, depth) tensor
    noutput: output depth
    scope: optional scope name
    name: optional name for output tensor
    linear_name: name for linear (pre-softmax) output

  Returns:
    A tensor of size (length, batch_size, noutput).

  """
  length, _, ninputs = _shape(inputs)
  inputs_u = array_ops.unstack(inputs)
  output_u = []
  with variable_scope.variable_scope(scope, "SequenceSoftmax", [inputs]):
    initial_w = random_ops.truncated_normal([0 + ninputs, noutput], stddev=0.1)
    initial_b = constant_op.constant(0.1, shape=[noutput])
    w = variables.model_variable("weights", initializer=initial_w)
    b = variables.model_variable("biases", initializer=initial_b)
    for i in xrange(length):
      with variable_scope.variable_scope(scope, "SequenceSoftmaxStep",
                                         [inputs_u[i]]):
        # TODO(tmb) consider using slim.fully_connected(...,
        # activation_fn=tf.nn.softmax)
        linear = nn_ops.xw_plus_b(inputs_u[i], w, b, name=linear_name)
        output = nn_ops.softmax(linear)
        output_u += [output]
    outputs = array_ops.stack(output_u, name=name)
  return outputs
