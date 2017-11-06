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
"""Module for constructing Multi-Dimentional LSTM cells"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops

def ln(tensor, scope=None, epsilon=1e-5):
  """ Layer normalizes a 2D tensor along its second axis """
  assert len(tensor.get_shape()) == 2
  m, v = nn.moments(tensor, [1], keep_dims=True)
  if not isinstance(scope, str):
    scope = ''
  with vs.variable_scope(scope + 'layer_norm'):
    scale = vs.get_variable('scale',
                            shape=[tensor.get_shape()[1]],
                            initializer=init_ops.constant_initializer(1))
    shift = vs.get_variable('shift',
                            shape=[tensor.get_shape()[1]],
                            initializer=init_ops.constant_initializer(0))
  ln_initial = (tensor - m) / math_ops.sqrt(v + epsilon)

  return ln_initial * scale + shift


class MultiDimentionalLSTMCell(rnn.RNNCell):
  """
  Adapted from TF's BasicLSTMCell to use Layer Normalization.
  Note that state_is_tuple is always True.
  """

  def __init__(self, num_units, forget_bias=0.0, activation=nn.tanh):
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    return rnn.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: (batch,n) tensor
      state: the states and hidden unit of the two cells

    Returns:
      new_state, new_inputs
    """
    with vs.variable_scope(scope or type(self).__name__):
      c1, c2, h1, h2 = state

      # change bias argument to False since LN will add bias via shift
      concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

      i, j, f1, f2, o = array_ops.split(concat, 5, 1)

      # add layer normalization to each gate
      i = ln(i, scope='i/')
      j = ln(j, scope='j/')
      f1 = ln(f1, scope='f1/')
      f2 = ln(f2, scope='f2/')
      o = ln(o, scope='o/')

      new_c = (c1 * nn.sigmoid(f1 + self._forget_bias) +
               c2 * nn.sigmoid(f2 + self._forget_bias) +
               nn.sigmoid(i) * self._activation(j))

      # add layer_normalization in calculation of new hidden state
      new_h = self._activation(ln(new_c, scope='new_h/')) * nn.sigmoid(o)
      new_state = rnn.LSTMStateTuple(new_c, new_h)

      return new_h, new_state


def multi_dimentional_rnn(rnn_size,
                          input_data,
                          sh,
                          dims=None,
                          scope_name="layer1"):
  """Implements naive multidimentional recurent neural networks

  Args:
    rnn_size: int, the hidden units
    input_data: (num_images, height, width, depth) tensor,
    the data to process of shape
    sh: list, [heigth,width] of the windows
    dims: list, dimentions to reverse the input data
    scope_name: string,  the scope

  Returns:
    (batch,h/sh[0],w/sh[1],chanels*sh[0]*sh[1]) tensor
  """
  with vs.variable_scope("MultiDimentionalLSTMCell-"+scope_name):
    cell = MultiDimentionalLSTMCell(rnn_size)

    shape = input_data.get_shape().as_list()

    #pad if the dimmention are not correct for the block size
    if shape[1]%sh[0] != 0:
      offset = array_ops.zeros([shape[0], sh[0]-(shape[1]%sh[0]),
                                shape[2],
                                shape[3]])
      input_data = array_ops.concat([input_data, offset], 1)
      shape = input_data.get_shape().as_list()
    if shape[2]%sh[1] != 0:
      offset = array_ops.zeros([shape[0],
                                shape[1], sh[1]-(shape[2]%sh[1]),
                                shape[3]])
      input_data = array_ops.concat([input_data, offset], 2)
      shape = input_data.get_shape().as_list()

    h, w = int(shape[1]/sh[0]), int(shape[2]/sh[1])
    features = sh[1]*sh[0]*shape[3]
    batch_size = shape[0]

    lines = array_ops.split(input_data, h, axis=1)
    line_blocks = []
    for line in lines:
      line = array_ops.transpose(line, [0, 2, 3, 1])
      line = array_ops.reshape(line, [batch_size, w, features])
      line_blocks.append(line)
    x = array_ops.stack(line_blocks, axis=1)
    if dims is not None:
      x = array_ops.reverse(x, dims)
    x = array_ops.transpose(x, [1, 2, 0, 3])
    x = array_ops.reshape(x, [-1, features])
    x = array_ops.split(x, h*w, 0)

    inputs_ta = tensor_array_ops.TensorArray(dtype=dtypes.float32,
                                             size=h*w,
                                             name='input_ta')
    inputs_ta = inputs_ta.unstack(x)
    states_ta = tensor_array_ops.TensorArray(dtype=dtypes.float32,
                                             size=h*w+1,
                                             name='state_ta',
                                             clear_after_read=False)
    outputs_ta = tensor_array_ops.TensorArray(dtype=dtypes.float32,
                                              size=h*w,
                                              name='output_ta')

    states_ta = states_ta.write(h*w,
                                rnn.LSTMStateTuple(
                                    array_ops.zeros([batch_size, rnn_size],
                                                    dtypes.float32),
                                    array_ops.zeros([batch_size, rnn_size],
                                                    dtypes.float32)))
    def get_index_state_up(t, w):
      """get_index_state_up"""
      return control_flow_ops.cond(math_ops.less_equal(
          array_ops.constant(w),
          t),
                                   lambda: t-array_ops.constant(w),
                                   lambda: array_ops.constant(h*w))
    def get_index_state_last(t, w):
      """get_index_state_last"""
      return control_flow_ops.cond(math_ops.less(
          array_ops.constant(0),
          math_ops.mod(t,
                       array_ops.constant(w))),
                                   lambda: t-array_ops.constant(1),
                                   lambda: array_ops.constant(h*w))

    time = array_ops.constant(0)

    def body(time, outputs_ta, states_ta):
      """Implements multi dimmentions lstm while_loop

      Args:
        time: int
        outputs_ta: tensor_array
        states_ta: tensor_array
      """
      constant_val = array_ops.constant(0)
      state_up = control_flow_ops.cond(
          math_ops.less_equal(array_ops.constant(w),
                              time),
          lambda: states_ta.read(get_index_state_up(time,
                                                    w)),
          lambda: states_ta.read(h*w))
      state_last = control_flow_ops.cond(
          math_ops.less(constant_val,
                        math_ops.mod(time,
                                     array_ops.constant(w))),
          lambda: states_ta.read(get_index_state_last(time,
                                                      w)),
          lambda: states_ta.read(h*w))

      current_state = state_up[0], state_last[0], state_up[1], state_last[1]
      out, state = cell(inputs_ta.read(time), current_state)
      outputs_ta = outputs_ta.write(time, out)
      states_ta = states_ta.write(time, state)
      return time + 1, outputs_ta, states_ta

    def condition(time, outputs_ta, states_ta):
      return math_ops.less(time, array_ops.constant(h*w))

    _, outputs_ta, _ = control_flow_ops.while_loop(condition,
                                                   body,
                                                   [time,
                                                    outputs_ta,
                                                    states_ta],
                                                   parallel_iterations=1)


    outputs = outputs_ta.stack()

    outputs = array_ops.reshape(outputs, [h, w, batch_size, rnn_size])
    outputs = array_ops.transpose(outputs, [2, 0, 1, 3])
    if dims is not None:
      outputs = array_ops.reverse(outputs, dims)

    return outputs

def tanh_and_sum(rnn_size, input_data, sh, scope='mdLSTM'):
  """Sum and tanh over four directions for MDLSTM

  Args:
    rnn_size: int, the size of the cell
    input_data: (num_images, height, width, depth) tensor
    sh: list, first element - the height of the block,
    second - the width of the lstm block
    scope: string, the function scope

  Returns:
    (num_images, height/sh[0], width/sh[1], sh[0]*sh[1]) tensor,
    the mean over four iteration with MDLSTM
  """
  outs = []
  for i in range(2):
    for j in range(2):
      dims = []
      if i != 0:
        dims.append(1)
      if j != 0:
        dims.append(2)
        outputs = multi_dimentional_rnn(rnn_size, input_data, sh,
                                        dims, scope+"-l{0}".format(i*2+j))
        outs.append(outputs)

  outs = array_ops.stack(outs, axis=0)
  mean = math_ops.reduce_mean(outs, 0)
  return nn.tanh(mean)
