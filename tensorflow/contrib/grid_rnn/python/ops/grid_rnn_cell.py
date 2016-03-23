# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Module for constructing GridRNN cells"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


# TODO: update docs

class GridRNNCell(rnn_cell.RNNCell):
  """Grid Long short-term memory unit (GridLSTM) recurrent network cell.

  This implementation is based on:

    http://arxiv.org/pdf/1507.01526v3.pdf

    By default, a GridRNN cell has a depth dimension, which is also the "priority dimension" (section 3.2).
    Therefore an N-RNN cell is defined with dims = N-1.
    The length of the depth dimension is specified by the `depth` parameter in the constructor of this class.
    Users can also specify an `Op` that will be used along the depth dimension, instead of recurrent. This is called
     "Non-LSTM dimension" in section 3.3 of the paper.
  """

  def __init__(self, num_units, input_size=None, num_dims=1, input_dims=None, output_dims=None, priority_dims=None,
               non_recurrent_dims=None, tied=False, cell_fn=None, non_recurrent_fn=None):
    """Initialize the parameters for a Grid RNN cell

    def conversion_func(value, dtype=None, name=None, as_ref=False):
      # ...

    Args:
      num_units: int, The number of units in the cells in all dimensions of this GridRNN cell
      input_size: int, The dimensionality of the inputs into the cells
      cell_class: class, the cell class used in this Grid cell, should be a subclass of :class:RNNCell
      num_dims: int, number of dimensions (excluding the depth dimension).
      tied: bool, whether to share the weights among all the dimensions.
      cell_kwargs: dict, additional arguments passed to the constructor of cell.
    """
    if num_dims < 1:
      raise ValueError('dims must be >= 1: {}'.format(num_dims))

    self._config = _parse_rnn_config(num_dims, input_dims, output_dims, priority_dims,
                                     non_recurrent_dims, non_recurrent_fn or nn.relu, tied, num_units)
    self._input_size = input_size or num_units

    cell_input_size = (self._config.num_dims - 1) * num_units
    if cell_fn is None:
      self._cell = rnn_cell.LSTMCell(num_units=num_units, input_size=cell_input_size)
    else:
      self._cell = cell_fn(num_units, cell_input_size)
      if not isinstance(self._cell, rnn_cell.RNNCell):
        raise ValueError('cell_fn must return an object of type RNNCell')

  @property
  def input_size(self):
    return self._input_size * len(self._config.inputs)

  @property
  def output_size(self):
    return self._cell.output_size * len(self._config.outputs)

  @property
  def state_size(self):
    return self._cell.state_size * len(self._config.recurrents)

  def __call__(self, inputs, state, scope=None):
    """Run one step of GridRNN.

    Args:
      inputs: input Tensor, 2D, batch x input_size.
      state: state Tensor, 2D, batch x state_size. Note that state_size = cell_state_size * recurrent_dims
      scope: VariableScope for the created subgraph; defaults to "GridRNNCell".

    Returns:
      A tuple containing:
      - A 2D, batch x output_size, Tensor representing the output of the cell
        after reading "inputs" when previous state was "state".
      - A 2D, batch x state_size, Tensor representing the new state of the cell
        after reading "inputs" when previous state was "state".
    """
    sz = inputs.get_shape().as_list()[1]
    if self.input_size != sz:
      raise ValueError('Actual input size not same as specified: {} vs {}.'.format(sz, self.input_size))
    sz = state.get_shape().as_list()[1]
    if self.state_size != sz:
      raise ValueError('Actual state size not same as specified: {} vs {}.'.format(sz, self.state_size))

    # c_prev is `m`, and m_prev is `h` in the paper. Keep c and m here for consistency with the codebase
    c_prev = [None] * self._config.num_dims
    m_prev = [None] * self._config.num_dims
    cell_units = self._cell.state_size - self._cell.output_size

    # for LSTM cell: state_size = num_units + output_size, because state = concat(cell_values + previous output)
    # for GRU/RNN: state_size = output_size, because state = previous output
    for recurrent_dim, start_idx in zip(self._config.recurrents, range(0, self.state_size, self._cell.state_size)):
      c_prev[recurrent_dim] = array_ops.slice(state, [0, start_idx], [-1, cell_units])
      m_prev[recurrent_dim] = array_ops.slice(state, [0, start_idx + cell_units], [-1, self._cell.output_size])

    conf = self._config

    # input dimensions
    dtype = inputs.dtype
    input_splits = array_ops.split(1, len(conf.inputs), inputs)

    new_output = [None] * conf.num_dims
    new_state = [None] * conf.num_dims

    with vs.variable_scope(scope or type(self).__name__) as grid_scope:  # GridRNNCell
      for i, j in enumerate(conf.inputs):
        input_project_c = vs.get_variable('project_c_{}'.format(j), [self._input_size, conf.num_units], dtype=dtype)
        input_project_m = vs.get_variable('project_m_{}'.format(j), [self._input_size, conf.num_units], dtype=dtype)
        c_prev[j] = math_ops.matmul(input_splits[i], input_project_c)
        m_prev[j] = math_ops.matmul(input_splits[i], input_project_m)

      _propagate(conf.non_priority, conf, self._cell, c_prev, m_prev, new_output, new_state, True)
      _propagate(conf.priority, conf, self._cell, c_prev, m_prev, new_output, new_state, False)

      output_tensors = [new_output[i] for i in self._config.outputs]
      output = array_ops.zeros([0, 0], inputs.dtype) if len(output_tensors) == 0 else array_ops.concat(1,
                                                                                                       output_tensors)

      state_tensors = [new_state[i] for i in self._config.recurrents]
      states = array_ops.zeros([0, 0], inputs.dtype) if len(state_tensors) == 0 else array_ops.concat(1, state_tensors)

      grid_scope.reuse_variables()

    return output, states


class Grid1BasicRNNCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid1BasicRNNCell, self).__init__(num_units=num_units, num_dims=1, input_dims=0, output_dims=0,
                                            priority_dims=0, tied=tied,
                                            cell_fn=lambda n, i: rnn_cell.BasicRNNCell(num_units=n, input_size=i))


class Grid2BasicRNNCell(GridRNNCell):
  def __init__(self, num_units, tied=False, non_recurrent_fn=None):
    super(Grid2BasicRNNCell, self).__init__(num_units=num_units, num_dims=2, input_dims=0, output_dims=0,
                                            priority_dims=0, tied=tied,
                                            non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                            cell_fn=lambda n, i: rnn_cell.BasicRNNCell(num_units=n, input_size=i),
                                            non_recurrent_fn=non_recurrent_fn)


class Grid1BasicLSTMCell(GridRNNCell):
  def __init__(self, num_units, forget_bias=1, tied=False):
    super(Grid1BasicLSTMCell, self).__init__(num_units=num_units, num_dims=1, input_dims=0, output_dims=0,
                                             priority_dims=0, tied=tied,
                                             cell_fn=lambda n, i: rnn_cell.BasicLSTMCell(num_units=n,
                                                                                forget_bias=forget_bias, input_size=i))


class Grid2BasicLSTMCell(GridRNNCell):
  def __init__(self, num_units, input_size=None, tied=False, non_recurrent_fn=None, forget_bias=1):
    super(Grid2BasicLSTMCell, self).__init__(num_units=num_units, input_size=input_size, num_dims=2,
                                             input_dims=0, output_dims=0, priority_dims=0, tied=tied,
                                             non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                             cell_fn=lambda n, i: rnn_cell.BasicLSTMCell(
                                               num_units=n, forget_bias=forget_bias, input_size=i),
                                             non_recurrent_fn=non_recurrent_fn)


class Grid1LSTMCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid1LSTMCell, self).__init__(num_units=num_units, num_dims=1, input_dims=0, output_dims=0, priority_dims=0,
                                        tied=tied, cell_fn=lambda n, i: rnn_cell.LSTMCell(num_units=n, input_size=i))


class Grid2LSTMCell(GridRNNCell):
  def __init__(self, num_units, input_size=None, tied=False, non_recurrent_fn=None,
               use_peepholes=False, forget_bias=1.0):
    super(Grid2LSTMCell, self).__init__(num_units=num_units, input_size=input_size, num_dims=2,
                                        input_dims=0, output_dims=0, priority_dims=0, tied=tied,
                                        non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                        cell_fn=lambda n, i: rnn_cell.LSTMCell(
                                          num_units=n, input_size=i, forget_bias=forget_bias,
                                          use_peepholes=use_peepholes),
                                        non_recurrent_fn=non_recurrent_fn)


"""
Helpers
"""

_GridRNNDimension = namedtuple('_GridRNNDimension', ['idx', 'is_input', 'is_output', 'is_priority', 'non_recurrent_fn'])

_GridRNNConfig = namedtuple('_GridRNNConfig', ['num_dims', 'dims',
                                               'inputs', 'outputs', 'recurrents',
                                               'priority', 'non_priority', 'tied', 'num_units'])


def _parse_rnn_config(num_dims, ls_input_dims, ls_output_dims, ls_priority_dims, ls_non_recurrent_dims,
                      non_recurrent_fn, tied, num_units):
  def check_dim_list(ls):
    if ls is None:
      ls = []
    if not isinstance(ls, (list, tuple)):
      ls = [ls]
    ls = sorted(set(ls))
    if any(_ < 0 or _ >= num_dims for _ in ls):
      raise ValueError('Invalid dims: {}. Must be in [0, {})'.format(ls, num_dims))
    return ls

  input_dims = check_dim_list(ls_input_dims)
  output_dims = check_dim_list(ls_output_dims)
  priority_dims = check_dim_list(ls_priority_dims)
  non_recurrent_dims = check_dim_list(ls_non_recurrent_dims)

  rnn_dims = []
  for i in range(num_dims):
    rnn_dims.append(_GridRNNDimension(idx=i, is_input=(i in input_dims), is_output=(i in output_dims),
                                      is_priority=(i in priority_dims),
                                      non_recurrent_fn=non_recurrent_fn if i in non_recurrent_dims else None))
  return _GridRNNConfig(num_dims=num_dims, dims=rnn_dims, inputs=input_dims, outputs=output_dims,
                        recurrents=[x for x in range(num_dims) if x not in non_recurrent_dims],
                        priority=priority_dims,
                        non_priority=[x for x in range(num_dims) if x not in priority_dims],
                        tied=tied, num_units=num_units)


def _get_first(*args):
  return next((i for i in args if i is not None), None)


def _propagate(dim_indices, conf, cell, c_prev, m_prev, new_output, new_state, first_call):
  """
  """
  if len(dim_indices) == 0:
    return

  ls_cell_inputs = [None] * (conf.num_dims - 1)
  for d in conf.dims[:-1]:
    ls_cell_inputs[d.idx] = _get_first(new_output[d.idx], m_prev[d.idx])
  cell_inputs = array_ops.concat(1, ls_cell_inputs)
  last_dim_output = _get_first(new_output[-1], m_prev[-1])

  for i in dim_indices:
    d = conf.dims[i]
    if d.non_recurrent_fn:
      if conf.tied:
        with vs.variable_scope('non_recurrent', reuse=not(first_call and i == dim_indices[0])):
          new_output[d.idx] = d.non_recurrent_fn(rnn_cell.linear(args=[cell_inputs, last_dim_output],
                                                                 output_size=conf.num_units, bias=True))
      else:
          new_output[d.idx] = d.non_recurrent_fn(rnn_cell.linear(args=[cell_inputs, last_dim_output],
                                                                 output_size=conf.num_units, bias=True,
                                                                 scope='non_recurrent/cell_{}'.format(i)))
      new_output[d.idx] = logging_ops.Print(new_output[d.idx], [new_output[d.idx], cell_inputs, last_dim_output])
    else:
      cell_state = array_ops.concat(1, [c_prev[i], last_dim_output])
      if conf.tied:
        with vs.variable_scope('recurrent', reuse=not(first_call and i == dim_indices[0])):
          a, b = cell(cell_inputs, cell_state)
      else:
        a, b = cell(cell_inputs, cell_state, scope='recurrent/cell_{}'.format(i))

      a = logging_ops.Print(a, [cell_state, cell_inputs, a, b], message='tensor{} '.format(i), summarize=100)
      new_output[d.idx], new_state[d.idx] = a, b
