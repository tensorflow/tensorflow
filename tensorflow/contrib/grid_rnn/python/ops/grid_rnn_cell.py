# Copyright 2016 Google Inc. All Rights Reserved.
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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import layers


class GridRNNCell(rnn_cell.RNNCell):
  """Grid recurrent cell.

  This implementation is based on:

    http://arxiv.org/pdf/1507.01526v3.pdf

    This is the generic implementation of GridRNN. Users can specify arbitrary number of dimensions,
    set some of them to be priority (section 3.2), non-recurrent (section 3.3)
    and input/output dimensions (section 3.4).
    Weight sharing can also be specified using the `tied` parameter.
    Type of recurrent units can be specified via `cell_fn`.
  """

  def __init__(self, num_units, num_dims=1, input_dims=None, output_dims=None, priority_dims=None,
               non_recurrent_dims=None, tied=False, cell_fn=None, non_recurrent_fn=None):
    """Initialize the parameters of a Grid RNN cell

    Args:
      num_units: int, The number of units in all dimensions of this GridRNN cell
      num_dims: int, Number of dimensions of this grid.
      input_dims: int or list, List of dimensions which will receive input data.
      output_dims: int or list, List of dimensions from which the output will be recorded.
      priority_dims: int or list, List of dimensions to be considered as priority dimensions.
              If None, no dimension is prioritized.
      non_recurrent_dims: int or list, List of dimensions that are not recurrent.
              The transfer function for non-recurrent dimensions is specified via `non_recurrent_fn`,
              which is default to be `tensorflow.nn.relu`.
      tied: bool, Whether to share the weights among the dimensions of this GridRNN cell.
              If there are non-recurrent dimensions in the grid, weights are shared between each
              group of recurrent and non-recurrent dimensions.
      cell_fn: function, a function which returns the recurrent cell object. Has to be in the following signature:
              def cell_func(num_units, input_size):
                # ...

              and returns an object of type `RNNCell`. If None, LSTMCell with default parameters will be used.
      non_recurrent_fn: a tensorflow Op that will be the transfer function of the non-recurrent dimensions
    """
    if num_dims < 1:
      raise ValueError('dims must be >= 1: {}'.format(num_dims))

    self._config = _parse_rnn_config(num_dims, input_dims, output_dims, priority_dims,
                                     non_recurrent_dims, non_recurrent_fn or nn.relu, tied, num_units)

    cell_input_size = (self._config.num_dims - 1) * num_units
    if cell_fn is None:
      self._cell = rnn_cell.LSTMCell(num_units=num_units, input_size=cell_input_size)
    else:
      self._cell = cell_fn(num_units, cell_input_size)
      if not isinstance(self._cell, rnn_cell.RNNCell):
        raise ValueError('cell_fn must return an object of type RNNCell')

  @property
  def input_size(self):
    # temporarily using num_units as the input_size of each dimension.
    # The actual input size only determined when this cell get invoked,
    # so this information can be considered unreliable.
    return self._config.num_units * len(self._config.inputs)

  @property
  def output_size(self):
    return self._cell.output_size * len(self._config.outputs)

  @property
  def state_size(self):
    return self._cell.state_size * len(self._config.recurrents)

  def __call__(self, inputs, state, scope=None):
    """Run one step of GridRNN.

    Args:
      inputs: input Tensor, 2D, batch x input_size. Or None
      state: state Tensor, 2D, batch x state_size. Note that state_size = cell_state_size * recurrent_dims
      scope: VariableScope for the created subgraph; defaults to "GridRNNCell".

    Returns:
      A tuple containing:
      - A 2D, batch x output_size, Tensor representing the output of the cell
        after reading "inputs" when previous state was "state".
      - A 2D, batch x state_size, Tensor representing the new state of the cell
        after reading "inputs" when previous state was "state".
    """
    state_sz = state.get_shape().as_list()[1]
    if self.state_size != state_sz:
      raise ValueError('Actual state size not same as specified: {} vs {}.'.format(state_sz, self.state_size))

    conf = self._config
    dtype = inputs.dtype if inputs is not None else state.dtype

    # c_prev is `m`, and m_prev is `h` in the paper. Keep c and m here for consistency with the codebase
    c_prev = [None] * self._config.num_dims
    m_prev = [None] * self._config.num_dims
    cell_output_size = self._cell.state_size - conf.num_units

    # for LSTM   : state = memory cell + output, hence cell_output_size > 0
    # for GRU/RNN: state = output (whose size is equal to _num_units), hence cell_output_size = 0
    for recurrent_dim, start_idx in zip(self._config.recurrents, range(0, self.state_size, self._cell.state_size)):
      if cell_output_size > 0:
        c_prev[recurrent_dim] = array_ops.slice(state, [0, start_idx], [-1, conf.num_units])
        m_prev[recurrent_dim] = array_ops.slice(state, [0, start_idx + conf.num_units], [-1, cell_output_size])
      else:
        m_prev[recurrent_dim] = array_ops.slice(state, [0, start_idx], [-1, conf.num_units])

    new_output = [None] * conf.num_dims
    new_state = [None] * conf.num_dims

    with vs.variable_scope(scope or type(self).__name__):  # GridRNNCell

      # project input
      if inputs is not None and sum(inputs.get_shape().as_list()) > 0 and len(conf.inputs) > 0:
        input_splits = array_ops.split(1, len(conf.inputs), inputs)
        input_sz = input_splits[0].get_shape().as_list()[1]

        for i, j in enumerate(conf.inputs):
          input_project_m = vs.get_variable('project_m_{}'.format(j), [input_sz, conf.num_units], dtype=dtype)
          m_prev[j] = math_ops.matmul(input_splits[i], input_project_m)

          if cell_output_size > 0:
            input_project_c = vs.get_variable('project_c_{}'.format(j), [input_sz, conf.num_units], dtype=dtype)
            c_prev[j] = math_ops.matmul(input_splits[i], input_project_c)


      _propagate(conf.non_priority, conf, self._cell, c_prev, m_prev, new_output, new_state, True)
      _propagate(conf.priority, conf, self._cell, c_prev, m_prev, new_output, new_state, False)

      output_tensors = [new_output[i] for i in self._config.outputs]
      output = array_ops.zeros([0, 0], dtype) if len(output_tensors) == 0 else array_ops.concat(1,
                                                                                                       output_tensors)

      state_tensors = [new_state[i] for i in self._config.recurrents]
      states = array_ops.zeros([0, 0], dtype) if len(state_tensors) == 0 else array_ops.concat(1, state_tensors)

    return output, states


"""
Specialized cells, for convenience
"""

class Grid1BasicRNNCell(GridRNNCell):
  """1D BasicRNN cell"""
  def __init__(self, num_units):
    super(Grid1BasicRNNCell, self).__init__(num_units=num_units, num_dims=1,
                                            input_dims=0, output_dims=0, priority_dims=0, tied=False,
                                            cell_fn=lambda n, i: rnn_cell.BasicRNNCell(num_units=n, input_size=i))


class Grid2BasicRNNCell(GridRNNCell):
  """2D BasicRNN cell
  This creates a 2D cell which receives input and gives output in the first dimension.
  The first dimension can optionally be non-recurrent if `non_recurrent_fn` is specified.
  """
  def __init__(self, num_units, tied=False, non_recurrent_fn=None):
    super(Grid2BasicRNNCell, self).__init__(num_units=num_units, num_dims=2,
                                            input_dims=0, output_dims=0, priority_dims=0, tied=tied,
                                            non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                            cell_fn=lambda n, i: rnn_cell.BasicRNNCell(num_units=n, input_size=i),
                                            non_recurrent_fn=non_recurrent_fn)


class Grid1BasicLSTMCell(GridRNNCell):
  """1D BasicLSTM cell"""
  def __init__(self, num_units, forget_bias=1):
    super(Grid1BasicLSTMCell, self).__init__(num_units=num_units, num_dims=1,
                                             input_dims=0, output_dims=0, priority_dims=0, tied=False,
                                             cell_fn=lambda n, i: rnn_cell.BasicLSTMCell(num_units=n,
                                                                                forget_bias=forget_bias, input_size=i))


class Grid2BasicLSTMCell(GridRNNCell):
  """2D BasicLSTM cell
    This creates a 2D cell which receives input and gives output in the first dimension.
    The first dimension can optionally be non-recurrent if `non_recurrent_fn` is specified.
  """
  def __init__(self, num_units, tied=False, non_recurrent_fn=None, forget_bias=1):
    super(Grid2BasicLSTMCell, self).__init__(num_units=num_units, num_dims=2,
                                             input_dims=0, output_dims=0, priority_dims=0, tied=tied,
                                             non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                             cell_fn=lambda n, i: rnn_cell.BasicLSTMCell(
                                               num_units=n, forget_bias=forget_bias, input_size=i),
                                             non_recurrent_fn=non_recurrent_fn)


class Grid1LSTMCell(GridRNNCell):
  """1D LSTM cell
    This is different from Grid1BasicLSTMCell because it gives options to specify the forget bias and enabling peepholes
  """
  def __init__(self, num_units, use_peepholes=False, forget_bias=1.0):
    super(Grid1LSTMCell, self).__init__(num_units=num_units, num_dims=1,
                                        input_dims=0, output_dims=0, priority_dims=0,
                                        cell_fn=lambda n, i: rnn_cell.LSTMCell(
                                          num_units=n, input_size=i, use_peepholes=use_peepholes,
                                          forget_bias=forget_bias))


class Grid2LSTMCell(GridRNNCell):
  """2D LSTM cell
    This creates a 2D cell which receives input and gives output in the first dimension.
    The first dimension can optionally be non-recurrent if `non_recurrent_fn` is specified.
  """
  def __init__(self, num_units, tied=False, non_recurrent_fn=None,
               use_peepholes=False, forget_bias=1.0):
    super(Grid2LSTMCell, self).__init__(num_units=num_units, num_dims=2,
                                        input_dims=0, output_dims=0, priority_dims=0, tied=tied,
                                        non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                        cell_fn=lambda n, i: rnn_cell.LSTMCell(
                                          num_units=n, input_size=i, forget_bias=forget_bias,
                                          use_peepholes=use_peepholes),
                                        non_recurrent_fn=non_recurrent_fn)


class Grid3LSTMCell(GridRNNCell):
  """3D BasicLSTM cell
    This creates a 2D cell which receives input and gives output in the first dimension.
    The first dimension can optionally be non-recurrent if `non_recurrent_fn` is specified.
    The second and third dimensions are LSTM.
  """
  def __init__(self, num_units, tied=False, non_recurrent_fn=None,
               use_peepholes=False, forget_bias=1.0):
    super(Grid3LSTMCell, self).__init__(num_units=num_units, num_dims=3,
                                        input_dims=0, output_dims=0, priority_dims=0, tied=tied,
                                        non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                        cell_fn=lambda n, i: rnn_cell.LSTMCell(
                                          num_units=n, input_size=i, forget_bias=forget_bias,
                                          use_peepholes=use_peepholes),
                                        non_recurrent_fn=non_recurrent_fn)

class Grid2GRUCell(GridRNNCell):
  """2D LSTM cell
    This creates a 2D cell which receives input and gives output in the first dimension.
    The first dimension can optionally be non-recurrent if `non_recurrent_fn` is specified.
  """

  def __init__(self, num_units, tied=False, non_recurrent_fn=None):
    super(Grid2GRUCell, self).__init__(num_units=num_units, num_dims=2,
                                       input_dims=0, output_dims=0, priority_dims=0, tied=tied,
                                       non_recurrent_dims=None if non_recurrent_fn is None else 0,
                                       cell_fn=lambda n, i: rnn_cell.GRUCell(num_units=n, input_size=i),
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


def _propagate(dim_indices, conf, cell, c_prev, m_prev, new_output, new_state, first_call):
  """
  Propagates through all the cells in dim_indices dimensions.
  """
  if len(dim_indices) == 0:
    return

  # Because of the way RNNCells are implemented, we take the last dimension (H_{N-1}) out
  # and feed it as the state of the RNN cell (in `last_dim_output`)
  # The input of the cell (H_0 to H_{N-2}) are concatenated into `cell_inputs`
  if conf.num_dims > 1:
    ls_cell_inputs = [None] * (conf.num_dims - 1)
    for d in conf.dims[:-1]:
      ls_cell_inputs[d.idx] = new_output[d.idx] if new_output[d.idx] is not None else m_prev[d.idx]
    cell_inputs = array_ops.concat(1, ls_cell_inputs)
  else:
    cell_inputs = array_ops.zeros([m_prev[0].get_shape().as_list()[0], 0], m_prev[0].dtype)

  last_dim_output = new_output[-1] if new_output[-1] is not None else m_prev[-1]

  for i in dim_indices:
    d = conf.dims[i]
    if d.non_recurrent_fn:
      linear_args = array_ops.concat(1, [cell_inputs, last_dim_output]) if conf.num_dims > 1 else last_dim_output
      with vs.variable_scope('non_recurrent' if conf.tied else 'non_recurrent/cell_{}'.format(i)):
        if conf.tied and not(first_call and i == dim_indices[0]):
          vs.get_variable_scope().reuse_variables()
        new_output[d.idx] = layers.fully_connected(linear_args, num_output_units=conf.num_units,
                                                   activation_fn=d.non_recurrent_fn,
                                                   weight_init=vs.get_variable_scope().initializer or
                                                               layers.initializers.xavier_initializer)
    else:
      if c_prev[i] is not None:
        cell_state = array_ops.concat(1, [c_prev[i], last_dim_output])
      else:
        # for GRU/RNN, the state is just the previous output
        cell_state = last_dim_output

      with vs.variable_scope('recurrent' if conf.tied else 'recurrent/cell_{}'.format(i)):
        if conf.tied and not (first_call and i == dim_indices[0]):
          vs.get_variable_scope().reuse_variables()
        new_output[d.idx], new_state[d.idx] = cell(cell_inputs, cell_state)
