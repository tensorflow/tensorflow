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
"""Module for constructing GridRNN cells"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


class GridRNNCell(rnn.RNNCell):
  """Grid recurrent cell.

  This implementation is based on:

    http://arxiv.org/pdf/1507.01526v3.pdf

    This is the generic implementation of GridRNN. Users can specify arbitrary
    number of dimensions,
    set some of them to be priority (section 3.2), non-recurrent (section 3.3)
    and input/output dimensions (section 3.4).
    Weight sharing can also be specified using the `tied` parameter.
    Type of recurrent units can be specified via `cell_fn`.
  """

  def __init__(self,
               num_units,
               num_dims=1,
               input_dims=None,
               output_dims=None,
               priority_dims=None,
               non_recurrent_dims=None,
               tied=False,
               cell_fn=None,
               non_recurrent_fn=None,
               state_is_tuple=True,
               output_is_tuple=True):
    """Initialize the parameters of a Grid RNN cell

    Args:
      num_units: int, The number of units in all dimensions of this GridRNN cell
      num_dims: int, Number of dimensions of this grid.
      input_dims: int or list, List of dimensions which will receive input data.
      output_dims: int or list, List of dimensions from which the output will be
        recorded.
      priority_dims: int or list, List of dimensions to be considered as
        priority dimensions.
              If None, no dimension is prioritized.
      non_recurrent_dims: int or list, List of dimensions that are not
        recurrent.
              The transfer function for non-recurrent dimensions is specified
                via `non_recurrent_fn`, which is
                default to be `tensorflow.nn.relu`.
      tied: bool, Whether to share the weights among the dimensions of this
        GridRNN cell.
              If there are non-recurrent dimensions in the grid, weights are
                shared between each group of recurrent and non-recurrent
                dimensions.
      cell_fn: function, a function which returns the recurrent cell object.
        Has to be in the following signature:
              ```
              def cell_func(num_units):
                # ...
              ```
              and returns an object of type `RNNCell`. If None, LSTMCell with
                default parameters will be used.
        Note that if you use a custom RNNCell (with `cell_fn`), it is your
        responsibility to make sure the inner cell use `state_is_tuple=True`.

      non_recurrent_fn: a tensorflow Op that will be the transfer function of
        the non-recurrent dimensions
      state_is_tuple: If True, accepted and returned states are tuples of the
        states of the recurrent dimensions. If False, they are concatenated
        along the column axis. The latter behavior will soon be deprecated.

        Note that if you use a custom RNNCell (with `cell_fn`), it is your
        responsibility to make sure the inner cell use `state_is_tuple=True`.

      output_is_tuple: If True, the output is a tuple of the outputs of the
        recurrent dimensions. If False, they are concatenated along the
        column axis. The later behavior will soon be deprecated.

    Raises:
      TypeError: if cell_fn does not return an RNNCell instance.
    """
    if not state_is_tuple:
      logging.warning('%s: Using a concatenated state is slower and will '
                      'soon be deprecated.  Use state_is_tuple=True.', self)
    if not output_is_tuple:
      logging.warning('%s: Using a concatenated output is slower and will'
                      'soon be deprecated.  Use output_is_tuple=True.', self)

    if num_dims < 1:
      raise ValueError('dims must be >= 1: {}'.format(num_dims))

    self._config = _parse_rnn_config(num_dims, input_dims, output_dims,
                                     priority_dims, non_recurrent_dims,
                                     non_recurrent_fn or nn.relu, tied,
                                     num_units)

    self._state_is_tuple = state_is_tuple
    self._output_is_tuple = output_is_tuple

    if cell_fn is None:
      my_cell_fn = functools.partial(
          rnn.LSTMCell, num_units=num_units, state_is_tuple=state_is_tuple)
    else:
      my_cell_fn = lambda: cell_fn(num_units)
    if tied:
      self._cells = [my_cell_fn()] * num_dims
    else:
      self._cells = [my_cell_fn() for _ in range(num_dims)]
    if not isinstance(self._cells[0], rnn.RNNCell):
      raise TypeError('cell_fn must return an RNNCell instance, saw: %s' %
                      type(self._cells[0]))

    if self._output_is_tuple:
      self._output_size = tuple(self._cells[0].output_size
                                for _ in self._config.outputs)
    else:
      self._output_size = self._cells[0].output_size * len(self._config.outputs)

    if self._state_is_tuple:
      self._state_size = tuple(self._cells[0].state_size
                               for _ in self._config.recurrents)
    else:
      self._state_size = self._cell_state_size() * len(self._config.recurrents)

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of GridRNN.

    Args:
      inputs: input Tensor, 2D, batch x input_size. Or None
      state: state Tensor, 2D, batch x state_size. Note that state_size =
        cell_state_size * recurrent_dims
      scope: VariableScope for the created subgraph; defaults to "GridRNNCell".

    Returns:
      A tuple containing:

      - A 2D, batch x output_size, Tensor representing the output of the cell
        after reading "inputs" when previous state was "state".
      - A 2D, batch x state_size, Tensor representing the new state of the cell
        after reading "inputs" when previous state was "state".
    """
    conf = self._config
    dtype = inputs.dtype

    c_prev, m_prev, cell_output_size = self._extract_states(state)

    new_output = [None] * conf.num_dims
    new_state = [None] * conf.num_dims

    with vs.variable_scope(scope or type(self).__name__):  # GridRNNCell
      # project input, populate c_prev and m_prev
      self._project_input(inputs, c_prev, m_prev, cell_output_size > 0)

      # propagate along dimensions, first for non-priority dimensions
      # then priority dimensions
      _propagate(conf.non_priority, conf, self._cells, c_prev, m_prev,
                 new_output, new_state, True)
      _propagate(conf.priority, conf, self._cells,
                 c_prev, m_prev, new_output, new_state, False)

      # collect outputs and states
      output_tensors = [new_output[i] for i in self._config.outputs]
      if self._output_is_tuple:
        output = tuple(output_tensors)
      else:
        if output_tensors:
          output = array_ops.concat(output_tensors, 1)
        else:
          output = array_ops.zeros([0, 0], dtype)

      if self._state_is_tuple:
        states = tuple(new_state[i] for i in self._config.recurrents)
      else:
        # concat each state first, then flatten the whole thing
        state_tensors = [
            x for i in self._config.recurrents for x in new_state[i]
        ]
        if state_tensors:
          states = array_ops.concat(state_tensors, 1)
        else:
          states = array_ops.zeros([0, 0], dtype)

    return output, states

  def _extract_states(self, state):
    """Extract the cell and previous output tensors from the given state.

    Args:
      state: The RNN state.

    Returns:
      Tuple of the cell value, previous output, and cell_output_size.

    Raises:
      ValueError: If len(self._config.recurrents) != len(state).
    """
    conf = self._config

    # c_prev is `m` (cell value), and
    # m_prev is `h` (previous output) in the paper.
    # Keeping c and m here for consistency with the codebase
    c_prev = [None] * conf.num_dims
    m_prev = [None] * conf.num_dims

    # for LSTM   : state = memory cell + output, hence cell_output_size > 0
    # for GRU/RNN: state = output (whose size is equal to _num_units),
    #              hence cell_output_size = 0
    total_cell_state_size = self._cell_state_size()
    cell_output_size = total_cell_state_size - conf.num_units

    if self._state_is_tuple:
      if len(conf.recurrents) != len(state):
        raise ValueError('Expected state as a tuple of {} '
                         'element'.format(len(conf.recurrents)))

      for recurrent_dim, recurrent_state in zip(conf.recurrents, state):
        if cell_output_size > 0:
          c_prev[recurrent_dim], m_prev[recurrent_dim] = recurrent_state
        else:
          m_prev[recurrent_dim] = recurrent_state
    else:
      for recurrent_dim, start_idx in zip(conf.recurrents,
                                          range(0, self.state_size,
                                                total_cell_state_size)):
        if cell_output_size > 0:
          c_prev[recurrent_dim] = array_ops.slice(state, [0, start_idx],
                                                  [-1, conf.num_units])
          m_prev[recurrent_dim] = array_ops.slice(
              state, [0, start_idx + conf.num_units], [-1, cell_output_size])
        else:
          m_prev[recurrent_dim] = array_ops.slice(state, [0, start_idx],
                                                  [-1, conf.num_units])
    return c_prev, m_prev, cell_output_size

  def _project_input(self, inputs, c_prev, m_prev, with_c):
    """Fills in c_prev and m_prev with projected input, for input dimensions.

    Args:
      inputs: inputs tensor
      c_prev: cell value
      m_prev: previous output
      with_c: boolean; whether to include project_c.

    Raises:
      ValueError: if len(self._config.input) != len(inputs)
    """
    conf = self._config

    if (inputs is not None and inputs.get_shape().with_rank(2)[1].value > 0 and
        conf.inputs):
      if isinstance(inputs, tuple):
        if len(conf.inputs) != len(inputs):
          raise ValueError('Expect inputs as a tuple of {} '
                           'tensors'.format(len(conf.inputs)))
        input_splits = inputs
      else:
        input_splits = array_ops.split(
            value=inputs, num_or_size_splits=len(conf.inputs), axis=1)
      input_sz = input_splits[0].get_shape().with_rank(2)[1].value

      for i, j in enumerate(conf.inputs):
        input_project_m = vs.get_variable(
            'project_m_{}'.format(j), [input_sz, conf.num_units],
            dtype=inputs.dtype)
        m_prev[j] = math_ops.matmul(input_splits[i], input_project_m)

        if with_c:
          input_project_c = vs.get_variable(
              'project_c_{}'.format(j), [input_sz, conf.num_units],
              dtype=inputs.dtype)
          c_prev[j] = math_ops.matmul(input_splits[i], input_project_c)

  def _cell_state_size(self):
    """Total size of the state of the inner cell used in this grid.

    Returns:
      Total size of the state of the inner cell.
    """
    state_sizes = self._cells[0].state_size
    if isinstance(state_sizes, tuple):
      return sum(state_sizes)
    return state_sizes


"""Specialized cells, for convenience
"""


class Grid1BasicRNNCell(GridRNNCell):
  """1D BasicRNN cell"""

  def __init__(self, num_units, state_is_tuple=True, output_is_tuple=True):
    super(Grid1BasicRNNCell, self).__init__(
        num_units=num_units,
        num_dims=1,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        tied=False,
        cell_fn=lambda n: rnn.BasicRNNCell(num_units=n),
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


class Grid2BasicRNNCell(GridRNNCell):
  """2D BasicRNN cell

  This creates a 2D cell which receives input and gives output in the first
  dimension.

  The first dimension can optionally be non-recurrent if `non_recurrent_fn` is
  specified.
  """

  def __init__(self,
               num_units,
               tied=False,
               non_recurrent_fn=None,
               state_is_tuple=True,
               output_is_tuple=True):
    super(Grid2BasicRNNCell, self).__init__(
        num_units=num_units,
        num_dims=2,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        tied=tied,
        non_recurrent_dims=None if non_recurrent_fn is None else 0,
        cell_fn=lambda n: rnn.BasicRNNCell(num_units=n),
        non_recurrent_fn=non_recurrent_fn,
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


class Grid1BasicLSTMCell(GridRNNCell):
  """1D BasicLSTM cell."""

  def __init__(self,
               num_units,
               forget_bias=1,
               state_is_tuple=True,
               output_is_tuple=True):
    def cell_fn(n):
      return rnn.BasicLSTMCell(num_units=n, forget_bias=forget_bias)
    super(Grid1BasicLSTMCell, self).__init__(
        num_units=num_units,
        num_dims=1,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        tied=False,
        cell_fn=cell_fn,
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


class Grid2BasicLSTMCell(GridRNNCell):
  """2D BasicLSTM cell.

  This creates a 2D cell which receives input and gives output in the first
  dimension.

  The first dimension can optionally be non-recurrent if `non_recurrent_fn` is
  specified.
  """

  def __init__(self,
               num_units,
               tied=False,
               non_recurrent_fn=None,
               forget_bias=1,
               state_is_tuple=True,
               output_is_tuple=True):
    def cell_fn(n):
      return rnn.BasicLSTMCell(num_units=n, forget_bias=forget_bias)
    super(Grid2BasicLSTMCell, self).__init__(
        num_units=num_units,
        num_dims=2,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        tied=tied,
        non_recurrent_dims=None if non_recurrent_fn is None else 0,
        cell_fn=cell_fn,
        non_recurrent_fn=non_recurrent_fn,
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


class Grid1LSTMCell(GridRNNCell):
  """1D LSTM cell.

  This is different from Grid1BasicLSTMCell because it gives options to
  specify the forget bias and enabling peepholes.
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=True,
               output_is_tuple=True):

    def cell_fn(n):
      return rnn.LSTMCell(
          num_units=n, forget_bias=forget_bias, use_peepholes=use_peepholes)

    super(Grid1LSTMCell, self).__init__(
        num_units=num_units,
        num_dims=1,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        cell_fn=cell_fn,
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


class Grid2LSTMCell(GridRNNCell):
  """2D LSTM cell.

    This creates a 2D cell which receives input and gives output in the first
    dimension.
    The first dimension can optionally be non-recurrent if `non_recurrent_fn` is
    specified.
  """

  def __init__(self,
               num_units,
               tied=False,
               non_recurrent_fn=None,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=True,
               output_is_tuple=True):

    def cell_fn(n):
      return rnn.LSTMCell(
          num_units=n, forget_bias=forget_bias, use_peepholes=use_peepholes)

    super(Grid2LSTMCell, self).__init__(
        num_units=num_units,
        num_dims=2,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        tied=tied,
        non_recurrent_dims=None if non_recurrent_fn is None else 0,
        cell_fn=cell_fn,
        non_recurrent_fn=non_recurrent_fn,
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


class Grid3LSTMCell(GridRNNCell):
  """3D BasicLSTM cell.

    This creates a 2D cell which receives input and gives output in the first
    dimension.
    The first dimension can optionally be non-recurrent if `non_recurrent_fn` is
    specified.
    The second and third dimensions are LSTM.
  """

  def __init__(self,
               num_units,
               tied=False,
               non_recurrent_fn=None,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=True,
               output_is_tuple=True):

    def cell_fn(n):
      return rnn.LSTMCell(
          num_units=n, forget_bias=forget_bias, use_peepholes=use_peepholes)

    super(Grid3LSTMCell, self).__init__(
        num_units=num_units,
        num_dims=3,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        tied=tied,
        non_recurrent_dims=None if non_recurrent_fn is None else 0,
        cell_fn=cell_fn,
        non_recurrent_fn=non_recurrent_fn,
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


class Grid2GRUCell(GridRNNCell):
  """2D LSTM cell.

    This creates a 2D cell which receives input and gives output in the first
    dimension.
    The first dimension can optionally be non-recurrent if `non_recurrent_fn` is
    specified.
  """

  def __init__(self,
               num_units,
               tied=False,
               non_recurrent_fn=None,
               state_is_tuple=True,
               output_is_tuple=True):
    super(Grid2GRUCell, self).__init__(
        num_units=num_units,
        num_dims=2,
        input_dims=0,
        output_dims=0,
        priority_dims=0,
        tied=tied,
        non_recurrent_dims=None if non_recurrent_fn is None else 0,
        cell_fn=lambda n: rnn.GRUCell(num_units=n),
        non_recurrent_fn=non_recurrent_fn,
        state_is_tuple=state_is_tuple,
        output_is_tuple=output_is_tuple)


# Helpers

_GridRNNDimension = namedtuple('_GridRNNDimension', [
    'idx', 'is_input', 'is_output', 'is_priority', 'non_recurrent_fn'
])

_GridRNNConfig = namedtuple('_GridRNNConfig',
                            ['num_dims', 'dims', 'inputs', 'outputs',
                             'recurrents', 'priority', 'non_priority', 'tied',
                             'num_units'])


def _parse_rnn_config(num_dims, ls_input_dims, ls_output_dims, ls_priority_dims,
                      ls_non_recurrent_dims, non_recurrent_fn, tied, num_units):
  def check_dim_list(ls):
    if ls is None:
      ls = []
    if not isinstance(ls, (list, tuple)):
      ls = [ls]
    ls = sorted(set(ls))
    if any(_ < 0 or _ >= num_dims for _ in ls):
      raise ValueError('Invalid dims: {}. Must be in [0, {})'.format(ls,
                                                                     num_dims))
    return ls

  input_dims = check_dim_list(ls_input_dims)
  output_dims = check_dim_list(ls_output_dims)
  priority_dims = check_dim_list(ls_priority_dims)
  non_recurrent_dims = check_dim_list(ls_non_recurrent_dims)

  rnn_dims = []
  for i in range(num_dims):
    rnn_dims.append(
        _GridRNNDimension(
            idx=i,
            is_input=(i in input_dims),
            is_output=(i in output_dims),
            is_priority=(i in priority_dims),
            non_recurrent_fn=non_recurrent_fn
            if i in non_recurrent_dims else None))
  return _GridRNNConfig(
      num_dims=num_dims,
      dims=rnn_dims,
      inputs=input_dims,
      outputs=output_dims,
      recurrents=[x for x in range(num_dims) if x not in non_recurrent_dims],
      priority=priority_dims,
      non_priority=[x for x in range(num_dims) if x not in priority_dims],
      tied=tied,
      num_units=num_units)


def _propagate(dim_indices, conf, cells, c_prev, m_prev, new_output, new_state,
               first_call):
  """Propagates through all the cells in dim_indices dimensions.
  """
  if len(dim_indices) == 0:
    return

  # Because of the way RNNCells are implemented, we take the last dimension
  # (H_{N-1}) out and feed it as the state of the RNN cell
  # (in `last_dim_output`).
  # The input of the cell (H_0 to H_{N-2}) are concatenated into `cell_inputs`
  if conf.num_dims > 1:
    ls_cell_inputs = [None] * (conf.num_dims - 1)
    for d in conf.dims[:-1]:
      if new_output[d.idx] is None:
        ls_cell_inputs[d.idx] = m_prev[d.idx]
      else:
        ls_cell_inputs[d.idx] = new_output[d.idx]
    cell_inputs = array_ops.concat(ls_cell_inputs, 1)
  else:
    cell_inputs = array_ops.zeros([m_prev[0].get_shape().as_list()[0], 0],
                                  m_prev[0].dtype)

  last_dim_output = (new_output[-1]
                     if new_output[-1] is not None else m_prev[-1])

  for i in dim_indices:
    d = conf.dims[i]
    if d.non_recurrent_fn:
      if conf.num_dims > 1:
        linear_args = array_ops.concat([cell_inputs, last_dim_output], 1)
      else:
        linear_args = last_dim_output
      with vs.variable_scope('non_recurrent' if conf.tied else
                             'non_recurrent/cell_{}'.format(i)):
        if conf.tied and not (first_call and i == dim_indices[0]):
          vs.get_variable_scope().reuse_variables()

        new_output[d.idx] = layers.fully_connected(
            linear_args,
            num_outputs=conf.num_units,
            activation_fn=d.non_recurrent_fn,
            weights_initializer=(vs.get_variable_scope().initializer or
                                 layers.initializers.xavier_initializer),
            weights_regularizer=vs.get_variable_scope().regularizer)
    else:
      if c_prev[i] is not None:
        cell_state = (c_prev[i], last_dim_output)
      else:
        # for GRU/RNN, the state is just the previous output
        cell_state = last_dim_output

      with vs.variable_scope('recurrent' if conf.tied else
                             'recurrent/cell_{}'.format(i)):
        if conf.tied and not (first_call and i == dim_indices[0]):
          vs.get_variable_scope().reuse_variables()
        cell = cells[i]
        new_output[d.idx], new_state[d.idx] = cell(cell_inputs, cell_state)
