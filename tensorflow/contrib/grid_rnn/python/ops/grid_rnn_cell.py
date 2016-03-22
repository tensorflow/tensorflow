from collections import namedtuple

from tensorflow.python.ops import array_ops, variable_scope as vs
from tensorflow.python.ops.rnn_cell import RNNCell, BasicRNNCell, BasicLSTMCell, LSTMCell, linear
from tensorflow.python.ops.nn import relu


class GridRNNCell(RNNCell):
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
    if input_size is not None and input_size != num_units:
      raise ValueError('GridRNN requires input_size ({}) to be '
                       'the same with num_units ({})'.format(input_size, num_units))

    self._config = _parse_rnn_config(num_dims, input_dims, output_dims, priority_dims,
                                     non_recurrent_dims, non_recurrent_fn or relu, tied)
    self._input_size = input_size or num_units

    cell_input_size = (self._config.num_dims - 1) * self._input_size
    if cell_fn is None:
      self._cell = LSTMCell(num_units=num_units, input_size=cell_input_size)
    else:
      self._cell = cell_fn(num_units=num_units, input_size=cell_input_size)
      if not isinstance(self._cell, RNNCell):
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
    # c_prev is `m`, and m_prev is `h` in the paper. Keep c and m here for consistency with the codebase
    c_prev = []
    m_prev = []
    cell_units = self._cell.state_size - self._cell.output_size
    for start_idx in range(0, self.state_size, self._cell.state_size):
      c_prev.append(array_ops.slice(state, [0, start_idx], [-1, cell_units]))
      m_prev.append(array_ops.slice(state, [0, start_idx + cell_units], [-1, self._cell.output_size]))

    actual_input_size = inputs.get_shape().as_list()[1]
    if self.input_size != actual_input_size:
      raise ValueError('Actual input size not same as specified: {} vs {}.'.format(actual_input_size, self.input_size))

    conf = self._config
    # input dimensions
    all_inputs = [None] * conf.num_dims
    input_splits = array_ops.split(1, len(conf.inputs), inputs)
    for d, split in zip(conf.inputs, input_splits):
      all_inputs[d] = split

    new_output = [None] * conf.num_dims
    new_state = []

    with vs.variable_scope(scope or type(self).__name__):   # GridRNNCell
      _propagate(conf.non_priority, conf, self._cell, all_inputs, c_prev, m_prev, new_output, new_state)
      _propagate(conf.priority, conf, self._cell, all_inputs, c_prev, m_prev, new_output, new_state)

      output = array_ops.concat(1, new_output)

    return output, array_ops.concat(1, new_state)

class Grid1BasicRNNCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid1BasicRNNCell, self).__init__(num_units=num_units, num_dims=1, input_dims=0, output_dims=0,
                                            priority_dims=0, tied=tied,
                                            cell_fn=lambda n, i: BasicRNNCell(num_units=n, input_size=i))

class Grid2BasicRNNCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid2BasicRNNCell, self).__init__(num_units=num_units, num_dims=2, input_dims=0, output_dims=0,
                                            priority_dims=0, tied=tied,
                                            cell_fn=lambda n, i: BasicRNNCell(num_units=n, input_size=i))

class Grid1BasicLSTMCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid1BasicLSTMCell, self).__init__(num_units=num_units, num_dims=1, input_dims=0, output_dims=0,
                                             priority_dims=0, tied=tied,
                                             cell_fn=lambda n, i: BasicLSTMCell(num_units=n, input_size=i))

class Grid2BasicLSTMCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid2BasicLSTMCell, self).__init__(num_units=num_units, num_dims=2, input_dims=0, output_dims=0,
                                             priority_dims=0, tied=tied,
                                             cell_fn=lambda n, i: BasicLSTMCell(num_units=n, input_size=i))

class Grid1LSTMCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid1LSTMCell, self).__init__(num_units=num_units, num_dims=1, input_dims=0, output_dims=0, priority_dims=0,
                                        tied=tied, cell_fn=lambda n, i: LSTMCell(num_units=n, input_size=i))

class Grid2LSTMCell(GridRNNCell):
  def __init__(self, num_units, tied=False):
    super(Grid2LSTMCell, self).__init__(num_units=num_units, num_dims=2, input_dims=0, output_dims=0, priority_dims=0,
                                        tied=tied, cell_fn=lambda n, i: LSTMCell(num_units=n, input_size=i))


"""
Helpers
"""

_GridRNNDimension = namedtuple('_GridRNNDimension', ['idx', 'is_input', 'is_output', 'is_priority', 'non_recurrent_fn'])

_GridRNNConfig = namedtuple('_GridRNNConfig', ['num_dims', 'dims',
                                               'inputs', 'outputs', 'recurrents',
                                               'priority', 'non_priority', 'tied'])

def _parse_rnn_config(num_dims, ls_input_dims, ls_output_dims, ls_priority_dims, ls_non_recurrent_dims,
                      non_recurrent_fn, tied):

  def check_dim_list(ls):
    if ls is None:
      ls = []
    if not isinstance(ls, (list, tuple)):
      ls = [ls]
    ls = list(set(ls))
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
                        tied=tied)

def _propagate(dim_indices, conf, cell, all_inputs, c_prev, m_prev, new_output, new_state):
  """
  """
  if len(dim_indices) == 0:
    return

  ls_cell_inputs = [None] * (conf.num_dims - 1)
  for d in conf.dims[:-1]:
    ls_cell_inputs[d.idx] = all_inputs[d.idx] or new_output[d.idx] or m_prev[d.idx]
  cell_inputs = array_ops.concat(1, ls_cell_inputs)
  last_dim_output = all_inputs[-1] or new_output[-1] or m_prev[-1]

  for i in dim_indices:
    d = conf.dims[i]
    if d.non_recurrent_fn:
      new_output[d.idx] = d.non_recurrent_fn(linear(args=[cell_inputs, last_dim_output],
                                                    output_size=cell.num_units,
                                                    bias=True,
                                                    scope='non_recurrent' if conf.tied else
                                                    'non_recurrent/cell:{}'.format(i)))
    else:
      cell_state = array_ops.concat(1, [c_prev[i], last_dim_output])
      cell_new_output, cell_new_state = cell(cell_inputs, cell_state,
                                             scope='recurrent' if conf.tied else 'recurrent/cell:{}'.format(i))
      new_output[d.idx] = cell_new_output
      new_state.append(cell_new_state)