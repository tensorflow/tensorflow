from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import RNNCell


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

  def __init__(self, num_units, input_size=None, cell_class=None, dims=1, depth=1, depth_op=None, tied=False,
               cell_kwargs=None):
    """Initialize the parameters for a Grid RNN cell

    Args:
      num_units: int, The number of units in the cells in all dimensions of this GridRNN cell
      input_size: int, The dimensionality of the inputs into the cells
      cell_class: class, the cell class used in this Grid cell, should be a subclass of :class:RNNCell
      dims: int, number of dimensions (excluding the depth dimension).
      depth: int, the depth of this GridLSTM (i.e. the length of the depth dimension.
      depth_op: Op, an optional `Op` that will be applied on the depth dimension.
      tied: bool, whether to share the weights among all the dimensions.
      cell_kwargs: dict, additional arguments passed to the constructor of cell.
    """
    if depth < 1:
      raise ValueError('depth must be >= 1: {}'.format(depth))
    if dims < 0:
      raise ValueError('dims must be >= 0: {}'.format(dims))

    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._depth = depth
    self._depth_op = depth_op
    self._dims = dims
    self._tied = tied

    cell_kwargs = cell_kwargs or {}
    cell_kwargs.update({'num_units': num_units, 'input_size': input_size})
    self._cells = [cell_class(**cell_kwargs) for _ in xrange(self.recurrent_dims)]

    self._output_size = self._cells[0].output_size
    self._state_size = self._cells[0].state_size * self.recurrent_dims

  @property
  def recurrent_dims(self):
    return self._dims + (1 if self._depth_op is None else 0)

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

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
    start_idx = 0
    for cell in self._cells:
      n = cell.state_size - cell.output_size
      c_prev.append(array_ops.slice(state, [0, start_idx], [-1, n]))
      m_prev.append(array_ops.slice(state, [0, start_idx + n], [-1, cell.output_size]))
      start_idx += cell.state_size

    # concatenate m_prev
    dtype = inputs.dtype

    grid_scope = scope or type(self).__name__
    with vs.variable_scope(grid_scope):
      cell_scope = grid_scope if self._tied else None
      for cell in self._cells:
        cell()

    return m, array_ops.concat(1, [c, m])