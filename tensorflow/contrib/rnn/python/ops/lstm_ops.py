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
"""LSTM Block Cell ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.contrib.rnn.ops import gen_lstm_ops
from tensorflow.contrib.rnn.python.ops import fused_rnn_cell
from tensorflow.contrib.util import loader
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import resource_loader

_lstm_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_lstm_ops.so"))


# pylint: disable=invalid-name
def _lstm_block_cell(x,
                     cs_prev,
                     h_prev,
                     w,
                     b,
                     wci=None,
                     wcf=None,
                     wco=None,
                     forget_bias=None,
                     cell_clip=None,
                     use_peephole=None,
                     name=None):
  r"""Computes the LSTM cell forward propagation for 1 time step.

  This implementation uses 1 weight matrix and 1 bias vector, and there's an
  optional peephole connection.

  This kernel op implements the following mathematical equations:

  ```python
  xh = [x, h_prev]
  [i, f, ci, o] = xh * w + b
  f = f + forget_bias

  if not use_peephole:
    wci = wcf = wco = 0

  i = sigmoid(cs_prev * wci + i)
  f = sigmoid(cs_prev * wcf + f)
  ci = tanh(ci)

  cs = ci .* i + cs_prev .* f
  cs = clip(cs, cell_clip)

  o = sigmoid(cs * wco + o)
  co = tanh(cs)
  h = co .* o
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the cell state at previous time step.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Output of the previous cell at previous time step.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `3`.
      Value to clip the 'cs' value to.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).
    i: A `Tensor`. Has the same type as `x`. The input gate.
    cs: A `Tensor`. Has the same type as `x`. The cell state before the tanh.
    f: A `Tensor`. Has the same type as `x`. The forget gate.
    o: A `Tensor`. Has the same type as `x`. The output gate.
    ci: A `Tensor`. Has the same type as `x`. The cell input.
    co: A `Tensor`. Has the same type as `x`. The cell after the tanh.
    h: A `Tensor`. Has the same type as `x`. The output h vector.

  Raises:
    ValueError: If cell_size is None.
  """
  if wci is None:
    cell_size = cs_prev.get_shape().with_rank(2)[1].value
    if cell_size is None:
      raise ValueError("cell_size from `cs_prev` should not be None.")
    wci = array_ops.constant(0, dtype=dtypes.float32, shape=[cell_size])
    wco = wci
    wcf = wci

  # pylint: disable=protected-access
  return gen_lstm_ops.lstm_block_cell(
      x=x,
      cs_prev=cs_prev,
      h_prev=h_prev,
      w=w,
      wci=wci,
      wco=wco,
      wcf=wcf,
      b=b,
      forget_bias=forget_bias,
      cell_clip=cell_clip,
      use_peephole=use_peephole,
      name=name)
  # pylint: enable=protected-access


def _block_lstm(seq_len_max,
                x,
                w,
                b,
                cs_prev=None,
                h_prev=None,
                wci=None,
                wcf=None,
                wco=None,
                forget_bias=None,
                cell_clip=None,
                use_peephole=None,
                name=None):
  r"""TODO(williamchan): add doc.

  Args:
    seq_len_max: A `Tensor` of type `int64`.
    x: A list of at least 1 `Tensor` objects of the same type in: `float32`.
    w: A `Tensor`. Must have the same type as `x`.
    b: A `Tensor`. Must have the same type as `x`.
    cs_prev: A `Tensor`. Must have the same type as `x`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    wci: A `Tensor`. Must have the same type as `x`.
    wcf: A `Tensor`. Must have the same type as `x`.
    wco: A `Tensor`. Must have the same type as `x`.
    forget_bias: An optional `float`. Defaults to `1`.
    cell_clip: An optional `float`. Defaults to `3`.
    use_peephole: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).
    i: A list with the same number of `Tensor` objects as `x` of `Tensor`
    objects of the same type as x.
    cs: A list with the same number of `Tensor` objects as `x` of `Tensor`
    objects of the same type as x.
    f: A list with the same number of `Tensor` objects as `x` of `Tensor`
    objects of the same type as x.
    o: A list with the same number of `Tensor` objects as `x` of `Tensor`
    objects of the same type as x.
    ci: A list with the same number of `Tensor` objects as `x` of `Tensor`
    objects of the same type as x.
    co: A list with the same number of `Tensor` objects as `x` of `Tensor`
    objects of the same type as x.
    h: A list with the same number of `Tensor` objects as `x` of `Tensor`
    objects of the same type as x.

  Raises:
    ValueError: If `b` does not have a valid shape.
  """
  batch_size = x[0].get_shape().with_rank(2)[0].value
  cell_size4 = b.get_shape().with_rank(1)[0].value
  if cell_size4 is None:
    raise ValueError("`b` shape must not be None.")
  cell_size = cell_size4 / 4
  zero_state = None
  if cs_prev is None or h_prev is None:
    zero_state = array_ops.constant(
        0, dtype=dtypes.float32, shape=[batch_size, cell_size])
  if cs_prev is None:
    cs_prev = zero_state
  if h_prev is None:
    h_prev = zero_state
  if wci is None:
    wci = array_ops.constant(0, dtype=dtypes.float32, shape=[cell_size])
    wco = wci
    wcf = wci

  # pylint: disable=protected-access
  i, cs, f, o, ci, co, h = gen_lstm_ops.block_lstm(
      seq_len_max=seq_len_max,
      x=array_ops.stack(x),
      cs_prev=cs_prev,
      h_prev=h_prev,
      w=w,
      wci=wci,
      wco=wco,
      wcf=wcf,
      b=b,
      forget_bias=forget_bias,
      cell_clip=cell_clip,
      name=name,
      use_peephole=use_peephole)

  return array_ops.unstack(i), array_ops.unstack(cs), array_ops.unstack(
      f), array_ops.unstack(o), array_ops.unstack(ci), array_ops.unstack(
          co), array_ops.unstack(h)
  # pylint: enable=protected-access
  # pylint: enable=invalid-name


_lstm_block_cell_grad_outputs = ["cs_prev_grad", "dicfo"]


@ops.RegisterGradient("LSTMBlockCell")
def _LSTMBlockCellGrad(op, *grad):
  """Gradient for LSTMBlockCell."""
  (x, cs_prev, h_prev, w, wci, wco, wcf, b) = op.inputs
  (i, cs, f, o, ci, co, _) = op.outputs
  (_, cs_grad, _, _, _, _, h_grad) = grad

  batch_size = x.get_shape().with_rank(2)[0].value
  if batch_size is None:
    batch_size = -1
  input_size = x.get_shape().with_rank(2)[1].value
  if input_size is None:
    raise ValueError("input_size from `x` should not be None.")
  cell_size = cs_prev.get_shape().with_rank(2)[1].value
  if cell_size is None:
    raise ValueError("cell_size from `cs_prev` should not be None.")

  (cs_prev_grad, dicfo, wci_grad, wcf_grad,
   wco_grad) = gen_lstm_ops.lstm_block_cell_grad(
       x,
       cs_prev,
       h_prev,
       w,
       wci,
       wcf,
       wco,
       b,
       i,
       cs,
       f,
       o,
       ci,
       co,
       cs_grad,
       h_grad,
       use_peephole=op.get_attr("use_peephole"))

  # Backprop from dicfo to xh.
  xh_grad = math_ops.matmul(dicfo, w, transpose_b=True)

  x_grad = array_ops.slice(xh_grad, (0, 0), (batch_size, input_size))
  x_grad.get_shape().merge_with(x.get_shape())

  h_prev_grad = array_ops.slice(xh_grad, (0, input_size),
                                (batch_size, cell_size))
  h_prev_grad.get_shape().merge_with(h_prev.get_shape())

  # Backprop from dicfo to w.
  xh = array_ops.concat([x, h_prev], 1)
  w_grad = math_ops.matmul(xh, dicfo, transpose_a=True)
  w_grad.get_shape().merge_with(w.get_shape())

  # Backprop from dicfo to b.
  b_grad = nn_ops.bias_add_grad(dicfo)
  b_grad.get_shape().merge_with(b.get_shape())

  return (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)


@ops.RegisterGradient("BlockLSTM")
def _BlockLSTMGrad(op, *grad):
  """Gradient for BlockLSTM."""
  seq_len_max, x, cs_prev, h_prev, w, wci, wco, wcf, b = op.inputs
  i, cs, f, o, ci, co, h = op.outputs

  cs_grad = grad[1]
  h_grad = grad[6]

  (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wco_grad, wcf_grad,
   b_grad) = gen_lstm_ops.block_lstm_grad(
       seq_len_max,
       x,
       cs_prev,
       h_prev,
       w,
       wci,
       wco,
       wcf,
       b,
       i,
       cs,
       f,
       o,
       ci,
       co,
       h,
       cs_grad,
       h_grad,
       use_peephole=op.get_attr("use_peephole"))

  return [None, x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wco_grad,
          wcf_grad, b_grad]


class LSTMBlockCell(rnn_cell_impl.RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add `forget_bias` (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  Unlike `rnn_cell_impl.LSTMCell`, this is a monolithic op and should be much
  faster.  The weight and bias matrices should be compatible as long as the
  variable scope matches.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               use_peephole=False):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      use_peephole: Whether to use peephole connections or not.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._use_peephole = use_peephole
    self._names = {
        "W": "kernel",
        "b": "bias",
        "wci": "w_i_diag",
        "wco": "w_o_diag",
        "wcf": "w_f_diag",
        "scope": "lstm_cell"
    }

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, x, states_prev, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or self._names["scope"]):
      x_shape = x.get_shape().with_rank(2)
      if not x_shape[1].value:
        raise ValueError("Expecting x_shape[1] to be set: %s" % str(x_shape))
      if len(states_prev) != 2:
        raise ValueError("Expecting states_prev to be a tuple with length 2.")
      input_size = x_shape[1].value
      w = vs.get_variable(self._names["W"], [input_size + self._num_units,
                                             self._num_units * 4])
      b = vs.get_variable(
          self._names["b"], [w.get_shape().with_rank(2)[1].value],
          initializer=init_ops.constant_initializer(0.0))
      if self._use_peephole:
        wci = vs.get_variable(self._names["wci"], [self._num_units])
        wco = vs.get_variable(self._names["wco"], [self._num_units])
        wcf = vs.get_variable(self._names["wcf"], [self._num_units])
      else:
        wci = wco = wcf = array_ops.zeros([self._num_units])
      (cs_prev, h_prev) = states_prev
      (_, cs, _, _, _, _, h) = _lstm_block_cell(
          x,
          cs_prev,
          h_prev,
          w,
          b,
          wci=wci,
          wco=wco,
          wcf=wcf,
          forget_bias=self._forget_bias,
          use_peephole=self._use_peephole)

      new_state = rnn_cell_impl.LSTMStateTuple(cs, h)
      return h, new_state


class LSTMBlockWrapper(fused_rnn_cell.FusedRNNCell):
  """This is a helper class that provides housekeeping for LSTM cells.

  This may be useful for alternative LSTM and similar type of cells.
  The subclasses must implement `_call_cell` method and `num_units` property.
  """

  @abc.abstractproperty
  def num_units(self):
    """Number of units in this cell (output dimension)."""
    pass

  @abc.abstractmethod
  def _call_cell(self, inputs, initial_cell_state, initial_output, dtype,
                 sequence_length):
    """Run this LSTM on inputs, starting from the given state.

    This method must be implemented by subclasses and does the actual work
    of calling the cell.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
      initial_cell_state: initial value for cell state, shape `[batch_size,
        self._num_units]`
      initial_output: initial value of cell output, shape `[batch_size,
        self._num_units]`
      dtype: The data type for the initial state and expected output.
      sequence_length: Specifies the length of each sequence in inputs. An int32
        or int64 vector (tensor) size [batch_size], values in [0, time_len) or
          None.

    Returns:
      A pair containing:

      - State: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
      - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
    """
    pass

  def __call__(self,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
    """Run this LSTM on inputs, starting from the given state.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
        or a list of `time_len` tensors of shape `[batch_size, input_size]`.
      initial_state: a tuple `(initial_cell_state, initial_output)` with tensors
        of shape `[batch_size, self._num_units]`. If this is not provided, the
        cell is expected to create a zero initial state of type `dtype`.
      dtype: The data type for the initial state and expected output. Required
        if `initial_state` is not provided or RNN state has a heterogeneous
        dtype.
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len).`
        Defaults to `time_len` for each element.
      scope: `VariableScope` for the created subgraph; defaults to class name.

    Returns:
      A pair containing:

      - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
        or a list of time_len tensors of shape `[batch_size, output_size]`,
        to match the type of the `inputs`.
      - Final state: a tuple `(cell_state, output)` matching `initial_state`.

    Raises:
      ValueError: in case of shape mismatches
    """
    with vs.variable_scope(scope or "lstm_block_wrapper"):
      is_list = isinstance(inputs, list)
      if is_list:
        inputs = array_ops.stack(inputs)
      inputs_shape = inputs.get_shape().with_rank(3)
      if not inputs_shape[2]:
        raise ValueError("Expecting inputs_shape[2] to be set: %s" %
                         inputs_shape)
      batch_size = inputs_shape[1].value
      if batch_size is None:
        batch_size = array_ops.shape(inputs)[1]
      time_len = inputs_shape[0].value
      if time_len is None:
        time_len = array_ops.shape(inputs)[0]

      # Provide default values for initial_state and dtype
      if initial_state is None:
        if dtype is None:
          raise ValueError(
              "Either initial_state or dtype needs to be specified")
        z = array_ops.zeros(
            array_ops.stack([batch_size, self.num_units]), dtype=dtype)
        initial_state = z, z
      else:
        if len(initial_state) != 2:
          raise ValueError(
              "Expecting initial_state to be a tuple with length 2 or None")
        if dtype is None:
          dtype = initial_state[0].dtype

      # create the actual cell
      if sequence_length is not None:
        sequence_length = ops.convert_to_tensor(sequence_length)
      initial_cell_state, initial_output = initial_state  # pylint: disable=unpacking-non-sequence
      cell_states, outputs = self._call_cell(inputs, initial_cell_state,
                                             initial_output, dtype,
                                             sequence_length)

      if sequence_length is not None:
        # Mask out the part beyond sequence_length
        mask = array_ops.transpose(
            array_ops.sequence_mask(
                sequence_length, time_len, dtype=dtype), [1, 0])
        mask = array_ops.tile(
            array_ops.expand_dims(mask, [-1]), [1, 1, self.num_units])
        outputs *= mask
        # Prepend initial states to cell_states and outputs for indexing to work
        # correctly,since we want to access the last valid state at
        # sequence_length - 1, which can even be -1, corresponding to the
        # initial state.
        mod_cell_states = array_ops.concat(
            [array_ops.expand_dims(initial_cell_state, [0]), cell_states], 0)
        mod_outputs = array_ops.concat(
            [array_ops.expand_dims(initial_output, [0]), outputs], 0)
        final_cell_state = self._gather_states(mod_cell_states, sequence_length,
                                               batch_size)
        final_output = self._gather_states(mod_outputs, sequence_length,
                                           batch_size)
      else:
        # No sequence_lengths used: final state is the last state
        final_cell_state = cell_states[-1]
        final_output = outputs[-1]

      if is_list:
        # Input was a list, so return a list
        outputs = array_ops.unstack(outputs)

      final_state = rnn_cell_impl.LSTMStateTuple(final_cell_state, final_output)
      return outputs, final_state

  def _gather_states(self, data, indices, batch_size):
    """Produce `out`, s.t. out(i, j) = data(indices(i), i, j)."""
    mod_indices = indices * batch_size + math_ops.range(batch_size)
    return array_ops.gather(
        array_ops.reshape(data, [-1, self.num_units]), mod_indices)


class LSTMBlockFusedCell(LSTMBlockWrapper):
  """FusedRNNCell implementation of LSTM.

  This is an extremely efficient LSTM implementation, that uses a single TF op
  for the entire LSTM. It should be both faster and more memory-efficient than
  LSTMBlockCell defined above.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  The variable naming is consistent with `rnn_cell_impl.LSTMCell`.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               cell_clip=None,
               use_peephole=False):
    """Initialize the LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      cell_clip: clip the cell to this value. Defaults to `3`.
      use_peephole: Whether to use peephole connections or not.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._cell_clip = cell_clip
    self._use_peephole = use_peephole

  @property
  def num_units(self):
    """Number of units in this cell (output dimension)."""
    return self._num_units

  def _call_cell(self, inputs, initial_cell_state, initial_output, dtype,
                 sequence_length):
    """Run this LSTM on inputs, starting from the given state.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
      initial_cell_state: initial value for cell state, shape `[batch_size,
        self._num_units]`
      initial_output: initial value of cell output, shape `[batch_size,
        self._num_units]`
      dtype: The data type for the initial state and expected output.
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len)` or None.

    Returns:
      A pair containing:

      - Cell state (cs): A `3-D` tensor of shape `[time_len, batch_size,
                         output_size]`
      - Output (h): A `3-D` tensor of shape `[time_len, batch_size,
                    output_size]`
    """

    inputs_shape = inputs.get_shape().with_rank(3)
    time_len = inputs_shape[0].value
    if time_len is None:
      time_len = array_ops.shape(inputs)[0]
    input_size = inputs_shape[2].value
    w = vs.get_variable(
        "kernel",
        [input_size + self._num_units, self._num_units * 4], dtype=dtype)
    b = vs.get_variable(
        "bias", [w.get_shape().with_rank(2)[1]],
        initializer=init_ops.constant_initializer(0.0),
        dtype=dtype)
    if self._use_peephole:
      wci = vs.get_variable("w_i_diag", [self._num_units], dtype=dtype)
      wco = vs.get_variable("w_o_diag", [self._num_units], dtype=dtype)
      wcf = vs.get_variable("w_f_diag", [self._num_units], dtype=dtype)
    else:
      wci = wco = wcf = array_ops.zeros([self._num_units], dtype=dtype)

    if sequence_length is None:
      max_seq_len = math_ops.to_int64(time_len)
    else:
      max_seq_len = math_ops.to_int64(math_ops.reduce_max(sequence_length))

    _, cs, _, _, _, _, h = gen_lstm_ops.block_lstm(
        seq_len_max=max_seq_len,
        x=inputs,
        cs_prev=initial_cell_state,
        h_prev=initial_output,
        w=w,
        wci=wci,
        wco=wco,
        wcf=wcf,
        b=b,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole)
    return cs, h
