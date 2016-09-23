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

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import resource_loader

_lstm_ops_so = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_lstm_ops.so"))
assert _lstm_ops_so, "Could not load _lstm_ops.so."


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

  This implementation uses 1 weight matrix and 1 bias vector, there is no
  diagonal peephole connection.

  This kernel op implements the following mathematical equations:

  ```python
  xh = [x, h_prev]
  [i, f, ci, o] = xh * w + b
  f = f + forget_bias

  i = sigmoid(i)
  f = sigmoid(f)
  ci = tanh(ci)
  o = sigmoid(o)

  cs = ci .* i + cs_prev .* f
  co = tanh(cs)

  h = co .* o
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      The input to the LSTM cell.
    cs_prev: A `Tensor`. Must have the same type as `x`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    wci: A `Tensor`. Must have the same type as `x`.
    wcf: A `Tensor`. Must have the same type as `x`.
    wco: A `Tensor`. Must have the same type as `x`.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `3`.
    use_peephole: An optional `bool`. Defaults to `False`.
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
  return _lstm_ops_so.lstm_block_cell(x=x,
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
    zero_state = array_ops.constant(0,
                                    dtype=dtypes.float32,
                                    shape=[batch_size, cell_size])
  if cs_prev is None:
    cs_prev = zero_state
  if h_prev is None:
    h_prev = zero_state
  if wci is None:
    wci = array_ops.constant(0, dtype=dtypes.float32, shape=[cell_size])
    wco = wci
    wcf = wci

  # pylint: disable=protected-access
  return _lstm_ops_so.block_lstm(seq_len_max=seq_len_max,
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
                                 name=name,
                                 use_peephole=use_peephole)
  # pylint: enable=protected-access
  # pylint: enable=invalid-name


_lstm_block_cell_grad_outputs = ["cs_prev_grad", "dicfo"]


ops.RegisterShape("LSTMBlockCell")(common_shapes.call_cpp_shape_fn)


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
   wco_grad) = _lstm_ops_so.lstm_block_cell_grad(
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
  xh = array_ops.concat(1, [x, h_prev])
  w_grad = math_ops.matmul(xh, dicfo, transpose_a=True)
  w_grad.get_shape().merge_with(w.get_shape())

  # Backprop from dicfo to b.
  b_grad = nn_ops.bias_add_grad(dicfo)
  b_grad.get_shape().merge_with(b.get_shape())

  return (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)


ops.RegisterShape("LSTMBlockCellGrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("BlockLSTM")(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient("BlockLSTM")
def _BlockLSTMGrad(op, *grad):
  """Gradient for BlockLSTM."""
  max_len = op.get_attr("max_len")

  seq_len_max = op.inputs[0]
  x = op.inputs[1:1 + max_len]
  cs_prev = op.inputs[-7]
  h_prev = op.inputs[-6]
  w = op.inputs[-5]
  wci = op.inputs[-4]
  wco = op.inputs[-3]
  wcf = op.inputs[-2]
  b = op.inputs[-1]

  i = op.outputs[0 * max_len:1 * max_len]
  cs = op.outputs[1 * max_len:2 * max_len]
  f = op.outputs[2 * max_len:3 * max_len]
  o = op.outputs[3 * max_len:4 * max_len]
  ci = op.outputs[4 * max_len:5 * max_len]
  co = op.outputs[5 * max_len:6 * max_len]
  h = op.outputs[6 * max_len:7 * max_len]

  cs_grad = grad[-max_len * 2:-max_len]
  h_grad = grad[-max_len:]

  (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wco_grad, wcf_grad,
   b_grad) = _lstm_ops_so.block_lstm_grad(
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

  return [None] + x_grad + [cs_prev_grad, h_prev_grad, w_grad, wci_grad,
                            wco_grad, wcf_grad, b_grad]


ops.RegisterShape("BlockLSTMGrad")(common_shapes.call_cpp_shape_fn)


class LSTMBlockCell(rnn_cell.RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  Unlike BasicLSTMCell, this is a monolithic op and should be much faster. The
  weight and bias matrixes should be compatible as long as the variabel scope
  matches.
  """

  def __init__(self, num_units, forget_bias=1.0, use_peephole=False):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      use_peephole: Whether to use peephole connections or not.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._use_peephole = use_peephole

  @property
  def state_size(self):
    return (self._num_units,) * 2

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, x, states_prev, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):
      x_shape = x.get_shape().with_rank(2)
      if not x_shape[1]:
        raise ValueError("Expecting x_shape[1] to be sets: %s" % str(x_shape))
      if len(states_prev) != 2:
        raise ValueError("Expecting states_prev to be a tuple with length 2.")
      input_size = x_shape[1]
      w = vs.get_variable("W", [input_size + self._num_units,
                                self._num_units * 4])
      b = vs.get_variable("b", [w.get_shape().with_rank(2)[1]],
                          initializer=init_ops.constant_initializer(0.0))
      wci = vs.get_variable("wci", [self._num_units])
      wco = vs.get_variable("wco", [self._num_units])
      wcf = vs.get_variable("wcf", [self._num_units])
      (cs_prev, h_prev) = states_prev
      (_, cs, _, _, _, _, h) = _lstm_block_cell(x,
                                                cs_prev,
                                                h_prev,
                                                w,
                                                b,
                                                wci=wci,
                                                wco=wco,
                                                wcf=wcf,
                                                forget_bias=self._forget_bias,
                                                use_peephole=self._use_peephole)

      return (h, (cs, h))
