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
"""Python wrapper for the Block GRU Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.ops import gen_gru_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import resource_loader
from tensorflow.python.util.deprecation import deprecated_args

_gru_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_gru_ops.so"))

LayerRNNCell = rnn_cell_impl.LayerRNNCell  # pylint: disable=invalid-name


@ops.RegisterGradient("GRUBlockCell")
def _GRUBlockCellGrad(op, *grad):
  r"""Gradient for GRUBlockCell.

  Args:
    op: Op for which the gradient is defined.
    *grad: Gradients of the optimization function wrt output
      for the Op.

  Returns:
    d_x: Gradients wrt to x
    d_h: Gradients wrt to h
    d_w_ru: Gradients wrt to w_ru
    d_w_c: Gradients wrt to w_c
    d_b_ru: Gradients wrt to b_ru
    d_b_c: Gradients wrt to b_c

  Mathematics behind the Gradients below:
  ```
  d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
  d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)

  d_r_bar_u_bar = [d_r_bar d_u_bar]

  [d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T

  [d_x_component_2 d_h_prevr] = d_c_bar * w_c^T

  d_x = d_x_component_1 + d_x_component_2

  d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
  ```
  Below calculation is performed in the python wrapper for the Gradients
  (not in the gradient kernel.)
  ```
  d_w_ru = x_h_prevr^T * d_c_bar

  d_w_c = x_h_prev^T * d_r_bar_u_bar

  d_b_ru = sum of d_r_bar_u_bar along axis = 0

  d_b_c = sum of d_c_bar along axis = 0
  ```
  """
  x, h_prev, w_ru, w_c, b_ru, b_c = op.inputs
  r, u, c, _ = op.outputs
  _, _, _, d_h = grad

  d_x, d_h_prev, d_c_bar, d_r_bar_u_bar = gen_gru_ops.gru_block_cell_grad(
      x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h)

  x_h_prev = array_ops.concat([x, h_prev], 1)
  d_w_ru = math_ops.matmul(x_h_prev, d_r_bar_u_bar, transpose_a=True)
  d_b_ru = nn_ops.bias_add_grad(d_r_bar_u_bar)

  x_h_prevr = array_ops.concat([x, h_prev * r], 1)
  d_w_c = math_ops.matmul(x_h_prevr, d_c_bar, transpose_a=True)
  d_b_c = nn_ops.bias_add_grad(d_c_bar)

  return d_x, d_h_prev, d_w_ru, d_w_c, d_b_ru, d_b_c


class GRUBlockCell(LayerRNNCell):
  r"""Block GRU cell implementation.

  Deprecated: use GRUBlockCellV2 instead.

  The implementation is based on:  http://arxiv.org/abs/1406.1078
  Computes the GRU cell forward propagation for 1 time step.

  This kernel op implements the following mathematical equations:

  Biases are initialized with:

  * `b_ru` - constant_initializer(1.0)
  * `b_c` - constant_initializer(0.0)

  ```
  x_h_prev = [x, h_prev]

  [r_bar u_bar] = x_h_prev * w_ru + b_ru

  r = sigmoid(r_bar)
  u = sigmoid(u_bar)

  h_prevr = h_prev \circ r

  x_h_prevr = [x h_prevr]

  c_bar = x_h_prevr * w_c + b_c
  c = tanh(c_bar)

  h = (1-u) \circ c + u \circ h_prev
  ```

  """

  @deprecated_args(None, "cell_size is deprecated, use num_units instead",
                   "cell_size")
  def __init__(self,
               num_units=None,
               cell_size=None,
               reuse=None,
               name="gru_cell"):
    """Initialize the Block GRU cell.

    Args:
      num_units: int, The number of units in the GRU cell.
      cell_size: int, The old (deprecated) name for `num_units`.
      reuse: (optional) boolean describing whether to reuse variables in an
        existing scope.  If not `True`, and the existing scope already has the
        given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.  By default this is "lstm_cell", for variable-name compatibility
        with `tf.nn.rnn_cell.GRUCell`.

    Raises:
      ValueError: if both cell_size and num_units are not None;
        or both are None.
    """
    super(GRUBlockCell, self).__init__(_reuse=reuse, name=name)
    if (cell_size is None) == (num_units is None):
      raise ValueError(
          "Exactly one of num_units or cell_size must be provided.")
    if num_units is None:
      num_units = cell_size
    self._cell_size = num_units
    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

  @property
  def state_size(self):
    return self._cell_size

  @property
  def output_size(self):
    return self._cell_size

  def build(self, input_shape):
    # Check if the input size exist.
    input_size = tensor_shape.dimension_value(input_shape[1])
    if input_size is None:
      raise ValueError("Expecting input_size to be set.")

    self._gate_kernel = self.add_variable(
        "w_ru", [input_size + self._cell_size, self._cell_size * 2])
    self._gate_bias = self.add_variable(
        "b_ru", [self._cell_size * 2],
        initializer=init_ops.constant_initializer(1.0))
    self._candidate_kernel = self.add_variable(
        "w_c", [input_size + self._cell_size, self._cell_size])
    self._candidate_bias = self.add_variable(
        "b_c", [self._cell_size],
        initializer=init_ops.constant_initializer(0.0))

    self.built = True

  def call(self, inputs, h_prev):
    """GRU cell."""
    # Check cell_size == state_size from h_prev.
    cell_size = h_prev.get_shape().with_rank(2)[1]
    if cell_size != self._cell_size:
      raise ValueError("Shape of h_prev[1] incorrect: cell_size %i vs %s" %
                       (self._cell_size, cell_size))

    _gru_block_cell = gen_gru_ops.gru_block_cell  # pylint: disable=invalid-name
    _, _, _, new_h = _gru_block_cell(
        x=inputs,
        h_prev=h_prev,
        w_ru=self._gate_kernel,
        w_c=self._candidate_kernel,
        b_ru=self._gate_bias,
        b_c=self._candidate_bias)

    return new_h, new_h


class GRUBlockCellV2(GRUBlockCell):
  """Temporary GRUBlockCell impl with a different variable naming scheme.

  Only differs from GRUBlockCell by variable names.
  """

  def build(self, input_shape):
    """GRU cell."""
    input_size = tensor_shape.dimension_value(input_shape[1])
    if input_size is None:
      raise ValueError("Expecting input_size to be set.")

    self._gate_kernel = self.add_variable(
        "gates/kernel", [input_size + self._cell_size, self._cell_size * 2])
    self._gate_bias = self.add_variable(
        "gates/bias", [self._cell_size * 2],
        initializer=init_ops.constant_initializer(1.0))
    self._candidate_kernel = self.add_variable(
        "candidate/kernel", [input_size + self._cell_size, self._cell_size])
    self._candidate_bias = self.add_variable(
        "candidate/bias", [self._cell_size],
        initializer=init_ops.constant_initializer(0.0))

    self.built = True
