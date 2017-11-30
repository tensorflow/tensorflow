# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Support for scaled softplus, a smoothed version of ReLU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def scaled_softplus(x, alpha, name=None):
  """Returns `alpha * ln(1 + exp(x / alpha))`, for scalar `alpha > 0`.

  This can be seen as a softplus applied to the scaled input, with the output
  appropriately scaled. As `alpha` tends to 0, `scaled_softplus(x, alpha)` tends
  to `relu(x)`.

  Note: the gradient for this operation is defined to depend on the backprop
  inputs as well as the outputs of this operation.

  Args:
    x: A `Tensor` of inputs.
    alpha: A scalar `Tensor`, indicating the amount of smoothness. The caller
        must ensure that `alpha > 0`.
    name: A name for the scope of the operations (optional).

  Returns:
    A tensor of same size and type as `x`.

  """
  with ops.name_scope(name, 'scaled_softplus', [x, alpha]):
    x = ops.convert_to_tensor(x, name='x')
    dtype = x.dtype
    alpha = ops.convert_to_tensor(alpha, dtype=dtype, name='alpha')
    # Verify that alpha is a scalar.
    alpha.get_shape().assert_has_rank(0)

    def _grad(op, g):
      """Backprop for scaled softplus."""
      y = op.outputs[0]
      alpha = op.inputs[1]
      # Prevent the expensive computations from happening before g is available.
      with ops.control_dependencies([g]):
        y /= alpha
      emy = math_ops.exp(-y)
      dy_dx = 1. - emy
      # The eps below avoids log(0). Note that t*log(t) -> 0 as t->0.
      eps = 1e-8
      dy_dalpha = y * emy - dy_dx * math_ops.log(dy_dx + eps)
      return g * dy_dx, math_ops.reduce_sum(g * dy_dalpha)

    @function.Defun(dtype, dtype,
                    func_name='ScaledSoftplus_%s' % dtype.name,
                    shape_func=lambda op: [op.inputs[0].get_shape()],
                    python_grad_func=_grad)
    def _forward(x, alpha):
      """Forward computation of scaled softplus."""
      return alpha * nn.softplus(x / alpha)

    return _forward(x, alpha)

