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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def _reduce_and_reshape_grad(g, t):
  """Returns the gradient, sum-reduced and reshaped to `t`'s shape."""
  shape = array_ops.shape(t)
  g_shape = array_ops.shape(g)
  bcast_dims, _ = gen_array_ops.broadcast_gradient_args(shape, g_shape)
  return array_ops.reshape(math_ops.reduce_sum(g, bcast_dims), shape)


def scaled_softplus(x, alpha, clip=None, name=None):
  """Returns `y = alpha * ln(1 + exp(x / alpha))` or `min(y, clip)`.

  This can be seen as a softplus applied to the scaled input, with the output
  appropriately scaled. As `alpha` tends to 0, `scaled_softplus(x, alpha)` tends
  to `relu(x)`. The clipping is optional. As alpha->0, scaled_softplus(x, alpha)
  tends to relu(x), and scaled_softplus(x, alpha, clip=6) tends to relu6(x).

  Note: the gradient for this operation is defined to depend on the backprop
  inputs as well as the outputs of this operation.

  Args:
    x: A `Tensor` of inputs.
    alpha: A `Tensor`, indicating the amount of smoothness. The caller
        must ensure that `alpha > 0`.
    clip: (optional) A `Tensor`, the upper bound to clip the values.
    name: A name for the scope of the operations (optional).

  Returns:
    A tensor of the size and type determined by broadcasting of the inputs.

  """
  clipping = clip is not None
  with ops.name_scope(name, 'scaled_softplus',
                      [x, alpha] + ([clip] if clipping else [])):
    x = ops.convert_to_tensor(x, name='x')
    dtype = x.dtype
    alpha = ops.convert_to_tensor(alpha, dtype=dtype, name='alpha')
    # Compute the forward value.
    y = alpha * nn.softplus(x / alpha)
    if clipping:
      clip = ops.convert_to_tensor(clip, dtype=dtype, name='clip')
      y = math_ops.minimum(y, clip)

    def _grad(op, g):
      """Backprop for scaled softplus, with optional clipping."""
      y, x, alpha = op.inputs[:3]
      # Prevent the memory-expensive computations from happening before g is
      # available.
      with ops.control_dependencies([g]):
        y = array_ops.identity(y)
      clip_grad = []
      if clipping:
        clip = op.inputs[3]
        unclipped = math_ops.cast(y < clip, g.dtype)
        clip_grad = [_reduce_and_reshape_grad(g * (1. - unclipped), clip)]
        g *= unclipped
      y /= alpha
      emy = math_ops.exp(-y)
      dy_dx = 1. - emy
      # The eps below avoids log(0). Note that t*log(t) -> 0 as t->0.
      eps = 1e-8
      dy_dalpha = y * emy - dy_dx * math_ops.log(dy_dx + eps)
      # Backprop to the actual inputs, but not to the output.
      return [None,
              _reduce_and_reshape_grad(g * dy_dx, x),
              _reduce_and_reshape_grad(g * dy_dalpha, alpha)] + clip_grad

    if clipping:
      @function.Defun(dtype, dtype, dtype, dtype,
                      func_name='ScaledSoftplusHelper_clip_%s' % dtype.name,
                      shape_func=lambda op: [op.inputs[0].shape],
                      python_grad_func=_grad)
      def _forward_helper_clip(y, x, alpha, clip):
        del x, alpha, clip  # Unused.
        return y
      return _forward_helper_clip(y, x, alpha, clip)
    # No clipping.
    @function.Defun(dtype, dtype, dtype,
                    func_name='ScaledSoftplusHelper_%s' % dtype.name,
                    shape_func=lambda op: [op.inputs[0].shape],
                    python_grad_func=_grad)
    def _forward_helper(y, x, alpha):
      del x, alpha  # Unused.
      return y
    return _forward_helper(y, x, alpha)

