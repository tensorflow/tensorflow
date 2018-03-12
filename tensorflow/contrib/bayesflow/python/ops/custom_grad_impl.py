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
"""Functions for specifying custom gradients.

@@custom_gradient

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops

__all__ = [
    'custom_gradient',
]


def is_list_like(x):
  return isinstance(x, (tuple, list))


def identity(x, dtype=None, name=None):
  return array_ops.identity(ops.convert_to_tensor(
      x, dtype=dtype, name=name), name=name)


def custom_gradient(fx, gx, x, fx_gx_manually_stopped=False, name=None):
  """Embeds a custom gradient into a `Tensor`.

  This function works by clever application of `stop_gradient`. I.e., observe
  that:

  ```none
  h(x) = stop_gradient(f(x)) + stop_gradient(g(x)) * (x - stop_gradient(x))
  ```

  is such that `h(x) == stop_gradient(f(x))` and
  `grad[h(x), x] == stop_gradient(g(x)).`

  In addition to scalar-domain/scalar-range functions, this function also
  supports tensor-domain/scalar-range functions.

  Partial Custom Gradient:

  Suppose `h(x) = htilde(x, y)`. Note that `dh/dx = stop(g(x))` but `dh/dy =
  None`. This is because a `Tensor` cannot have only a portion of its gradient
  stopped. To circumvent this issue, one must manually `stop_gradient` the
  relevant portions of `f`, `g`. For example see the unit-test,
  `test_works_correctly_fx_gx_manually_stopped`.

  Args:
    fx: `Tensor`. Output of function evaluated at `x`.
    gx: `Tensor` or list of `Tensor`s. Gradient of function at (each) `x`.
    x: `Tensor` or list of `Tensor`s. Args of evaluation for `f`.
    fx_gx_manually_stopped: Python `bool` indicating that `fx`, `gx` manually
      have `stop_gradient` applied.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    fx: Floating-type `Tensor` equal to `f(x)` but which has gradient
      `stop_gradient(g(x))`.
  """
  def maybe_stop(x):
    if fx_gx_manually_stopped:
      return x
    return array_ops.stop_gradient(x)
  with ops.name_scope(name, 'custom_gradient', [fx, gx, x]):
    fx = ops.convert_to_tensor(fx, name='fx')
    # We don't want to bother eagerly computing `gx` since we may not even need
    # it.
    with ops.control_dependencies([fx]):
      if is_list_like(x):
        x = [identity(x_, name='x') for x_ in x]
      else:
        x = [identity(x, name='x')]

      if is_list_like(gx):
        gx = [identity(gx_, dtype=fx.dtype, name='gx')
              for gx_ in gx]
      else:
        gx = [identity(gx, dtype=fx.dtype, name='gx')]

      override_grad = []
      for x_, gx_ in zip(x, gx):
        # Observe: tf.gradients(f(x), x)[i].shape == x[i].shape
        # thus we check that the user is supplying correct shapes.
        equal_shape = check_ops.assert_equal(
            array_ops.shape(x_),
            array_ops.shape(gx_),
            message='Each `x` must have the same shape as each `gx`.')
        with ops.control_dependencies([equal_shape]):
          # IEEE754 ensures `(x-x)==0.` and that `0.*x==0.` so we make sure to
          # write the code this way, rather than, e.g.,
          # `sum_x * stop(gx) + stop(fx - sum_x * gx)`.
          # For more discussion regarding the relevant portions of the IEEE754
          # standard, see the StackOverflow question,
          # "Is there a floating point value of x, for which x-x == 0 is false?"
          # http://stackoverflow.com/q/2686644
          zeros_like_x_ = x_ - array_ops.stop_gradient(x_)
          override_grad.append(math_ops.reduce_sum(
              maybe_stop(gx_) * zeros_like_x_))
      override_grad = sum(override_grad)
      override_grad /= math_ops.cast(array_ops.size(fx),
                                     dtype=fx.dtype.base_dtype)

      # Proof of correctness:
      #
      #  f(x) = x * stop[gx] + stop[fx - x * gx]
      #       = stop[fx]
      #
      #  g(x) = grad[fx]
      #       = stop[gx] + grad[stop[fx - x * gx]]
      #       = stop[gx] + 0
      #
      # Notice that when x is zero it still works:
      # grad[x * stop(gx) + stop(fx - x * gx)] = 1 * stop[gx] + 0 = stop[gx]
      #
      # The proof is similar for the tensor-domain case, except that we
      # `reduce_sum` the `stop[gx] * (x - stop[x])` then rescale by
      # `tf.size(fx)` since this reduced version is broadcast to `fx`.
      return maybe_stop(fx) + override_grad
