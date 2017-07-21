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
from tensorflow.python.ops import math_ops

__all__ = [
    "custom_gradient",
]


def custom_gradient(fx, gx, x, axis=(),
                    fx_gx_manually_stopped=False,
                    name=None):
  """Enables specifying a custom gradient.

  This function works by clever application of `stop_gradient`. I.e., observe
  that:

  ```none
  h(x) = x * stop_gradient(g(x)) + stop_gradient(f(x) - x * g(x))
  ```

  is such that `h(x) = stop(f(x))` and `grad[h(x), x] = stop_gradient(g(x)).`

  In addition to scalar-domain/scalar-range functions, this function also
  supports tensor-domain/scalar-range functions. However, in the latter case it
  is necessary to reduce `x` to a scalar. This can be done by indicating the
  `axis` over which `f` operates or by appropriately `reduce_sum`-ing `x`, prior
  to calling this function.

  Partial Custom Gradient:

  Suppose `h(x) = htilde(x, y)`. Note that `dh/dx = stop(g(x))` but `dh/dy =
  None`. This is because a `Tensor` cannot have only a portion of its gradient
  stopped. To circumvent this issue, one must manually `stop_gradient` the
  relevant portions of `f`, `g`. For example see the unit-test,
  `test_works_correctly_fx_gx_manually_stopped`.

  Args:
    fx: `Tensor`. Output of function evaluated at `x`.
    gx: `Tensor`. Gradient of function evaluated at `x`.
    x: `Tensor`. Point of evaluation for `f, g`.
    axis: 1D `int` `Tensor` representing dimensions of `x` which are the domain
      of `f`. If `()` (the default), `f` is assumed scalar-domain/scalar-range.
      If `None` `f` is assumed to render one scalar given all of `x`. Otherwise
      `f` is assumed to output one scalar for each of `axis` dimensions of `x`.
    fx_gx_manually_stopped: Python `bool` indicating that `fx`, `gx` manually
      have `stop_gradient` applied.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    fx: Floating-type `Tensor` equal to `f(x)` but which has gradient
      `stop_gradient(g(x))`.
  """
  with ops.name_scope(name, "custom_gradient", [fx, gx, x]):
    fx = ops.convert_to_tensor(fx, name="fx")
    # We don't want to bother eagerly computing `gx` since we may not even need
    # it.
    with ops.control_dependencies([fx]):
      gx = ops.convert_to_tensor(gx, dtype=fx.dtype, name="gx")
      gx = array_ops.identity(gx, name="gx")
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
    # The proof is similar for the tensor-domain case, except that `x` is
    # replaced by `reduce_sum(x)`.
    sum_x = math_ops.reduce_sum(x, axis=axis, name="sum_x")
    if not fx_gx_manually_stopped:
      fx = array_ops.stop_gradient(fx)
      gx = array_ops.stop_gradient(gx)
    # IEEE754 ensures `(x-x)==0.` and that `0.*x==0.` so we make sure to write
    # the code this way, rather than, e.g.,
    # `sum_x * stop(gx) + stop(fx - sum_x * gx)`.
    # For more discussion regarding the relevant portions of the IEEE754
    # standard, see the StackOverflow question,
    # "Is there a floating point value of x, for which x-x == 0 is false?"
    # http://stackoverflow.com/q/2686644
    return (sum_x - array_ops.stop_gradient(sum_x)) * gx + fx
