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
"""Arithmetic Operations that don't fit into math_ops due to dependencies.

To avoid circular dependencies, some math_ops should go here.  Documentation
callouts, e.g. "@@my_op" should go in math_ops.  To the user, these are just
normal math_ops.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


# TODO(b/27419586) Change docstring for required dtype of x once int allowed
def lbeta(x, name='lbeta'):
  r"""Computes `ln(|Beta(x)|)`, reducing along the last dimension.

  Given one-dimensional `z = [z_0,...,z_{K-1}]`, we define

  ```Beta(z) = \prod_j Gamma(z_j) / Gamma(\sum_j z_j)```

  And for `n + 1` dimensional `x` with shape `[N1, ..., Nn, K]`, we define
  `lbeta(x)[i1, ..., in] = Log(|Beta(x[i1, ..., in, :])|)`.  In other words,
  the last dimension is treated as the `z` vector.

  Note that if `z = [u, v]`, then
  `Beta(z) = int_0^1 t^{u-1} (1 - t)^{v-1} dt`, which defines the traditional
  bivariate beta function.

  Args:
    x: A rank `n + 1` `Tensor` with type `float`, or `double`.
    name: A name for the operation (optional).

  Returns:
    The logarithm of `|Beta(x)|` reducing along the last dimension.

  Raises:
    ValueError:  If `x` is empty with rank one or less.
  """
  with ops.op_scope([x], name):
    x = ops.convert_to_tensor(x, name='x')
    x = control_flow_ops.with_dependencies(
        [check_ops.assert_rank_at_least(x, 1)], x)

    is_empty = math_ops.equal(0, array_ops.size(x))

    def nonempty_lbeta():
      log_prod_gamma_x = math_ops.reduce_sum(
          math_ops.lgamma(x), reduction_indices=[-1])
      sum_x = math_ops.reduce_sum(x, reduction_indices=[-1])
      log_gamma_sum_x = math_ops.lgamma(sum_x)
      result = log_prod_gamma_x - log_gamma_sum_x
      return result

    def empty_lbeta():
      # If x is empty, return version with one less dimension.
      # Can only do this if rank >= 2.
      assertion = check_ops.assert_rank_at_least(x, 2)
      with ops.control_dependencies([assertion]):
        return array_ops.squeeze(x, squeeze_dims=[0])

    static_size = x.get_shape().num_elements()
    if static_size is not None:
      if static_size > 0:
        return nonempty_lbeta()
      else:
        return empty_lbeta()
    else:
      return control_flow_ops.cond(is_empty, empty_lbeta, nonempty_lbeta)
