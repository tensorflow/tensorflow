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

import re

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
  with ops.name_scope(name, values=[x]):
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


def einsum(axes, *inputs):
  """
  A generalized contraction between tensors of arbitrary dimension.

  Like `numpy.einsum`, but does not support:
  * Ellipses (subscripts like `ij...,jk...->ik...`)
  * Subscripts where an axis appears more than once for a single input (e.g. `ijj,jk->ik`).

  Args:
    axes: a `str` describing the contraction, in the same format as `numpy.einsum`.
    inputs: the inputs to contract (each one a `Tensor`), whose shapes should be consistent with `axes`.

  Returns:
    The contracted `Tensor`, with shape determined by `axes`.

  Raises:
    ValueError: If the format of `axes` is incorrect,
                or the number of inputs implied by `axes` does not match `len(inputs)`,
                or an axis appears in the output subscripts but not in any of the inputs,
                or the number of dimensions of an input differs from the number of indices in its subscript,
                or the input shapes are inconsistent along a particular axis.
  """
  if '...' in axes:
    raise ValueError("Subscripts with ellipses are not yet supported.")

  match = re.match('([a-z,]+)(->[a-z]*)?', axes)
  if not match:
    raise ValueError(
      "Indices have incorrect format: %s" % axes
    )

  inputs = list(inputs)
  idx_in = match.group(1).split(',')
  idx_all = set(''.join(idx_in))
  indices = ''.join(sorted(idx_all))

  if match.group(2):
    idx_out = match.group(2)[2:]

  else:
    # infer the output subscripts if not given, assume alphabetical order
    counts = {ax: 0 for ax in indices}
    for axes_ in idx_in:
      for ax in axes_:
        counts[ax] += 1

    idx_out = ''.join(sorted(
      ax for ax in indices
      if counts[ax] == 1
    ))

  if len(idx_in) != len(inputs):
    raise ValueError(
      "Expected %d inputs but got %d" % (len(idx_in), len(inputs))
    )

  missing_idx = set(idx_out).difference(idx_all)
  if missing_idx:
    raise ValueError(
      "Unknown ouput axes: %s" % missing_idx
    )

  axis_order = {}
  for ax in indices:
    if ax not in idx_out:
      axis_order[ax] = len(axis_order)
  for ax in idx_out:
    axis_order[ax] = len(axis_order)

  # transpose inputs so axes are in order
  for i, (input_, axes_) in enumerate(zip(inputs, idx_in)):
    if input_.get_shape().ndims != len(axes_):
      raise ValueError(
        "Input %d with axes %s has incorrect" \
        " number of dimensions (expected %d, got %d)" % (
          i, axes_, len(axes_), input_.get_shape().ndims
        )
      )

    sorted_idx = sorted(axes_, key=axis_order.get)

    if len(set(axes_)) != len(axes_):
      raise ValueError(
        "Subscript not supported: an axis appears more than once: %s" % axes_
      )

    if list(axes_) != sorted_idx:
      permuted = [axes_.find(ax) for ax in sorted_idx]
      inputs[i] = array_ops.transpose(input_, permuted)
      idx_in[i] = sorted_idx

  reduction_idx = []
  shapes = [[dim if dim else -1
             for dim in tensor.get_shape().as_list()]
            for tensor in inputs]

  # validate shapes for broadcasting
  for j, ax in enumerate(sorted(idx_all, key=axis_order.get)):
    dims = []
    for i, idx in enumerate(idx_in):
      if ax not in idx:
        shapes[i].insert(j, 1)
      else:
        dim = shapes[i][j]
        if isinstance(dim, int) and dim > 1:
          dims.append(dim)

    if len(set(dims)) > 1:
      raise ValueError(
        "Dimension mismatch on axis: %s" % ax
      )

    if ax not in idx_out:
      reduction_idx.append(j)

  # reshape, multiply
  expanded_inputs = [array_ops.reshape(input_, shape)
                     for input_, shape in zip(inputs, shapes)]
  expanded_output = 1
  for input_ in expanded_inputs:
    expanded_output *= input_

  # contract
  return math_ops.reduce_sum(expanded_output, reduction_idx)
