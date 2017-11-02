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

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging


# TODO(b/27419586) Change docstring for required dtype of x once int allowed
def lbeta(x, name='lbeta'):
  r"""Computes \\(ln(|Beta(x)|)\\), reducing along the last dimension.

  Given one-dimensional `z = [z_0,...,z_{K-1}]`, we define

  $$Beta(z) = \prod_j Gamma(z_j) / Gamma(\sum_j z_j)$$

  And for `n + 1` dimensional `x` with shape `[N1, ..., Nn, K]`, we define
  $$lbeta(x)[i1, ..., in] = Log(|Beta(x[i1, ..., in, :])|)$$.

  In other words, the last dimension is treated as the `z` vector.

  Note that if `z = [u, v]`, then
  \\(Beta(z) = int_0^1 t^{u-1} (1 - t)^{v-1} dt\\), which defines the
  traditional bivariate beta function.

  If the last dimension is empty, we follow the convention that the sum over
  the empty set is zero, and the product is one.

  Args:
    x: A rank `n + 1` `Tensor`, `n >= 0` with type `float`, or `double`.
    name: A name for the operation (optional).

  Returns:
    The logarithm of \\(|Beta(x)|\\) reducing along the last dimension.
  """
  # In the event that the last dimension has zero entries, we return -inf.
  # This is consistent with a convention that the sum over the empty set 0, and
  # the product is 1.
  # This is standard.  See https://en.wikipedia.org/wiki/Empty_set.
  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name='x')

    # Note reduce_sum([]) = 0.
    log_prod_gamma_x = math_ops.reduce_sum(
        math_ops.lgamma(x), reduction_indices=[-1])

    # Note lgamma(0) = infinity, so if x = []
    # log_gamma_sum_x = lgamma(0) = infinity, and
    # log_prod_gamma_x = lgamma(1) = 0,
    # so result = -infinity
    sum_x = math_ops.reduce_sum(x, axis=[-1])
    log_gamma_sum_x = math_ops.lgamma(sum_x)
    result = log_prod_gamma_x - log_gamma_sum_x

    return result


def einsum(equation, *inputs, **kwargs):
  """A generalized contraction between tensors of arbitrary dimension.

  This function returns a tensor whose elements are defined by `equation`,
  which is written in a shorthand form inspired by the Einstein summation
  convention.  As an example, consider multiplying two matrices
  A and B to form a matrix C.  The elements of C are given by:

  ```
    C[i,k] = sum_j A[i,j] * B[j,k]
  ```

  The corresponding `equation` is:

  ```
    ij,jk->ik
  ```

  In general, the `equation` is obtained from the more familiar element-wise
  equation by
    1. removing variable names, brackets, and commas,
    2. replacing "*" with ",",
    3. dropping summation signs, and
    4. moving the output to the right, and replacing "=" with "->".

  Many common operations can be expressed in this way.  For example:

  ```python
  # Matrix multiplication
  >>> einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]

  # Dot product
  >>> einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]

  # Outer product
  >>> einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]

  # Transpose
  >>> einsum('ij->ji', m)  # output[j,i] = m[i,j]

  # Batch matrix multiplication
  >>> einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
  ```

  This function behaves like `numpy.einsum`, but does not support:

  * Ellipses (subscripts like `ij...,jk...->ik...`)
  * Subscripts where an axis appears more than once for a single input
    (e.g. `ijj,k->ik`).
  * Subscripts that are summed across multiple inputs (e.g., `ij,ij,jk->ik`).

  Args:
    equation: a `str` describing the contraction, in the same format as
      `numpy.einsum`.
    *inputs: the inputs to contract (each one a `Tensor`), whose shapes should
      be consistent with `equation`.
    name: A name for the operation (optional).

  Returns:
    The contracted `Tensor`, with shape determined by `equation`.

  Raises:
    ValueError: If
      - the format of `equation` is incorrect,
      - the number of inputs implied by `equation` does not match `len(inputs)`,
      - an axis appears in the output subscripts but not in any of the inputs,
      - the number of dimensions of an input differs from the number of
        indices in its subscript, or
      - the input shapes are inconsistent along a particular axis.
  """
  name = kwargs.pop("name", None)
  if kwargs:
    raise TypeError("invalid keyword arguments for this function: " +
                    ", ".join([format(key)
                               for key in sorted(list(kwargs.keys()))]))
  with ops.name_scope(name, "einsum", [equation, inputs]) as name:
    if '...' in equation:
      raise ValueError('Subscripts with ellipses are not yet supported.')

    match = re.match('([a-z,]+)(->[a-z]*)?', equation)
    if not match:
      raise ValueError(
          'Indices have incorrect format: %s' % equation
      )

    inputs = list(inputs)
    input_axis_labels = match.group(1).split(',')

    if len(inputs) != len(input_axis_labels):
      raise ValueError('Got %d arguments for equation "%s", expecting %d' % (
          len(inputs), equation, len(input_axis_labels)))

    axis_labels = set(''.join(input_axis_labels))
    if match.group(2):
      output_axis_labels = match.group(2)[2:]
    else:
      # infer the output subscripts if not given, assume alphabetical order
      indices = ''.join(sorted(axis_labels))
      counts = {ax: 0 for ax in indices}
      for axes_ in input_axis_labels:
        for ax in axes_:
          counts[ax] += 1

      output_axis_labels = ''.join(sorted(
          ax for ax in indices
          if counts[ax] == 1
      ))

    for a in axis_labels:
      input_count = sum(1 for s in input_axis_labels if a in s)
      if input_count > 2 and a not in output_axis_labels:
        logging.warn(
            'Falling back to exponential-space implementation of einsum() because'
            ' index "%s" is summed over more than two inputs.', a)
        return _exponential_space_einsum(equation, *inputs)

    temp = inputs[0]
    temp_axis_labels = input_axis_labels[0]
    for i in xrange(len(inputs)-1):
      axes_to_sum = (set(temp_axis_labels) & set(input_axis_labels[i+1])
                     - set(output_axis_labels))
      temp, temp_axis_labels = _einsum_reduction(temp,
                                                 temp_axis_labels,
                                                 inputs[i+1],
                                                 input_axis_labels[i+1],
                                                 axes_to_sum)

    missing_indices = set(temp_axis_labels) - set(output_axis_labels)
    if missing_indices:
      reduction_indices = [i for i, a in enumerate(temp_axis_labels)
                           if a not in output_axis_labels]
      temp = math_ops.reduce_sum(temp, reduction_indices=reduction_indices)
      temp_axis_labels = ''.join(a for a in temp_axis_labels
                                 if a in output_axis_labels)

    if sorted(temp_axis_labels) != sorted(output_axis_labels):
      raise ValueError('Invalid equation: %s' % equation)

    perm = [temp_axis_labels.index(a) for a in output_axis_labels]
    return _transpose_if_necessary(temp, perm)


def _einsum_reduction(t0, t0_axis_labels, t1, t1_axis_labels, axes_to_sum):
  """Helper for einsum() that computes the result of a two-argument einsum().

  Args:
    t0: a `Tensor`
    t0_axis_labels: a string of axis labels.  This string's length must equal
      the rank of t0.
    t1: a `Tensor`
    t1_axis_labels: a string to axis labels.  This string's length must equal
      the rank of t1.
    axes_to_sum: set of labels of axes to be summed over

  Returns:
    A `Tensor` whose elements are obtained by summing, over all axes in
    `axes_to_sum`, the corresponding elements of `t0` and `t1`.

    For example, if t0_axis_labels == 'abijk', t1_axis_labels == 'acjkl', and
    axes_to_sum == {j,k}, this will return a tensor x where

      out[a,b,c,i,l] = sum_j sum_k t0[a,b,i,j,k] * t1[a,c,j,k,l]

  Raises:
    ValueError: if the rank of `t0` does not match the length of
      `t0_axis_labels`, or that of `t1` does not match the length of
      `t1_axis_labels`.
  """
  if len(t0_axis_labels) != len(t0.get_shape()):
    raise ValueError(
        'Tensor t0 of rank %d does not match einsum reduction of length %d' %
        (len(t0.get_shape()), len(t0_axis_labels)))
  if len(t1_axis_labels) != len(t1.get_shape()):
    raise ValueError(
        'Tensor t1 of rank %d does not match einsum reduction of length %d' %
        (len(t1.get_shape()), len(t1_axis_labels)))

  # This function computes the result of a two-argument einsum() using batch
  # matrix multiplication.  This involves
  # 1. transposing t0 and t1 so that axes are in the correct order for
  #    batch matrix multiplication, and
  # 2. reshaping t0 and t1 so that they are both of rank 3.

  # First, we divide axes into three groups:
  #  * "preserved" axes are present in both inputs and the output
  #  * "summed" axes are present in both inputs but not the output
  #  * "broadcast" axes are present in exactly one input and the output
  #
  # As an example, if the einsum is abijk,acjkl->abcil, then "a" is a
  # preserved axis, "b" and "c" are broadcast axes, and "j" and "k" are
  # summed axes.
  assert all(a in t0_axis_labels and a in t1_axis_labels for a in axes_to_sum)
  preserved_axes = (set(t0_axis_labels) & set(t1_axis_labels)) - axes_to_sum
  broadcast_axes = {}
  for i, sym_list in enumerate([t0_axis_labels, t1_axis_labels]):
    broadcast_axes[i] = set(sym_list) - preserved_axes - axes_to_sum

  # Reorder the axes so that:
  # 1. preserved axes come first in both inputs
  # 2. in input 0, broadcast axes come next, followed by summed axes
  # 3. in input 1, summed axes come next, followed by broadcast axes
  def sort_key(input_index, a):
    if a in preserved_axes:
      return (-1, a)
    elif ((input_index == 0 and a in broadcast_axes[0]) or
          (input_index == 1 and a in axes_to_sum)):
      return (0, a)
    else:
      return (1, a)

  axis_labels = [t0_axis_labels, t1_axis_labels]
  sorted_axes = [sorted(sym_list, key=lambda a: sort_key(i, a))
                 for i, sym_list in enumerate(axis_labels)]
  inputs = [t0, t1]
  for i, axes_str in enumerate(axis_labels):
    perm = [axes_str.find(a) for a in sorted_axes[i]]
    inputs[i] = _transpose_if_necessary(inputs[i], perm)
  t0, t1 = inputs

  if not axes_to_sum:
    # In the special case where there are no axes to sum over, reduce to mul()
    # rather than to batch matrix multiplication.
    for _ in broadcast_axes[1]:
      t0 = array_ops.expand_dims(t0, -1)
    for _ in broadcast_axes[0]:
      t1 = array_ops.expand_dims(t1, len(preserved_axes))
    product = math_ops.multiply(t0, t1)
    product_axes = sorted_axes[0] + sorted_axes[1][len(preserved_axes):]
    return product, ''.join(product_axes)
  else:
    # Reduce to matmul().

    # Reshape both inputs so as to combine multiple broadcast axes
    # into a single axis, and combine multiple summed axes into a
    # single axis.

    t0_shape = _get_shape(t0)
    num_broadcast_elements_t0 = _total_size(
        t0_shape[len(preserved_axes):-len(axes_to_sum)])
    num_summed_elements = _total_size(t0_shape[-len(axes_to_sum):])
    new_shape = (t0_shape[:len(preserved_axes)]
                 + [num_broadcast_elements_t0, num_summed_elements])
    t0 = _reshape_if_necessary(t0, new_shape)

    t1_shape = _get_shape(t1)
    num_broadcast_elements_t1 = _total_size(
        t1_shape[len(preserved_axes)+len(axes_to_sum):])
    new_shape = (t1_shape[:len(preserved_axes)]
                 + [num_summed_elements, num_broadcast_elements_t1])
    t1 = _reshape_if_necessary(t1, new_shape)

    product = math_ops.matmul(t0, t1)

    # Undo compaction of broadcast axes
    uncompacted_shape = (
        t0_shape[:len(preserved_axes)+len(broadcast_axes[0])]
        + t1_shape[len(t1_shape)-len(broadcast_axes[1]):]
    )
    product = _reshape_if_necessary(product, uncompacted_shape)

    product_axes = (
        sorted_axes[0][:len(preserved_axes)+len(broadcast_axes[0])] +
        sorted_axes[1][len(sorted_axes[1])-len(broadcast_axes[1]):]
    )

    return product, ''.join(product_axes)


def _transpose_if_necessary(tensor, perm):
  """Like transpose(), but avoids creating a new tensor if possible."""
  if perm != range(len(perm)):
    return array_ops.transpose(tensor, perm=perm)
  else:
    return tensor


def _reshape_if_necessary(tensor, new_shape):
  """Like reshape(), but avoids creating a new tensor if possible."""
  # Accept None as an alias for -1 in new_shape.
  new_shape = tuple(-1 if x is None else x for x in new_shape)
  cur_shape = tuple(x.value for x in tensor.get_shape())
  if (len(new_shape) == len(cur_shape) and
      all(d0 == d1 or d1 == -1 for d0, d1 in zip(cur_shape, new_shape))):
    return tensor
  else:
    return array_ops.reshape(tensor, new_shape)


def _get_shape(tensor):
  """Like get_shape().as_list(), but explicitly queries the shape of a tensor
  if necessary to ensure that the returned value contains no unknown value."""

  shape = tensor.get_shape().as_list()
  none_indices = [i for i, d in enumerate(shape) if d is None]
  if none_indices:
    # Query the shape if shape contains None values
    shape_tensor = array_ops.shape(tensor)
    for i in none_indices:
      shape[i] = shape_tensor[i]
  return shape


def _total_size(shape_values):
  """Given list of tensor shape values, returns total size.
  If shape_values contains tensor values (which are results of
  array_ops.shape), then it returns a scalar tensor.
  If not, it returns an integer."""

  result = 1
  for val in shape_values:
    result *= val
  return result


def _exponential_space_einsum(equation, *inputs):
  """Fallback implementation that supports summing an index over > 2 inputs."""
  if '...' in equation:
    raise ValueError("Subscripts with ellipses are not yet supported.")

  match = re.match('([a-z,]+)(->[a-z]*)?', equation)
  if not match:
    raise ValueError(
        'Indices have incorrect format: %s' % equation
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
        'Expected %d inputs but got %d' % (len(idx_in), len(inputs))
    )

  missing_idx = set(idx_out).difference(idx_all)
  if missing_idx:
    raise ValueError(
        'Unknown output axes: %s' % missing_idx
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
        'Input %d with axes %s has incorrect' \
        ' number of dimensions (expected %d, got %d)' % (
          i, axes_, len(axes_), input_.get_shape().ndims
        )
      )

    sorted_idx = sorted(axes_, key=axis_order.get)

    if len(set(axes_)) != len(axes_):
      raise ValueError(
          'Subscript not supported: an axis appears more than once: %s' % axes_
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
          'Dimension mismatch on axis: %s' % ax
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
