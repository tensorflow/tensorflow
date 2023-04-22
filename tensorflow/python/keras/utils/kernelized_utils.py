# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility methods related to kernelized layers."""

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _to_matrix(u):
  """If input tensor is a vector (i.e., has rank 1), converts it to matrix."""
  u_rank = len(u.shape)
  if u_rank not in [1, 2]:
    raise ValueError('The input tensor should have rank 1 or 2. Given rank: {}'
                     .format(u_rank))
  if u_rank == 1:
    return array_ops.expand_dims(u, 0)
  return u


def _align_matrices(x, y):
  """Aligns x and y tensors to allow computations over pairs of their rows."""
  x_matrix = _to_matrix(x)
  y_matrix = _to_matrix(y)
  x_shape = x_matrix.shape
  y_shape = y_matrix.shape
  if y_shape[1] != x_shape[1]:  # dimensions do not match.
    raise ValueError(
        'The outermost dimensions of the input tensors should match. Given: {} '
        'vs {}.'.format(y_shape[1], x_shape[1]))

  x_tile = array_ops.tile(
      array_ops.expand_dims(x_matrix, 1), [1, y_shape[0], 1])
  y_tile = array_ops.tile(
      array_ops.expand_dims(y_matrix, 0), [x_shape[0], 1, 1])
  return x_tile, y_tile


def inner_product(u, v):
  u = _to_matrix(u)
  v = _to_matrix(v)
  return math_ops.matmul(u, v, transpose_b=True)


def exact_gaussian_kernel(x, y, stddev):
  r"""Computes exact Gaussian kernel value(s) for tensors x and y and stddev.

  The Gaussian kernel for vectors u, v is defined as follows:
       K(u, v) = exp(-||u-v||^2 / (2* stddev^2))
  where the norm is the l2-norm. x, y can be either vectors or matrices. If they
  are vectors, they must have the same dimension. If they are matrices, they
  must have the same number of columns. In the latter case, the method returns
  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and
  v is a row from y.

  Args:
    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].
    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].
    stddev: The width of the Gaussian kernel.

  Returns:
    A single value (scalar) with shape (1, 1) (if x, y are vectors) or a matrix
      of shape (m, n) with entries K(u, v) (where K is the Gaussian kernel) for
      all (u,v) pairs where u, v are rows from x and y respectively.

  Raises:
    ValueError: if the shapes of x, y are not compatible.
  """
  x_aligned, y_aligned = _align_matrices(x, y)
  diff_squared_l2_norm = math_ops.reduce_sum(
      math_ops.squared_difference(x_aligned, y_aligned), 2)
  return math_ops.exp(-diff_squared_l2_norm / (2 * stddev * stddev))


def exact_laplacian_kernel(x, y, stddev):
  r"""Computes exact Laplacian kernel value(s) for tensors x and y using stddev.

  The Laplacian kernel for vectors u, v is defined as follows:
       K(u, v) = exp(-||u-v|| / stddev)
  where the norm is the l1-norm. x, y can be either vectors or matrices. If they
  are vectors, they must have the same dimension. If they are matrices, they
  must have the same number of columns. In the latter case, the method returns
  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and
  v is a row from y.

  Args:
    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].
    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].
    stddev: The width of the Gaussian kernel.

  Returns:
    A single value (scalar) with shape (1, 1)  if x, y are vectors or a matrix
    of shape (m, n) with entries K(u, v) (where K is the Laplacian kernel) for
    all (u,v) pairs where u, v are rows from x and y respectively.

  Raises:
    ValueError: if the shapes of x, y are not compatible.
  """
  x_aligned, y_aligned = _align_matrices(x, y)
  diff_l1_norm = math_ops.reduce_sum(
      math_ops.abs(math_ops.subtract(x_aligned, y_aligned)), 2)
  return math_ops.exp(-diff_l1_norm / stddev)
