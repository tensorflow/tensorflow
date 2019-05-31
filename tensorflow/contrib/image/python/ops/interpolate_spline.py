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
"""Polyharmonic spline interpolation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

EPSILON = 0.0000000001


def _cross_squared_distance_matrix(x, y):
  """Pairwise squared distance between two (batch) matrices' rows (2nd dim).

  Computes the pairwise distances between rows of x and rows of y
  Args:
    x: [batch_size, n, d] float `Tensor`
    y: [batch_size, m, d] float `Tensor`

  Returns:
    squared_dists: [batch_size, n, m] float `Tensor`, where
    squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2
  """
  x_norm_squared = math_ops.reduce_sum(math_ops.square(x), 2)
  y_norm_squared = math_ops.reduce_sum(math_ops.square(y), 2)

  # Expand so that we can broadcast.
  x_norm_squared_tile = array_ops.expand_dims(x_norm_squared, 2)
  y_norm_squared_tile = array_ops.expand_dims(y_norm_squared, 1)

  x_y_transpose = math_ops.matmul(x, y, adjoint_b=True)

  # squared_dists[b,i,j] = ||x_bi - y_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
  squared_dists = x_norm_squared_tile - 2 * x_y_transpose + y_norm_squared_tile

  return squared_dists


def _pairwise_squared_distance_matrix(x):
  """Pairwise squared distance among a (batch) matrix's rows (2nd dim).

  This saves a bit of computation vs. using _cross_squared_distance_matrix(x,x)

  Args:
    x: `[batch_size, n, d]` float `Tensor`

  Returns:
    squared_dists: `[batch_size, n, n]` float `Tensor`, where
    squared_dists[b,i,j] = ||x[b,i,:] - x[b,j,:]||^2
  """

  x_x_transpose = math_ops.matmul(x, x, adjoint_b=True)
  x_norm_squared = array_ops.matrix_diag_part(x_x_transpose)
  x_norm_squared_tile = array_ops.expand_dims(x_norm_squared, 2)

  # squared_dists[b,i,j] = ||x_bi - x_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
  squared_dists = x_norm_squared_tile - 2 * x_x_transpose + array_ops.transpose(
      x_norm_squared_tile, [0, 2, 1])

  return squared_dists


def _solve_interpolation(train_points, train_values, order,
                         regularization_weight):
  """Solve for interpolation coefficients.

  Computes the coefficients of the polyharmonic interpolant for the 'training'
  data defined by (train_points, train_values) using the kernel phi.

  Args:
    train_points: `[b, n, d]` interpolation centers
    train_values: `[b, n, k]` function values
    order: order of the interpolation
    regularization_weight: weight to place on smoothness regularization term

  Returns:
    w: `[b, n, k]` weights on each interpolation center
    v: `[b, d, k]` weights on each input dimension
  Raises:
    ValueError: if d or k is not fully specified.
  """

  # These dimensions are set dynamically at runtime.
  b, n, _ = array_ops.unstack(array_ops.shape(train_points), num=3)

  d = train_points.shape[-1]
  if tensor_shape.dimension_value(d) is None:
    raise ValueError('The dimensionality of the input points (d) must be '
                     'statically-inferrable.')

  k = train_values.shape[-1]
  if tensor_shape.dimension_value(k) is None:
    raise ValueError('The dimensionality of the output values (k) must be '
                     'statically-inferrable.')

  # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
  # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
  # To account for python style guidelines we use
  # matrix_a for A and matrix_b for B.

  c = train_points
  f = train_values

  # Next, construct the linear system.
  with ops.name_scope('construct_linear_system'):

    matrix_a = _phi(_pairwise_squared_distance_matrix(c), order)  # [b, n, n]
    if regularization_weight > 0:
      batch_identity_matrix = array_ops.expand_dims(
          linalg_ops.eye(n, dtype=c.dtype), 0)
      matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term in the linear model.
    ones = array_ops.ones_like(c[..., :1], dtype=c.dtype)
    matrix_b = array_ops.concat([c, ones], 2)  # [b, n, d + 1]

    # [b, n + d + 1, n]
    left_block = array_ops.concat(
        [matrix_a, array_ops.transpose(matrix_b, [0, 2, 1])], 1)

    num_b_cols = matrix_b.get_shape()[2]  # d + 1
    lhs_zeros = array_ops.zeros([b, num_b_cols, num_b_cols], train_points.dtype)
    right_block = array_ops.concat([matrix_b, lhs_zeros],
                                   1)  # [b, n + d + 1, d + 1]
    lhs = array_ops.concat([left_block, right_block],
                           2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = array_ops.zeros([b, d + 1, k], train_points.dtype)
    rhs = array_ops.concat([f, rhs_zeros], 1)  # [b, n + d + 1, k]

  # Then, solve the linear system and unpack the results.
  with ops.name_scope('solve_linear_system'):
    w_v = linalg_ops.matrix_solve(lhs, rhs)
    w = w_v[:, :n, :]
    v = w_v[:, n:, :]

  return w, v


def _apply_interpolation(query_points, train_points, w, v, order):
  """Apply polyharmonic interpolation model to data.

  Given coefficients w and v for the interpolation model, we evaluate
  interpolated function values at query_points.

  Args:
    query_points: `[b, m, d]` x values to evaluate the interpolation at
    train_points: `[b, n, d]` x values that act as the interpolation centers
                    ( the c variables in the wikipedia article)
    w: `[b, n, k]` weights on each interpolation center
    v: `[b, d, k]` weights on each input dimension
    order: order of the interpolation

  Returns:
    Polyharmonic interpolation evaluated at points defined in query_points.
  """

  # First, compute the contribution from the rbf term.
  pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
  phi_pairwise_dists = _phi(pairwise_dists, order)

  rbf_term = math_ops.matmul(phi_pairwise_dists, w)

  # Then, compute the contribution from the linear term.
  # Pad query_points with ones, for the bias term in the linear model.
  query_points_pad = array_ops.concat([
      query_points,
      array_ops.ones_like(query_points[..., :1], train_points.dtype)
  ], 2)
  linear_term = math_ops.matmul(query_points_pad, v)

  return rbf_term + linear_term


def _phi(r, order):
  """Coordinate-wise nonlinearity used to define the order of the interpolation.

  See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.

  Args:
    r: input op
    order: interpolation order

  Returns:
    phi_k evaluated coordinate-wise on r, for k = r
  """

  # using EPSILON prevents log(0), sqrt0), etc.
  # sqrt(0) is well-defined, but its gradient is not
  with ops.name_scope('phi'):
    if order == 1:
      r = math_ops.maximum(r, EPSILON)
      r = math_ops.sqrt(r)
      return r
    elif order == 2:
      return 0.5 * r * math_ops.log(math_ops.maximum(r, EPSILON))
    elif order == 4:
      return 0.5 * math_ops.square(r) * math_ops.log(
          math_ops.maximum(r, EPSILON))
    elif order % 2 == 0:
      r = math_ops.maximum(r, EPSILON)
      return 0.5 * math_ops.pow(r, 0.5 * order) * math_ops.log(r)
    else:
      r = math_ops.maximum(r, EPSILON)
      return math_ops.pow(r, 0.5 * order)


def interpolate_spline(train_points,
                       train_values,
                       query_points,
                       order,
                       regularization_weight=0.0,
                       name='interpolate_spline'):
  r"""Interpolate signal using polyharmonic interpolation.

  The interpolant has the form
  $$f(x) = \sum_{i = 1}^n w_i \phi(||x - c_i||) + v^T x + b.$$

  This is a sum of two terms: (1) a weighted sum of radial basis function (RBF)
  terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term with a bias.
  The \\(c_i\\) vectors are 'training' points. In the code, b is absorbed into v
  by appending 1 as a final dimension to x. The coefficients w and v are
  estimated such that the interpolant exactly fits the value of the function at
  the \\(c_i\\) points, the vector w is orthogonal to each \\(c_i\\), and the
  vector w sums to 0. With these constraints, the coefficients can be obtained
  by solving a linear system.

  \\(\phi\\) is an RBF, parametrized by an interpolation
  order. Using order=2 produces the well-known thin-plate spline.

  We also provide the option to perform regularized interpolation. Here, the
  interpolant is selected to trade off between the squared loss on the training
  data and a certain measure of its curvature
  ([details](https://en.wikipedia.org/wiki/Polyharmonic_spline)).
  Using a regularization weight greater than zero has the effect that the
  interpolant will no longer exactly fit the training data. However, it may be
  less vulnerable to overfitting, particularly for high-order interpolation.

  Note the interpolation procedure is differentiable with respect to all inputs
  besides the order parameter.

  We support dynamically-shaped inputs, where batch_size, n, and m are None
  at graph construction time. However, d and k must be known.

  Args:
    train_points: `[batch_size, n, d]` float `Tensor` of n d-dimensional
      locations. These do not need to be regularly-spaced.
    train_values: `[batch_size, n, k]` float `Tensor` of n c-dimensional values
      evaluated at train_points.
    query_points: `[batch_size, m, d]` `Tensor` of m d-dimensional locations
      where we will output the interpolant's values.
    order: order of the interpolation. Common values are 1 for
      \\(\phi(r) = r\\), 2 for \\(\phi(r) = r^2 * log(r)\\) (thin-plate spline),
       or 3 for \\(\phi(r) = r^3\\).
    regularization_weight: weight placed on the regularization term.
      This will depend substantially on the problem, and it should always be
      tuned. For many problems, it is reasonable to use no regularization.
      If using a non-zero value, we recommend a small value like 0.001.
    name: name prefix for ops created by this function

  Returns:
    `[b, m, k]` float `Tensor` of query values. We use train_points and
    train_values to perform polyharmonic interpolation. The query values are
    the values of the interpolant evaluated at the locations specified in
    query_points.
  """
  with ops.name_scope(name):
    train_points = ops.convert_to_tensor(train_points)
    train_values = ops.convert_to_tensor(train_values)
    query_points = ops.convert_to_tensor(query_points)

    # First, fit the spline to the observed data.
    with ops.name_scope('solve'):
      w, v = _solve_interpolation(train_points, train_values, order,
                                  regularization_weight)

    # Then, evaluate the spline at the query locations.
    with ops.name_scope('predict'):
      query_values = _apply_interpolation(query_points, train_points, w, v,
                                          order)

  return query_values
