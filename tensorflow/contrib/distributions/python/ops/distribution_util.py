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
"""Utilities for probability distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops


def assert_close(
    x, y, data=None, summarize=None, message=None, name="assert_close"):
  """Assert that that x and y are within machine epsilon of each other.

  Args:
    x: Numeric `Tensor`
    y: Numeric `Tensor`
    data: The tensors to print out if the condition is `False`. Defaults to
      error message and first few entries of `x` and `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).

  Returns:
    Op raising `InvalidArgumentError` if |x - y| > machine epsilon.
  """
  message = message or ""
  x = ops.convert_to_tensor(x, name="x")
  y = ops.convert_to_tensor(y, name="y")

  if x.dtype.is_integer:
    return check_ops.assert_equal(
        x, y, data=data, summarize=summarize, message=message, name=name)

  with ops.name_scope(name, "assert_close", [x, y, data]):
    tol = np.finfo(x.dtype.as_numpy_dtype).resolution
    if data is None:
      data = [
          message,
          "Condition x ~= y did not hold element-wise: x = ", x.name, x, "y = ",
          y.name, y
      ]
    condition = math_ops.reduce_all(math_ops.less_equal(math_ops.abs(x-y), tol))
    return logging_ops.Assert(
        condition, data, summarize=summarize)


def assert_integer_form(
    x, data=None, summarize=None, message=None, name="assert_integer_form"):
  """Assert that x has integer components (or floats equal to integers).

  Args:
    x: Numeric `Tensor`
    data: The tensors to print out if the condition is `False`. Defaults to
      error message and first few entries of `x` and `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).

  Returns:
    Op raising `InvalidArgumentError` if round(x) != x.
  """

  message = message or "x has non-integer components"
  x = ops.convert_to_tensor(x, name="x")
  casted_x = math_ops.to_int64(x)
  return check_ops.assert_equal(
      x, math_ops.cast(math_ops.round(casted_x), x.dtype),
      data=data, summarize=summarize, message=message, name=name)


def assert_symmetric(matrix):
  matrix_t = array_ops.batch_matrix_transpose(matrix)
  return control_flow_ops.with_dependencies(
      [check_ops.assert_equal(matrix, matrix_t)], matrix)


def get_logits_and_prob(
    logits=None, p=None, multidimensional=False, validate_args=True, name=None):
  """Converts logits to probabilities and vice-versa, and returns both.

  Args:
    logits: Numeric `Tensor` representing log-odds.
    p: Numeric `Tensor` representing probabilities.
    multidimensional: Given `p` a [N1, N2, ... k] dimensional tensor,
      whether the last dimension represents the probability between k classes.
      This will additionally assert that the values in the last dimension
      sum to one. If `False`, will instead assert that each value is in
      `[0, 1]`.
    validate_args: Whether to assert `0 <= p <= 1` if multidimensional is
      `False`, otherwise that the last dimension of `p` sums to one.
    name: A name for this operation (optional).

  Returns:
    Tuple with `logits` and `p`. If `p` has an entry that is `0` or `1`, then
    the corresponding entry in the returned logits will be `-Inf` and `Inf`
    respectively.

  Raises:
    ValueError: if neither `p` nor `logits` were passed in, or both were.
  """
  if p is None and logits is None:
    raise ValueError("Must pass p or logits.")
  elif p is not None and logits is not None:
    raise ValueError("Must pass either p or logits, not both.")
  elif p is None:
    with ops.name_scope(name, values=[logits]):
      logits = array_ops.identity(logits, name="logits")
    with ops.name_scope(name):
      with ops.name_scope("p"):
        p = math_ops.sigmoid(logits)
  elif logits is None:
    with ops.name_scope(name):
      with ops.name_scope("p"):
        p = array_ops.identity(p)
        if validate_args:
          one = constant_op.constant(1., p.dtype)
          dependencies = [check_ops.assert_non_negative(p)]
          if multidimensional:
            dependencies += [assert_close(
                math_ops.reduce_sum(p, reduction_indices=[-1]),
                one, message="p does not sum to 1.")]
          else:
            dependencies += [check_ops.assert_less_equal(
                p, one, message="p has components greater than 1.")]
          p = control_flow_ops.with_dependencies(dependencies, p)
      with ops.name_scope("logits"):
        logits = math_ops.log(p) - math_ops.log(1. - p)
  return (logits, p)


def log_combinations(n, counts, name="log_combinations"):
  """Multinomial coefficient.

  Given `n` and `counts`, where `counts` has last dimension `k`, we compute
  the multinomial coefficient as:

  ```n! / sum_i n_i!```

  where `i` runs over all `k` classes.

  Args:
    n: Numeric `Tensor` broadcastable with `counts`. This represents `n`
      outcomes.
    counts: Numeric `Tensor` broadcastable with `n`. This represents counts
      in `k` classes, where `k` is the last dimension of the tensor.
    name: A name for this operation (optional).

  Returns:
    `Tensor` representing the multinomial coefficient between `n` and `counts`.
  """
  # First a bit about the number of ways counts could have come in:
  # E.g. if counts = [1, 2], then this is 3 choose 2.
  # In general, this is (sum counts)! / sum(counts!)
  # The sum should be along the last dimension of counts.  This is the
  # "distribution" dimension. Here n a priori represents the sum of counts.
  with ops.name_scope(name, values=[n, counts]):
    total_permutations = math_ops.lgamma(n + 1)
    counts_factorial = math_ops.lgamma(counts + 1)
    redundant_permutations = math_ops.reduce_sum(counts_factorial,
                                                 reduction_indices=[-1])
    return total_permutations - redundant_permutations


def batch_matrix_diag_transform(matrix, transform=None, name=None):
  """Transform diagonal of [batch-]matrix, leave rest of matrix unchanged.

  Create a trainable covariance defined by a Cholesky factor:

  ```python
  # Transform network layer into 2 x 2 array.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))

  # Make the diagonal positive.  If the upper triangle was zero, this would be a
  # valid Cholesky factor.
  chol = batch_matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # OperatorPDCholesky ignores the upper triangle.
  operator = OperatorPDCholesky(chol)
  ```

  Example of heteroskedastic 2-D linear regression.

  ```python
  # Get a trainable Cholesky factor.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))
  chol = batch_matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # Get a trainable mean.
  mu = tf.contrib.layers.fully_connected(activations, 2)

  # This is a fully trainable multivariate normal!
  dist = tf.contrib.distributions.MVNCholesky(mu, chol)

  # Standard log loss.  Minimizing this will "train" mu and chol, and then dist
  # will be a distribution predicting labels as multivariate Gaussians.
  loss = -1 * tf.reduce_mean(dist.log_pdf(labels))
  ```

  Args:
    matrix:  Rank `R` `Tensor`, `R >= 2`, where the last two dimensions are
      equal.
    transform:  Element-wise function mapping `Tensors` to `Tensors`.  To
      be applied to the diagonal of `matrix`.  If `None`, `matrix` is returned
      unchanged.  Defaults to `None`.
    name:  A name to give created ops.
      Defaults to "batch_matrix_diag_transform".

  Returns:
    A `Tensor` with same shape and `dtype` as `matrix`.
  """
  with ops.name_scope(name, "batch_matrix_diag_transform", [matrix]):
    matrix = ops.convert_to_tensor(matrix, name="matrix")
    if transform is None:
      return matrix
    # Replace the diag with transformed diag.
    diag = array_ops.batch_matrix_diag_part(matrix)
    transformed_diag = transform(diag)
    transformed_mat = array_ops.batch_matrix_set_diag(matrix, transformed_diag)

  return transformed_mat
