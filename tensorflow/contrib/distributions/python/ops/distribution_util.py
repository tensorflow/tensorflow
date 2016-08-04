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

  with ops.op_scope([x, y, data], name, "assert_close"):
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
    with ops.op_scope([logits], name):
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
  with ops.op_scope([n, counts], name):
    total_permutations = math_ops.lgamma(n + 1)
    counts_factorial = math_ops.lgamma(counts + 1)
    redundant_permutations = math_ops.reduce_sum(counts_factorial,
                                                 reduction_indices=[-1])
    return total_permutations - redundant_permutations
