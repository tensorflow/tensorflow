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
"""Functions for computing moving-average statistics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


__all__ = [
    "assign_exponential_moving_mean_variance",
    "exponential_moving_mean_variance",
]


def assign_exponential_moving_mean_variance(
    mean_var, variance_var, value, decay, name=None):
  """Compute the exponential-moving mean, variance of a streaming value.

  The exponential moving `mean_var`, `variance_var` updated by `value` is
  given by the following recurrence relations,

  ```python
  variance_var = decay * (variance_var + (1-decay) * (value - mean_var)**2)
  mean_var     = decay * mean_var + (1 - decay) * value
  ```

  Note: `mean_var` is updated *after* `variance_var`, i.e., `variance_var` uses
  the lag-1 mean.

  For derivation justification, see equation 143 of:
    T. Finch, Feb 2009. "Incremental calculation of weighted mean and variance".
    http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

  Args:
    mean_var: `float`-like `Variable` representing the exponential-moving
      mean. Same shape as `variance_var` and `value`.
    variance_var: `float`-like `Variable` representing the exponential-moving
      variance. Same shape as `mean_var` and `value`.
    value: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
    decay: A `float`-like `Tensor`. The moving average decay. Typically close to
      `1.`, e.g., `0.999`.
    name: Optional name of the returned operation.

  Returns:
    mean_var: `Variable` representing the `value`-updated exponential-moving
      mean.
    variance_var: `Variable` representing the `value`-updated exponential-moving
      variance.

  Raises:
    TypeError: if `mean_var` is not a floating type.
    TypeError: if `mean_var`, `variance_var`, `value`, `decay` have different
      `base_dtype`.
  """
  with ops.name_scope(name, "assign_exponential_moving_mean_variance",
                      [variance_var, mean_var, value, decay]):
    with ops.colocate_with(variance_var):
      with ops.colocate_with(mean_var):
        base_dtype = mean_var.dtype.base_dtype
        if not base_dtype.is_floating:
          raise TypeError(
              "mean_var.base_dtype({}) is not a floating-type.".format(
                  base_dtype.name))
        if base_dtype != variance_var.dtype.base_dtype:
          raise TypeError(
              "mean_var.base_dtype({}) != variance_var.base_dtype({})".format(
                  base_dtype.name,
                  variance_var.dtype.base_dtype.name))
        value = ops.convert_to_tensor(value, dtype=base_dtype, name="value")
        decay = ops.convert_to_tensor(decay, dtype=base_dtype, name="decay")
        delta = value - mean_var
        with ops.control_dependencies([delta]):
          mean_var = state_ops.assign_add(
              mean_var,
              (1. - decay) * delta)
          variance_var = state_ops.assign_sub(
              variance_var,
              (1. - decay) * (variance_var - decay * math_ops.square(delta)))
        return mean_var, variance_var


def exponential_moving_mean_variance(value, decay, collections=None, name=None):
  """Compute the exponential-moving mean, variance of a streaming value.

  The exponential moving `mean_var`, `variance_var` updated by `value` is
  given by the following recurrence relations,

  ```python
  variance_var = decay * (variance_var + (1-decay) * (value - mean_var)**2)
  mean_var     = decay * mean_var + (1 - decay) * value
  ```

  Note: `mean_var` is updated *after* `variance_var`, i.e., `variance_var` uses
  the lag-`1` mean.

  For derivation justification, see equation 143 of:
    T. Finch, Feb 2009. "Incremental calculation of weighted mean and variance".
    http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

  Unlike `assign_exponential_moving_mean_variance`, this function handles
  variable creation.

  Args:
    value: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
    decay: A `float`-like `Tensor`. The moving average decay. Typically close to
      `1.`, e.g., `0.999`.
    collections: Python list of graph-collections keys to which the the internal
      variables `mean_var` and `variance_var` are added.
      Default value is `[GraphKeys.GLOBAL_VARIABLES]`.
    name: Optional name of the returned operation.

  Returns:
    mean_var: `Variable` representing the `value`-updated exponential-moving
      mean.
    variance_var: `Variable` representing the `value`-updated exponential-moving
      variance.

  Raises:
    TypeError: if `value_var` is not a floating type.
    TypeError: if `value`, `decay` have different `base_dtype`.
  """
  if collections is None:
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  with variable_scope.variable_scope(
      name, "exponential_moving_mean_variance", [value, decay]):
    value = ops.convert_to_tensor(value, name="value")
    base_dtype = value.dtype.base_dtype
    if not base_dtype.is_floating:
      raise TypeError(
          "value.base_dtype({}) is not a floating-type.".format(
              base_dtype.name))
    decay = ops.convert_to_tensor(decay, dtype=base_dtype, name="decay")
    variance_var = variable_scope.get_variable(
        "exponential_moving_variance",
        shape=value.shape,
        dtype=value.dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=collections)
    mean_var = variable_scope.get_variable(
        "exponential_moving_mean",
        shape=value.shape,
        dtype=value.dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=collections)
    return assign_exponential_moving_mean_variance(
        mean_var, variance_var, value, decay)
