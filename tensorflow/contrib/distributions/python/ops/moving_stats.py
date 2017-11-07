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
"""Functions for computing moving statistics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


__all__ = [
    "assign_moving_mean_variance",
    "assign_log_moving_mean_exp",
    "moving_mean_variance",
]


def assign_moving_mean_variance(
    mean_var, variance_var, value, decay, name=None):
  """Compute exponentially weighted moving {mean,variance} of a streaming value.

  The `value` updated exponentially weighted moving `mean_var` and
  `variance_var` are given by the following recurrence relations:

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
    mean_var: `float`-like `Variable` representing the exponentially weighted
      moving mean. Same shape as `variance_var` and `value`.
    variance_var: `float`-like `Variable` representing the
      exponentially weighted moving variance. Same shape as `mean_var` and
      `value`.
    value: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
    decay: A `float`-like `Tensor`. The moving mean decay. Typically close to
      `1.`, e.g., `0.999`.
    name: Optional name of the returned operation.

  Returns:
    mean_var: `Variable` representing the `value`-updated exponentially weighted
      moving mean.
    variance_var: `Variable` representing the `value`-updated
      exponentially weighted moving variance.

  Raises:
    TypeError: if `mean_var` does not have float type `dtype`.
    TypeError: if `mean_var`, `variance_var`, `value`, `decay` have different
      `base_dtype`.
  """
  with ops.name_scope(name, "assign_moving_mean_variance",
                      [variance_var, mean_var, value, decay]):
    with ops.colocate_with(variance_var):
      with ops.colocate_with(mean_var):
        base_dtype = mean_var.dtype.base_dtype
        if not base_dtype.is_floating:
          raise TypeError(
              "mean_var.base_dtype({}) does not have float type "
              "`dtype`.".format(base_dtype.name))
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


def assign_log_moving_mean_exp(
    log_mean_exp_var, log_value, decay, name=None):
  """Compute the log of the exponentially weighted moving mean of the exp.

  If `log_value` is a draw from a stationary random variable, this function
  approximates `log(E[exp(log_value)])`, i.e., a weighted log-sum-exp. More
  precisely, a `tf.Variable`, `log_mean_exp_var`, is updated by `log_value`
  using the following identity:

  ```none
  log_mean_exp_var =
  = log(decay exp(log_mean_exp_var) + (1 - decay) exp(log_value))
  = log(exp(log_mean_exp_var + log(decay)) + exp(log_value + log1p(-decay)))
  = log_mean_exp_var
    + log(  exp(log_mean_exp_var   - log_mean_exp_var + log(decay))
          + exp(log_value - log_mean_exp_var + log1p(-decay)))
  = log_mean_exp_var
    + log_sum_exp([log(decay), log_value - log_mean_exp_var + log1p(-decay)]).
  ```

  In addition to numerical stability, this formulation is advantageous because
  `log_mean_exp_var` can be updated in a lock-free manner, i.e., using
  `assign_add`. (Note: the updates are not thread-safe; it's just that the
  update to the tf.Variable is presumed efficient due to being lock-free.)

  Args:
    log_mean_exp_var: `float`-like `Variable` representing the log of the
      exponentially weighted moving mean of the exp. Same shape as `log_value`.
    log_value: `float`-like `Tensor` representing a new (streaming) observation.
      Same shape as `log_mean_exp_var`.
    decay: A `float`-like `Tensor`. The moving mean decay. Typically close to
      `1.`, e.g., `0.999`.
    name: Optional name of the returned operation.

  Returns:
    log_mean_exp_var: A reference to the input 'Variable' tensor with the
      `log_value`-updated log of the exponentially weighted moving mean of exp.

  Raises:
    TypeError: if `log_mean_exp_var` does not have float type `dtype`.
    TypeError: if `log_mean_exp_var`, `log_value`, `decay` have different
      `base_dtype`.
  """
  with ops.name_scope(name, "assign_log_moving_mean_exp",
                      [log_mean_exp_var, log_value, decay]):
    # We want to update the variable in a numerically stable and lock-free way.
    # To do this, observe that variable `x` updated by `v` is:
    # x = log(w exp(x) + (1-w) exp(v))
    #   = log(exp(x + log(w)) + exp(v + log1p(-w)))
    #   = x + log(exp(x - x + log(w)) + exp(v - x + log1p(-w)))
    #   = x + lse([log(w), v - x + log1p(-w)])
    with ops.colocate_with(log_mean_exp_var):
      base_dtype = log_mean_exp_var.dtype.base_dtype
      if not base_dtype.is_floating:
        raise TypeError(
            "log_mean_exp_var.base_dtype({}) does not have float type "
            "`dtype`.".format(base_dtype.name))
      log_value = ops.convert_to_tensor(log_value, dtype=base_dtype,
                                        name="log_value")
      decay = ops.convert_to_tensor(decay, dtype=base_dtype, name="decay")
      delta = (log_value - log_mean_exp_var)[array_ops.newaxis, ...]
      x = array_ops.concat([
          math_ops.log(decay) * array_ops.ones_like(delta),
          delta + math_ops.log1p(-decay)
      ], axis=0)
      x = math_ops.reduce_logsumexp(x, axis=0)
      return log_mean_exp_var.assign_add(x)


def moving_mean_variance(value, decay, collections=None, name=None):
  """Compute exponentially weighted moving {mean,variance} of a streaming value.

  The exponentially-weighting moving `mean_var` and `variance_var` are updated
  by `value` according to the following recurrence:

  ```python
  variance_var = decay * (variance_var + (1-decay) * (value - mean_var)**2)
  mean_var     = decay * mean_var + (1 - decay) * value
  ```

  Note: `mean_var` is updated *after* `variance_var`, i.e., `variance_var` uses
  the lag-`1` mean.

  For derivation justification, see equation 143 of:
    T. Finch, Feb 2009. "Incremental calculation of weighted mean and variance".
    http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

  Unlike `assign_moving_mean_variance`, this function handles
  variable creation.

  Args:
    value: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
    decay: A `float`-like `Tensor`. The moving mean decay. Typically close to
      `1.`, e.g., `0.999`.
    collections: Python list of graph-collections keys to which the internal
      variables `mean_var` and `variance_var` are added.
      Default value is `[GraphKeys.GLOBAL_VARIABLES]`.
    name: Optional name of the returned operation.

  Returns:
    mean_var: `Variable` representing the `value`-updated exponentially weighted
      moving mean.
    variance_var: `Variable` representing the `value`-updated
      exponentially weighted moving variance.

  Raises:
    TypeError: if `value_var` does not have float type `dtype`.
    TypeError: if `value`, `decay` have different `base_dtype`.
  """
  if collections is None:
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  with variable_scope.variable_scope(
      name, "moving_mean_variance", [value, decay]):
    value = ops.convert_to_tensor(value, name="value")
    base_dtype = value.dtype.base_dtype
    if not base_dtype.is_floating:
      raise TypeError(
          "value.base_dtype({}) does not have float type `dtype`.".format(
              base_dtype.name))
    decay = ops.convert_to_tensor(decay, dtype=base_dtype, name="decay")
    variance_var = variable_scope.get_variable(
        "moving_variance",
        shape=value.shape,
        dtype=value.dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=collections)
    mean_var = variable_scope.get_variable(
        "moving_mean",
        shape=value.shape,
        dtype=value.dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=collections)
    return assign_moving_mean_variance(
        mean_var, variance_var, value, decay)
