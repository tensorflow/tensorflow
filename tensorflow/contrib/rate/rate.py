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
"""Implementation of tf.contrib.rate module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

_to_replace = re.compile("[^A-Za-z0-9.]")


class Rate(object):
  """Computes the rate of change since the last rate call."""

  def __init__(self, name=None):
    self._built = False
    self._vars = []
    self._initial_values = {}
    name = name or self.__class__.__name__
    # Replace things like spaces in name to create a valid scope name.
    scope_name = _to_replace.sub("_", name)
    # We create the variable scope now to get the unique name that will
    # be used as a variable prefix when build() calls _add_variable().
    with variable_scope.variable_scope(
        scope_name, use_resource=True, reuse=False) as scope:
      pos = scope.name.rfind(scope_name)
      self._name = name + scope.name[pos + len(scope_name):]
      self._scope = scope

    # Ensures that if the user calls build directly we still set self._built to
    # True to prevent variables from being recreated.
    self._build = self.build
    if context.executing_eagerly():
      self._construction_scope = context.eager_mode
    else:
      # We make self.call() into a graph callable here, so that we can
      # return a single op that performs all of the variable updates.
      self._construction_scope = ops.get_default_graph().as_default
      self.call = function.defun(self.call)

  def build(self, values, denominator):
    """Method to create variables.

    Called by `__call__()` before `call()` for the first time.

    Args:
      values: The numerator for rate.
      denominator: Value to which the rate is taken with respect.
    """
    self.numer = self._add_variable(
        name="numer", shape=values.get_shape(), dtype=dtypes.float64)
    self.denom = self._add_variable(
        name="denom", shape=denominator.get_shape(), dtype=dtypes.float64)
    self.prev_values = self._add_variable(
        name="prev_values", shape=values.get_shape(), dtype=dtypes.float64)
    self.prev_denominator = self._add_variable(
        name="prev_denominator",
        shape=denominator.get_shape(),
        dtype=dtypes.float64)
    self._built = True

  def __call__(self, *args, **kwargs):
    """Returns op to execute to update.

    Returns None if eager execution is enabled.
    Returns a graph-mode function if graph execution is enabled.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to Rate, passed on to `call()`.
    """
    if not self._built:
      with variable_scope.variable_scope(
          self._scope), self._construction_scope():
        self.build(*args, **kwargs)
      self._built = True
    return self.call(*args, **kwargs)

  @property
  def name(self):
    return self._name

  @property
  def variables(self):
    return self._vars

  def _add_variable(self, name, shape=None, dtype=None):
    """Private method for adding variables to the graph."""
    if self._built:
      raise RuntimeError("Can't call add_variable() except in build().")
    v = resource_variable_ops.ResourceVariable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        validate_shape=True,
        name=name,
        collections=[ops.GraphKeys.LOCAL_VARIABLES])
    return v

  def call(self, values, denominator):
    """Computes the rate since the last call.

    Args:
      values: Tensor with the per-example value.
      denominator: Measure to take the rate with respect to.

    Returns:
      The rate or 0 if denominator is unchanged since last call.
    """
    if denominator.dtype != dtypes.float64:
      denominator = math_ops.cast(denominator, dtypes.float64)
    if values.dtype != dtypes.float64:
      values = math_ops.cast(values, dtypes.float64)

    state_ops.assign(self.numer, math_ops.subtract(values, self.prev_values))
    state_ops.assign(self.denom,
                     math_ops.subtract(denominator, self.prev_denominator))
    state_ops.assign(self.prev_values, values)
    state_ops.assign(self.prev_denominator, denominator)

    return math_ops.div_no_nan(self.numer,
                               math_ops.maximum(self.denom, 0),
                               name="safe_rate")
