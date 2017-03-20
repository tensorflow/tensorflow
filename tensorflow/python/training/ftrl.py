# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Ftrl-proximal for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class FtrlOptimizer(optimizer.Optimizer):
  """Optimizer that implements the FTRL algorithm.

  See this [paper](
  https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
  """

  def __init__(self, learning_rate,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               use_locking=False, name="Ftrl"):
    """Construct a new FTRL optimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: A float value, must be less or equal to zero.
      initial_accumulator_value: The starting value for accumulators.
        Only positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Ftrl".

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    super(FtrlOptimizer, self).__init__(use_locking, name)

    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value %f needs to be positive" %
                       initial_accumulator_value)
    if learning_rate_power > 0.0:
      raise ValueError("learning_rate_power %f needs to be negative or zero" %
                       learning_rate_power)
    if l1_regularization_strength < 0.0:
      raise ValueError(
          "l1_regularization_strength %f needs to be positive or zero" %
          l1_regularization_strength)
    if l2_regularization_strength < 0.0:
      raise ValueError(
          "l2_regularization_strength %f needs to be positive or zero" %
          l2_regularization_strength)

    self._learning_rate = learning_rate
    self._learning_rate_power = learning_rate_power
    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    self._learning_rate_tensor = None
    self._learning_rate_power_tensor = None
    self._l1_regularization_strength_tensor = None
    self._l2_regularization_strength_tensor = None

  def _create_slots(self, var_list):
    # Create the "accum" and "linear" slots.
    for v in var_list:
      with ops.colocate_with(v):
        val = constant_op.constant(self._initial_accumulator_value,
                                   dtype=v.dtype, shape=v.get_shape())
        self._get_or_make_slot(v, val, "accum", self._name)
        self._zeros_slot(v, "linear", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(
        self._learning_rate,
        name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength,
        name="l1_regularization_strength")
    self._l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength,
        name="l2_regularization_strength")
    self._learning_rate_power_tensor = ops.convert_to_tensor(
        self._learning_rate_power,
        name="learning_rate_power")

  def _apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    return training_ops.apply_ftrl(
        var, accum, linear, grad,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._l1_regularization_strength_tensor,
                      var.dtype.base_dtype),
        math_ops.cast(self._l2_regularization_strength_tensor,
                      var.dtype.base_dtype),
        math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    return training_ops.resource_apply_ftrl(
        var.handle, accum.handle, linear.handle, grad,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        math_ops.cast(self._l1_regularization_strength_tensor,
                      grad.dtype.base_dtype),
        math_ops.cast(self._l2_regularization_strength_tensor,
                      grad.dtype.base_dtype),
        math_ops.cast(self._learning_rate_power_tensor, grad.dtype.base_dtype),
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    return training_ops.sparse_apply_ftrl(
        var, accum, linear, grad.values, grad.indices,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._l1_regularization_strength_tensor,
                      var.dtype.base_dtype),
        math_ops.cast(self._l2_regularization_strength_tensor,
                      var.dtype.base_dtype),
        math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    return training_ops.resource_sparse_apply_ftrl(
        var.handle, accum.handle, linear.handle, grad, indices,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._l1_regularization_strength_tensor,
                      grad.dtype),
        math_ops.cast(self._l2_regularization_strength_tensor,
                      grad.dtype),
        math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
        use_locking=self._use_locking)
