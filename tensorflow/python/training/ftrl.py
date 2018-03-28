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
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.FtrlOptimizer")
class FtrlOptimizer(optimizer.Optimizer):
  """Optimizer that implements the FTRL algorithm.

  See this [paper](
  https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
  This version has support for both online L2 (the L2 penalty given in the paper
  above) and shrinkage-type L2 (which is the addition of an L2 penalty to the
  loss function).
  """

  def __init__(self,
               learning_rate,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               use_locking=False,
               name="Ftrl",
               accum_name=None,
               linear_name=None,
               l2_shrinkage_regularization_strength=0.0):
    r"""Construct a new FTRL optimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: A float value, must be less or equal to zero.
      initial_accumulator_value: The starting value for accumulators.
        Only zero or positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Ftrl".
      accum_name: The suffix for the variable that keeps the gradient squared
        accumulator.  If not present, defaults to name.
      linear_name: The suffix for the variable that keeps the linear gradient
        accumulator.  If not present, defaults to name + "_1".
      l2_shrinkage_regularization_strength: A float value, must be greater than
        or equal to zero. This differs from L2 above in that the L2 above is a
        stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
        The FTRL formulation can be written as:
        w_{t+1} = argmin_w(\hat{g}_{1:t}w + L1*||w||_1 + L2*||w||_2^2), where
        \hat{g} = g + (2*L2_shrinkage*w), and g is the gradient of the loss
        function w.r.t. the weights w.
        Specifically, in the absence of L1 regularization, it is equivalent to
        the following update rule:
        w_{t+1} = w_t - lr_t / (1 + 2*L2*lr_t) * g_t -
                  2*L2_shrinkage*lr_t / (1 + 2*L2*lr_t) * w_t
        where lr_t is the learning rate at t.
        When input is sparse shrinkage will only happen on the active weights.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    super(FtrlOptimizer, self).__init__(use_locking, name)

    if initial_accumulator_value < 0.0:
      raise ValueError(
          "initial_accumulator_value %f needs to be be positive or zero" %
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
    if l2_shrinkage_regularization_strength < 0.0:
      raise ValueError(
          "l2_shrinkage_regularization_strength %f needs to be positive"
          " or zero" % l2_shrinkage_regularization_strength)

    self._learning_rate = learning_rate
    self._learning_rate_power = learning_rate_power
    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    self._l2_shrinkage_regularization_strength = (
        l2_shrinkage_regularization_strength)
    self._learning_rate_tensor = None
    self._learning_rate_power_tensor = None
    self._l1_regularization_strength_tensor = None
    self._l2_regularization_strength_tensor = None
    self._l2_shrinkage_regularization_strength_tensor = None
    self._accum_name = accum_name
    self._linear_name = linear_name

  def _create_slots(self, var_list):
    # Create the "accum" and "linear" slots.
    for v in var_list:
      with ops.colocate_with(v):
        val = constant_op.constant(
            self._initial_accumulator_value, dtype=v.dtype, shape=v.get_shape())
        self._get_or_make_slot(v, val, "accum", self._accum_name or self._name)
        self._zeros_slot(v, "linear", self._linear_name or self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(
        self._learning_rate, name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength, name="l1_regularization_strength")
    self._l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength, name="l2_regularization_strength")
    self._l2_shrinkage_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_shrinkage_regularization_strength,
        name="l2_shrinkage_regularization_strength")
    self._learning_rate_power_tensor = ops.convert_to_tensor(
        self._learning_rate_power, name="learning_rate_power")

  def _apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.apply_ftrl(
          var,
          accum,
          linear,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return training_ops.apply_ftrl_v2(
          var,
          accum,
          linear,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.resource_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return training_ops.resource_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.sparse_apply_ftrl(
          var,
          accum,
          linear,
          grad.values,
          grad.indices,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return training_ops.sparse_apply_ftrl_v2(
          var,
          accum,
          linear,
          grad.values,
          grad.indices,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        grad.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.resource_sparse_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
          use_locking=self._use_locking)
    else:
      return training_ops.resource_sparse_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        grad.dtype),
          math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
          use_locking=self._use_locking)
