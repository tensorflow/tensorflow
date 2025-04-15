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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.FtrlOptimizer"])
class FtrlOptimizer(optimizer.Optimizer):
  """Optimizer that implements the FTRL algorithm.

  This version has support for both online L2 (McMahan et al., 2013) and
  shrinkage-type L2, which is the addition of an L2 penalty
  to the loss function.

  References:
    Ad-click prediction:
      [McMahan et al., 2013](https://dl.acm.org/citation.cfm?id=2488200)
      ([pdf](https://dl.acm.org/ft_gateway.cfm?id=2488200&ftid=1388399&dwn=1&CFID=32233078&CFTOKEN=d60fe57a294c056a-CB75C374-F915-E7A6-1573FBBC7BF7D526))
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
               l2_shrinkage_regularization_strength=0.0,
               beta=None):
    r"""Construct a new FTRL optimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for
        a fixed learning rate. See section 3.1 in (McMahan et al., 2013).
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
        w_{t+1} = w_t - lr_t / (beta + 2*L2*lr_t) * g_t -
                  2*L2_shrinkage*lr_t / (beta + 2*L2*lr_t) * w_t
        where lr_t is the learning rate at t.
        When input is sparse shrinkage will only happen on the active weights.
      beta: A float value; corresponds to the beta parameter in the paper.

    Raises:
      ValueError: If one of the arguments is invalid.

    References:
      Ad-click prediction:
        [McMahan et al., 2013](https://dl.acm.org/citation.cfm?id=2488200)
        ([pdf](https://dl.acm.org/ft_gateway.cfm?id=2488200&ftid=1388399&dwn=1&CFID=32233078&CFTOKEN=d60fe57a294c056a-CB75C374-F915-E7A6-1573FBBC7BF7D526))
    """
    super(FtrlOptimizer, self).__init__(use_locking, name)

    if initial_accumulator_value < 0.0:
      raise ValueError(
          "initial_accumulator_value %f needs to be positive or zero" %
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
    self._beta = (0.0 if beta is None else beta)
    self._l2_shrinkage_regularization_strength = (
        l2_shrinkage_regularization_strength)
    self._learning_rate_tensor = None
    self._learning_rate_power_tensor = None
    self._l1_regularization_strength_tensor = None
    self._adjusted_l2_regularization_strength_tensor = None
    self._l2_shrinkage_regularization_strength_tensor = None
    self._accum_name = accum_name
    self._linear_name = linear_name

  def _create_slots(self, var_list):
    # Create the "accum" and "linear" slots.
    def _accum_initializer(shape, dtype=dtypes.float32, partition_info=None):
      del partition_info
      return array_ops.ones(
          shape=shape, dtype=dtype) * self._initial_accumulator_value
    for v in var_list:
      self._get_or_make_slot_with_initializer(
          v, _accum_initializer, v.shape, v.dtype, "accum",
          self._accum_name or self._name)
      self._zeros_slot(v, "linear", self._linear_name or self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(
        self._learning_rate, name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength, name="l1_regularization_strength")
    # L2 regularization strength with beta added in so that the underlying
    # TensorFlow ops do not need to include that parameter.
    self._adjusted_l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength + self._beta /
        (2. * math_ops.maximum(self._learning_rate, 1e-36)),
        name="adjusted_l2_regularization_strength")
    assert self._adjusted_l2_regularization_strength_tensor is not None
    self._beta_tensor = ops.convert_to_tensor(self._beta, name="beta")
    self._l2_shrinkage_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_shrinkage_regularization_strength,
        name="l2_shrinkage_regularization_strength")
    self._learning_rate_power_tensor = ops.convert_to_tensor(
        self._learning_rate_power, name="learning_rate_power")

  def _apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.apply_ftrl(
          var,
          accum,
          linear,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.apply_ftrl_v2(
          var,
          accum,
          linear,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.resource_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.resource_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.sparse_apply_ftrl(
          var,
          accum,
          linear,
          grad.values,
          grad.indices,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.sparse_apply_ftrl_v2(
          var,
          accum,
          linear,
          grad.values,
          grad.indices,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._l1_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        var.dtype.base_dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        grad.dtype.base_dtype),
          math_ops.cast(self._learning_rate_power_tensor, var.dtype.base_dtype),
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.resource_sparse_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        grad.dtype),
          math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
          use_locking=self._use_locking)
    else:
      return gen_training_ops.resource_sparse_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                        grad.dtype),
          math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                        grad.dtype),
          math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
          use_locking=self._use_locking)
