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
"""Adagrad Dual Averaging for TensorFlow."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.AdagradDAOptimizer"])
class AdagradDAOptimizer(optimizer.Optimizer):
  """Adagrad Dual Averaging algorithm for sparse linear models.

  This optimizer takes care of regularization of unseen features in a mini batch
  by updating them when they are seen with a closed form update rule that is
  equivalent to having updated them on every mini-batch.

  AdagradDA is typically used when there is a need for large sparsity in the
  trained model. This optimizer only guarantees sparsity for linear models. Be
  careful when using AdagradDA for deep networks as it will require careful
  initialization of the gradient accumulators for it to train.

  References:
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
      :[Duchi et al., 2011](http://jmlr.org/papers/v12/duchi11a.html)
      ([pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))
  """

  def __init__(self,
               learning_rate,
               global_step,
               initial_gradient_squared_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               use_locking=False,
               name="AdagradDA"):
    """Construct a new AdagradDA optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      global_step: A `Tensor` containing the current training step number.
      initial_gradient_squared_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "AdagradDA".

    Raises:
      ValueError: If the `initial_gradient_squared_accumulator_value` is
      invalid.
    """
    if initial_gradient_squared_accumulator_value <= 0.0:
      raise ValueError("initial_gradient_squared_accumulator_value must be "
                       "positive: %s" %
                       initial_gradient_squared_accumulator_value)
    super(AdagradDAOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._initial_gradient_squared_accumulator_value = (
        initial_gradient_squared_accumulator_value)
    # Created in Initialize.
    self._learning_rate_tensor = None
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    self._global_step = global_step
    self._global_step_on_worker = None

  def _create_slots(self, var_list):
    for v in var_list:
      with ops.colocate_with(v):
        g_val = constant_op.constant(
            0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
        gg_val = constant_op.constant(
            self._initial_gradient_squared_accumulator_value,
            shape=v.get_shape(),
            dtype=v.dtype.base_dtype)
      self._get_or_make_slot(v, g_val, "gradient_accumulator", self._name)
      self._get_or_make_slot(v, gg_val, "gradient_squared_accumulator",
                             self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(
        self._learning_rate, name="learning_rate")
    # Performance optimization so that worker creates a copy of the global step
    # to avoid overloading the parameter server holding the global step.
    with ops.colocate_with(self._learning_rate_tensor):
      self._global_step_on_worker = array_ops.identity(self._global_step) + 1

  def _apply_dense(self, grad, var):
    g_acc = self.get_slot(var, "gradient_accumulator")
    gg_acc = self.get_slot(var, "gradient_squared_accumulator")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    return training_ops.apply_adagrad_da(
        var,
        g_acc,
        gg_acc,
        grad,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._l1_regularization_strength, var.dtype.base_dtype),
        math_ops.cast(self._l2_regularization_strength, var.dtype.base_dtype),
        global_step,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    g_acc = self.get_slot(var, "gradient_accumulator")
    gg_acc = self.get_slot(var, "gradient_squared_accumulator")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    return training_ops.resource_apply_adagrad_da(
        var.handle,
        g_acc.handle,
        gg_acc.handle,
        grad,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        math_ops.cast(self._l1_regularization_strength, grad.dtype.base_dtype),
        math_ops.cast(self._l2_regularization_strength, grad.dtype.base_dtype),
        global_step,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    g_acc = self.get_slot(var, "gradient_accumulator")
    gg_acc = self.get_slot(var, "gradient_squared_accumulator")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    return training_ops.sparse_apply_adagrad_da(
        var,
        g_acc,
        gg_acc,
        grad.values,
        grad.indices,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._l1_regularization_strength, var.dtype.base_dtype),
        math_ops.cast(self._l2_regularization_strength, var.dtype.base_dtype),
        global_step,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    g_acc = self.get_slot(var, "gradient_accumulator")
    gg_acc = self.get_slot(var, "gradient_squared_accumulator")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    return training_ops.resource_sparse_apply_adagrad_da(
        var.handle,
        g_acc.handle,
        gg_acc.handle,
        grad,
        indices,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._l1_regularization_strength, grad.dtype),
        math_ops.cast(self._l2_regularization_strength, grad.dtype),
        global_step,
        use_locking=self._use_locking)
