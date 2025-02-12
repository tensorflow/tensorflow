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

"""ProximalAdagrad for TensorFlow."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.ProximalAdagradOptimizer"])
class ProximalAdagradOptimizer(optimizer.Optimizer):
  # pylint: disable=line-too-long
  """Optimizer that implements the Proximal Adagrad algorithm.

  References:
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:
      [Duchi et al., 2011](http://jmlr.org/papers/v12/duchi11a.html)
      ([pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))
    Efficient Learning using Forward-Backward Splitting:
      [Duchi et al., 2009](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting)
      ([pdf](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf))
  """

  def __init__(self, learning_rate, initial_accumulator_value=0.1,
               l1_regularization_strength=0.0, l2_regularization_strength=0.0,
               use_locking=False, name="ProximalAdagrad"):
    """Construct a new ProximalAdagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.
    """
    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    super(ProximalAdagradOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    # Created in Initialize.
    self._l1_regularization_strength_tensor = None
    self._l2_regularization_strength_tensor = None
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      with ops.colocate_with(v):
        val = constant_op.constant(self._initial_accumulator_value,
                                   shape=v.get_shape(),
                                   dtype=v.dtype.base_dtype)
      self._get_or_make_slot(v, val, "accumulator", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength,
        name="l1_regularization_strength")
    self._l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength,
        name="l2_regularization_strength")

  def _apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.apply_proximal_adagrad(
        var, acc, self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad, use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.resource_apply_proximal_adagrad(
        var.handle, acc.handle, self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad, use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.sparse_apply_proximal_adagrad(
        var, acc, self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad.values, grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    acc = self.get_slot(var, "accumulator")
    return gen_training_ops.resource_sparse_apply_proximal_adagrad(
        var.handle, acc.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
        grad, indices,
        use_locking=self._use_locking)
