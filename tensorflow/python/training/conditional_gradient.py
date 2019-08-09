# Copyright 2018 Vishnu sai rao suresh Lokhande & Pengyu Kan. All Rights Reserved.
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

"""Conditional Gradient method for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
#from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops

from tensorflow.python.ops import resource_variable_ops
#from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops



@tf_export(v1=["train.ConditionalGradientOptimizer"])
class ConditionalGradientOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Conditional Gradient optimization.
  Helps handle constraints well.
  Currently only supports frobenius norm constraint.
  See https://arxiv.org/pdf/1803.06453.pdf
  ```
  variable -= (1-learning_rate)
    * (variable + lamda * gradient / frobenius_norm(gradient))
  ```
  """

  def __init__(self, learning_rate, lamda,
               use_locking=False, name="ConditionalGradient"):
    """Construct a conditional gradient optimizer.
        Args:
        learning_rate: A `Tensor` or a floating point value.  The learning rate.
        lamda: A `Tensor` or a floating point value.  The constraint.
        use_locking: If `True` use locks for update operations.
        name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "ConditionalGradient"
    """
    super(ConditionalGradientOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._lamda = lamda

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "conditional_gradient", self._name)

  def _prepare(self):
    learning_rate = self._learning_rate
    if callable(learning_rate):
      learning_rate = learning_rate()
    self._learning_rate_tensor = ops.convert_to_tensor(learning_rate,
                                                       name="learning_rate")
    lamda = self._lamda
    if callable(lamda):
      lamda = lamda()
    self._lamda_tensor = ops.convert_to_tensor(lamda, name="lamda")

  def _apply_dense(self, grad, var):
    def frobenius_norm(m):
      return math_ops.reduce_sum(m ** 2) ** 0.5
    norm = ops.convert_to_tensor(frobenius_norm(grad), name="norm")
    norm = math_ops.cast(norm, var.dtype.base_dtype)
    lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    lamda = math_ops.cast(self._lamda_tensor, var.dtype.base_dtype)
    var_update = state_ops.assign(var, math_ops.multiply(var, lr)  \
                - (1-lr)* lamda * grad / norm, use_locking=self._use_locking)
    return control_flow_ops.group(var_update)

  def _resource_apply_dense(self, grad, var):
    def frobenius_norm(m):
      return math_ops.reduce_sum(m ** 2) ** 0.5
    norm = ops.convert_to_tensor(frobenius_norm(grad), name="norm")
    norm = math_ops.cast(norm, var.dtype.base_dtype)
    lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    lamda = math_ops.cast(self._lamda_tensor, var.dtype.base_dtype)
    var_update_tensor = math_ops.multiply(var, lr) - (1-lr)* lamda * grad / norm
    var_update_op = resource_variable_ops.assign_variable_op(var.handle,
                                                             var_update_tensor)
    return control_flow_ops.group(var_update_op)

  def _apply_sparse(self, grad, var):
    def frobenius_norm(m):
      return math_ops.reduce_sum(m ** 2) ** 0.5
    norm = ops.convert_to_tensor(frobenius_norm(grad.value), name="norm")
    norm = math_ops.cast(norm, var.dtype.base_dtype)
    lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    lamda = math_ops.cast(self._lamda_tensor, var.dtype.base_dtype)
    var_slice = array_ops.gather(var, grad.indices)
    var_update_value = math_ops.multiply(var_slice, lr)  \
                                - (1-lr)* lamda * grad.value / norm
    var_update = state_ops.scatter_update(var, grad.indices,  \
                                var_update_value, use_locking=self._use_locking)
    return control_flow_ops.group(var_update)

  def _resource_apply_sparse(self, grad, var, indices):
    def frobenius_norm(m):
      return math_ops.reduce_sum(m ** 2) ** 0.5
    norm = ops.convert_to_tensor(frobenius_norm(grad), name="norm")
    norm = math_ops.cast(norm, var.dtype.base_dtype)
    lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    lamda = math_ops.cast(self._lamda_tensor, var.dtype.base_dtype)
    var_slice = array_ops.gather(var, indices)
    var_update_value = math_ops.multiply(var_slice, lr)  \
                                - (1-lr) * lamda * grad / norm
    var_update_op = resource_variable_ops.resource_scatter_update \
                                (var.handle, indices, var_update_value)
    return control_flow_ops.group(var_update_op)
