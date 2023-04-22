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

"""Adadelta for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.AdadeltaOptimizer"])
class AdadeltaOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adadelta algorithm.

  References:
    ADADELTA - An Adaptive Learning Rate Method:
      [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
      ([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))
  """

  def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-8,
               use_locking=False, name="Adadelta"):
    """Construct a new Adadelta optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value. The learning rate.
        To match the exact form in the original paper use 1.0.
      rho: A `Tensor` or a floating point value. The decay rate.
      epsilon: A `Tensor` or a floating point value.  A constant epsilon used
               to better conditioning the grad update.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adadelta".

    @compatibility(eager)
    When eager execution is enabled, `learning_rate`, `rho`, and `epsilon` can
    each be a callable that takes no arguments and returns the actual value to
    use. This can be useful for changing these values across different
    invocations of optimizer functions.
    @end_compatibility
    """
    super(AdadeltaOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._rho = rho
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._rho_t = None
    self._epsilon_t = None

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "accum", self._name)
      self._zeros_slot(v, "accum_update", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    rho = self._call_if_callable(self._rho)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="lr")
    self._rho_t = ops.convert_to_tensor(rho, name="rho")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    accum_update = self.get_slot(var, "accum_update")
    return training_ops.apply_adadelta(
        var,
        accum,
        accum_update,
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._rho_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    accum = self.get_slot(var, "accum")
    accum_update = self.get_slot(var, "accum_update")
    return training_ops.resource_apply_adadelta(
        var.handle,
        accum.handle,
        accum_update.handle,
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._rho_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    accum = self.get_slot(var, "accum")
    accum_update = self.get_slot(var, "accum_update")
    return training_ops.sparse_apply_adadelta(
        var,
        accum,
        accum_update,
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._rho_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    accum = self.get_slot(var, "accum")
    accum_update = self.get_slot(var, "accum_update")
    return training_ops.resource_sparse_apply_adadelta(
        var.handle,
        accum.handle,
        accum_update.handle,
        math_ops.cast(self._lr_t, grad.dtype),
        math_ops.cast(self._rho_t, grad.dtype),
        math_ops.cast(self._epsilon_t, grad.dtype),
        grad,
        indices,
        use_locking=self._use_locking)
