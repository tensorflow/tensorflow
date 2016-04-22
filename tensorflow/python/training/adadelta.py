# Copyright 2015 Google Inc. All Rights Reserved.
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
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class AdadeltaOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adadelta algorithm. 

  See [M. D. Zeiler](http://arxiv.org/abs/1212.5701)
  ([pdf](http://arxiv.org/pdf/1212.5701.pdf))
 
  @@__init__
  """

  def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-8,
               use_locking=False, name="Adadelta"):
    """Construct a new Adadelta optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value. The learning rate.
      rho: A `Tensor` or a floating point value. The decay rate.
      epsilon: A `Tensor` or a floating point value.  A constant epsilon used
               to better conditioning the grad update.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adadelta".
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
    self._lr_t = ops.convert_to_tensor(self._lr, name="lr")
    self._rho_t = ops.convert_to_tensor(self._rho, name="rho")
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

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
