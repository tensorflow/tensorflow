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
"""AddSign oprimizer"""

from __future__ import absolute_import, division, print_function
from tensorflow.python.ops import math_ops, state_ops
from tensorflow.python.training import optimizer

class AddSignOptimizer(optimizer.Optimizer):
  def __init__(self, alpha=1.0, decay=0.9, lr=1e-4,
               use_locking=False, name="PowerSign"):
    """Construct a new AddSignOptimizer.

    https://arxiv.org/abs/1709.07417

    Args:
      alpha: Float. The base of PowerSign update.
      decay: Float. Decay to use to maintain the moving averages
                    of trained variables.
      lr: Float. Learning rate.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "AddSign".
      """
    super(AddSignOptimizer, self).__init__(use_locking, name)
    # self._ema = moving_averages.ExponentialMovingAverage(decay, num_updates=num_updates)
    self._variable_map = None
    self._alpha = alpha
    self._lr = lr
    self._decay = decay
    self._name = "AddSign"

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "m", self._name)

  def _apply_dense(self, grad, var):
    return self._apply_add_sign(grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._apply_add_sign(grad.handle, var.handle)

  def _apply_sparse(self, grad, var):
    return self._apply_add_sign(grad, var)

  def _apply_add_sign(self, grad, var):
    m = self.get_slot(var, "m")
    lr = math_ops.cast(self._lr, var.dtype.base_dtype),
    decay = math_ops.cast(self._decay, var.dtype.base_dtype)
    
    m = (grad - m) * decay + m 
    same_sign = math_ops.sign(m) * math_ops.sign(grad)

    alpha = math_ops.cast(self._alpha, var.dtype.base_dtype),
    delta = (alpha + same_sign) * grad
    var = state_ops.assign_sub(var, delta * lr, use_locking=True)

    return var

