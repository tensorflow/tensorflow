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

"""Nadam for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.training import adam


class NadamOptimizer(adam.AdamOptimizer):
  """Optimizer that implements the Nadam algorithm.

  See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
  """

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.apply_adam(
        var, m, v,
        math_ops.cast(self._beta1_power, var.dtype.base_dtype),
        math_ops.cast(self._beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad, use_locking=self._use_locking,
        use_nesterov=True).op

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.resource_apply_adam(
        var.handle, m.handle, v.handle,
        math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad, use_locking=self._use_locking,
        use_nesterov=True)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
      # m_bar = (1 - beta1) * g_t + beta1 * m_t
      m_bar = m_scaled_g_values + beta1_t * m_t
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(var,
                                      lr * m_bar / (v_sqrt + epsilon_t),
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_bar, v_t])
