# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops


class Nadam(adam.Adam):
  r"""Optimizer that implements the NAdam algorithm.

  Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
  Nesterov momentum.

  Initialization:

  $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
  $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
  $$t := 0 \text{(Initialize timestep)}$$

  Computes:
  $$t := t + 1$$
  $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
  $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
  $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
  $$m_bar_t := beta_1 * v_t + (1 - beta_1) * g$$
  $$theta_t := theta_{t-1} - lr_t * m_bar_t / (\sqrt{v_t} + \epsilon)$$

  gradient is evaluated at theta(t) + momentum * v(t), and the variables always
  store theta + beta_1 * m / sqrt(v) instead of theta.

  References
    See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
  """

  def _resource_apply_dense(self, grad, var):
    grad_dtype = grad.dtype.base_dtype
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    local_step = math_ops.cast(self.iterations + 1, grad_dtype)
    beta_1_t = math_ops.cast(self._get_hyper('beta_1'), grad_dtype)
    beta_2_t = math_ops.cast(self._get_hyper('beta_2'), grad_dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    return training_ops.resource_apply_adam(
        var.handle,
        m.handle,
        v.handle,
        beta_1_power,
        beta_2_power,
        math_ops.cast(self._get_hyper('learning_rate'), grad_dtype),
        beta_1_t,
        beta_2_t,
        math_ops.cast(self._get_hyper('epsilon'), grad_dtype),
        grad,
        use_locking=self._use_locking,
        use_nesterov=True)

  def _resource_apply_sparse(self, grad, var, indices):
    var_dtype = var.dtype.base_dtype
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = math_ops.cast(self._get_hyper('beta_1'), var_dtype)
    beta_2_t = math_ops.cast(self._get_hyper('beta_2'), var_dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr_t = math_ops.cast(self._get_hyper('learning_rate'), var_dtype)
    epsilon_t = math_ops.cast(self._get_hyper('epsilon'), var_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * (1 - beta_1_t)
    m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
      # m_bar = (1 - beta1) * g_t + beta1 * m_t
      m_bar = m_scaled_g_values + beta_1_t * array_ops.gather(m_t, indices)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    v_t_slice = array_ops.gather(v_t, indices)
    v_sqrt = math_ops.sqrt(v_t_slice)
    var_update = self._resource_scatter_add(var, indices,
                                            -lr * m_bar / (v_sqrt + epsilon_t))
    return control_flow_ops.group(*[var_update, m_bar, v_t])
