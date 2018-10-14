# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Variant of the Adam optimizer that handles sparse updates more efficiently.

Compared with the original Adam optimizer, the one in this file can provide a
large improvement in model training throughput for some applications. However,
it provides slightly different semantics than the original Adam algorithm, and
may lead to different empirical results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import adam


class LazyAdamOptimizer(adam.AdamOptimizer):
  """Variant of the Adam optimizer that handles sparse updates more efficiently.

  The original Adam algorithm maintains two moving-average accumulators for
  each trainable variable; the accumulators are updated at every step.
  This class provides lazier handling of gradient updates for sparse variables.
  It only updates moving-average accumulators for sparse variable indices that
  appear in the current batch, rather than updating the accumulators for all
  indices. Compared with the original Adam optimizer, it can provide large
  improvements in model training throughput for some applications. However, it
  provides slightly different semantics than the original Adam algorithm, and
  may lead to different empirical results.
  """

  def _apply_sparse(self, grad, var):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

    # \\(m := beta1 * m + (1 - beta1) * g_t\\)
    m = self.get_slot(var, "m")
    m_t = state_ops.scatter_update(m, grad.indices,
                                   beta1_t * array_ops.gather(m, grad.indices) +
                                   (1 - beta1_t) * grad.values,
                                   use_locking=self._use_locking)

    # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
    v = self.get_slot(var, "v")
    v_t = state_ops.scatter_update(v, grad.indices,
                                   beta2_t * array_ops.gather(v, grad.indices) +
                                   (1 - beta2_t) * math_ops.square(grad.values),
                                   use_locking=self._use_locking)

    # \\(variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
    m_t_slice = array_ops.gather(m_t, grad.indices)
    v_t_slice = array_ops.gather(v_t, grad.indices)
    denominator_slice = math_ops.sqrt(v_t_slice) + epsilon_t
    var_update = state_ops.scatter_sub(var, grad.indices,
                                       lr * m_t_slice / denominator_slice,
                                       use_locking=self._use_locking)
    return control_flow_ops.group(var_update, m_t, v_t)

  def _resource_apply_sparse(self, grad, var, indices):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

    # \\(m := beta1 * m + (1 - beta1) * g_t\\)
    m = self.get_slot(var, "m")
    m_t_slice = beta1_t * array_ops.gather(m, indices) + (1 - beta1_t) * grad
    m_update_op = resource_variable_ops.resource_scatter_update(m.handle,
                                                                indices,
                                                                m_t_slice)

    # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
    v = self.get_slot(var, "v")
    v_t_slice = (beta2_t * array_ops.gather(v, indices) +
                 (1 - beta2_t) * math_ops.square(grad))
    v_update_op = resource_variable_ops.resource_scatter_update(v.handle,
                                                                indices,
                                                                v_t_slice)

    # \\(variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
    var_slice = lr * m_t_slice / (math_ops.sqrt(v_t_slice) + epsilon_t)
    var_update_op = resource_variable_ops.resource_scatter_sub(var.handle,
                                                               indices,
                                                               var_slice)

    return control_flow_ops.group(var_update_op, m_update_op, v_update_op)
