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
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Nadam')
class Nadam(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the NAdam algorithm.

  Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
  Nesterov momentum.

  Initialization:

  $$m_0 := 0 \text{(Initialize 1st moment vector)}$$
  $$v_0 := 0 \text{(Initialize 2nd moment vector)}$$
  $$mu_0 := 1$$
  $$t := 0 \text{(Initialize timestep)}$$

  Computes:
  $$t := t + 1$$
  $$\mu_t := \beta_1 * (1 - 0.5 * 0.96^{0.004 * t})$$
  $$g' := g / (1 - \prod_{i=1}^{t}{\mu_i})$$
  $$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
  $$m' := m_t / (1 - \prod_{i=1}^{t+1}{\mu_i})$$
  $$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g * g$$
  $$v' := v_t / (1 - \beta_2^t)$$
  $$\bar{m} := (1 - \mu_t) * g' + \mu_{t+1} * m'$$
  $$\theta_t := \theta_{t-1} - lr * \bar{m} / (\sqrt{v'} + \epsilon)$$

  gradient is evaluated at theta(t) + momentum * v(t), and the variables always
  store theta + beta_1 * m / sqrt(v) instead of theta.

  References
    See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               name='Nadam',
               **kwargs):
    """Construct a new Nadam optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the exponentially weighted infinity norm.
      epsilon: A small constant for numerical stability.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adamax".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """

    # Backwards compatiblity with keras NAdam optimizer.
    kwargs['decay'] = kwargs.pop('schedule_decay', 0.004)
    if epsilon is None:
      epsilon = backend_config.epsilon()
    super(Nadam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('epsilon', epsilon)
    self._m_cache = None

  def _create_slots(self, var_list):
    var_dtype = var_list[0].dtype.base_dtype
    if self._m_cache is None:
      self._m_cache = self.add_weight(
          'momentum_cache',
          shape=[],
          dtype=var_dtype,
          initializer='ones',
          trainable=False)
      self._weights.append(self._m_cache)
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      # Create slots for the first moments.
      self.add_slot(var, 'm')
    for var in var_list:
      # Create slots for the second moments.
      self.add_slot(var, 'v')

  def _prepare(self, var_list):
    var_dtype = var_list[0].dtype.base_dtype
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    decay_base = math_ops.cast(0.96, var_dtype)
    self.m_cache_t = beta_1_t * (
        1. - 0.5 * (math_ops.pow(decay_base, self._initial_decay * local_step)))
    self.m_cache_t_1 = beta_1_t * (
        1. - 0.5 *
        (math_ops.pow(decay_base, self._initial_decay * (local_step + 1))))
    m_schedule_new = self._m_cache * self.m_cache_t
    self.m_schedule_new = state_ops.assign(
        self._m_cache, m_schedule_new, use_locking=self._use_locking)
    self.m_schedule_next = self.m_schedule_new * self.m_cache_t_1

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    lr_t = self._get_hyper('learning_rate', var_dtype)
    epsilon_t = self._get_hyper('epsilon', var_dtype)
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)

    g_prime = grad / (1. - self.m_schedule_new)
    m_t = beta_1_t * m + (1 - beta_1_t) * grad
    m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
    m_t_prime = m_t / (1. - self.m_schedule_next)
    v_t = beta_2_t * v + (1 - beta_2_t) * math_ops.square(grad)
    v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)
    v_t_prime = v_t / (1. - math_ops.pow(beta_2_t, local_step))
    m_t_bar = (1. - self.m_cache_t) * g_prime + self.m_cache_t_1 * m_t_prime
    var_t = var - lr_t * m_t_bar / (math_ops.sqrt(v_t_prime) + epsilon_t)
    return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices):
    var_dtype = var.dtype.base_dtype
    lr_t = self._get_hyper('learning_rate', var_dtype)
    epsilon_t = self._get_hyper('epsilon', var_dtype)
    v = self.get_slot(var, 'v')
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)

    g_prime = grad / (1. - self.m_schedule_new)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * (1 - beta_1_t)
    m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
      m_t_slice = array_ops.gather(m_t, indices)

    m_t_prime = m_t_slice / (1. - self.m_schedule_next)
    m_t_bar = (1. - self.m_cache_t) * g_prime + self.m_cache_t_1 * m_t_prime

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
      v_t_slice = array_ops.gather(v_t, indices)

    v_t_prime = v_t_slice / (1. - math_ops.pow(beta_2_t, local_step))
    v_prime_sqrt = math_ops.sqrt(v_t_prime)

    var_update = self._resource_scatter_add(
        var, indices, -lr_t * m_t_bar / (v_prime_sqrt + epsilon_t))
    return control_flow_ops.group(*[var_update, m_t_bar, v_t])

  def get_config(self):
    config = super(Nadam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self._serialize_hyperparameter('epsilon'),
    })
    return config
