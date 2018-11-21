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
"""Adam for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops


class Adam(optimizer_v2.OptimizerV2):
  """Optimizer that implements the Adam algorithm.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments. According to the
  reference, the method is 'computationally efficient, has little memory
  requirement, invariant to diagonal rescaling of gradients, and is well suited
  for problems that are large in terms of data/parameters'.

  # References
      See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
        ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               name='Adam'):
    r"""Construct a new Adam optimizer.

    Initialization:

    $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
    $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
    $$t := 0 \text{(Initialize timestep)}$$

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    $$t := t + 1$$
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$

    $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".  @compatibility(eager) When eager execution is
        enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each be
        a callable that takes no arguments and returns the actual value to use.
        This can be useful for changing these values across different
        invocations of optimizer functions. @end_compatibility
    """

    super(Adam, self).__init__(name)
    self._set_hyper('learning_rate', learning_rate)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('epsilon', epsilon)

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for var in var_list:
      self.add_slot(var, 'm')
      self.add_slot(var, 'v')

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
        use_locking=self._use_locking)

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

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(
        var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def get_config(self):
    config = super(Adam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self._serialize_hyperparameter('epsilon'),
    })
    return config
