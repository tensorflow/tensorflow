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

"""Adam optimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops


class AdamOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the Adam algorithm.

  See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
  """

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam"):
    """Construct a new Adam optimizer.

    Initialization:

    $$m_0 \Leftarrow 0 (Initialize initial 1st moment vector)$$
    $$v_0 \Leftarrow 0 (Initialize initial 2nd moment vector)$$
    $$t \Leftarrow 0 (Initialize timestep)$$

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    $$t \Leftarrow t + 1$$
    $$lr_t \Leftarrow \text{learning_rate} * \sqrt{(1 - beta_2^t) / (1 - beta_1^t)}$$

    $$m_t \Leftarrow beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t \Leftarrow beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable \Leftarrow variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

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

    Some of the args below are hyperparameters where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate.
      beta1: A float hyperparameter. The exponential decay rate for the 1st
        moment estimates.
      beta2: A float hyperparameter. The exponential decay rate for the 2nd
        moment estimates.
      epsilon: A float hyperparameter. This epsilon is "epsilon hat" in the
        Kingma and Ba paper (in the formula just before Section 2.1), not the
        epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
    """
    super(AdamOptimizer, self).__init__(use_locking, name)

    self._set_hyper("learning_rate", learning_rate)
    self._set_hyper("beta1", beta1)
    self._set_hyper("beta2", beta2)
    self._set_hyper("epsilon", epsilon)

  def _get_beta_accumulators(self, state=None):
    if state is None:
      state = self._get_per_graph_state()
    return (state.get_non_slot("beta1_power"),
            state.get_non_slot("beta2_power"))

  def _create_vars(self, var_list, state):
    # Non-slot variables end up on the same device(s).
    state.create_non_slot(initial_value=state.get_hyper("beta1"),
                          name="beta1_power")
    state.create_non_slot(initial_value=state.get_hyper("beta2"),
                          name="beta2_power")

    # Create slots for the first and second moments.
    for v in var_list:
      state.zeros_slot(v, "m")
      state.zeros_slot(v, "v")

  def _apply_dense(self, grad, var, state):
    m = state.get_slot(var, "m")
    v = state.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators(state)
    return training_ops.apply_adam(
        var, m, v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("beta1", var.dtype.base_dtype),
        state.get_hyper("beta2", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad, use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var, state):
    m = state.get_slot(var, "m")
    v = state.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators(state)
    return training_ops.resource_apply_adam(
        var.handle, m.handle, v.handle,
        math_ops.cast(beta1_power, grad.dtype.base_dtype),
        math_ops.cast(beta2_power, grad.dtype.base_dtype),
        state.get_hyper("learning_rate", grad.dtype.base_dtype),
        state.get_hyper("beta1", grad.dtype.base_dtype),
        state.get_hyper("beta2", grad.dtype.base_dtype),
        state.get_hyper("epsilon", grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add, state):
    beta1_power, beta2_power = self._get_beta_accumulators(state)
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = state.get_hyper("learning_rate", var.dtype.base_dtype)
    beta1_t = state.get_hyper("beta1", var.dtype.base_dtype)
    beta2_t = state.get_hyper("beta2", var.dtype.base_dtype)
    epsilon_t = state.get_hyper("epsilon", var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = state.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = state.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(var,
                                      lr * m_t / (v_sqrt + epsilon_t),
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var, state):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x, i, v, use_locking=self._use_locking),
        state)

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(
            x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices, state):
    return self._apply_sparse_shared(
        grad, var, indices, self._resource_scatter_add, state)

  def _finish(self, state):
    # Update the power accumulators.
    beta1_power, beta2_power = self._get_beta_accumulators(state)
    update_beta1 = beta1_power.assign(
        beta1_power * state.get_hyper("beta1"),
        use_locking=self._use_locking)
    update_beta2 = beta2_power.assign(
        beta2_power * state.get_hyper("beta2"),
        use_locking=self._use_locking)
    return control_flow_ops.group(update_beta1, update_beta2)
