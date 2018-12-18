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

"""Adamax for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Adamax', v1=[])
class Adamax(adam.Adam):
  """Optimizer that implements the Adamax algorithm.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.
  Adamax is sometimes superior to adam, specially in models with embeddings.

  References
    see Section 7 of [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
      ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               name='Adamax',
               **kwargs):
    """Construct a new Adamax optimizer.

    Initialization:

    ```
    m_0 <- 0 (Initialize initial 1st moment vector)
    v_0 <- 0 (Initialize the exponentially weighted infinity norm)
    t <- 0 (Initialize timestep)
    ```

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section 7.1 of the paper:

    ```
    t <- t + 1

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- max(beta2 * v_{t-1}, abs(g))
    variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)
    ```

    Similar to AdamOptimizer, the epsilon is added for numerical stability
    (especially to get rid of division by zero when v_t = 0).

    Contrast to AdamOptimizer, the sparse implementation of this algorithm
    (used when the gradient is an IndexedSlices object, typically because of
    `tf.gather` or an embedding lookup in the forward pass) only updates
    variable slices and corresponding `m_t`, `v_t` terms when that part of
    the variable was used in the forward pass. This means that the sparse
    behavior is contrast to the dense behavior (similar to some momentum
    implementations which ignore momentum unless a variable slice was actually
    used).

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the exponentially weighted infinity norm.
      epsilon: A small constant for numerical stability.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adamax".
      **kwargs: keyword arguments. Allowed to be {`decay`}
    """
    # pylint: disable=useless-super-delegation
    super(Adamax, self).__init__(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        amsgrad=False,
        name=name,
        **kwargs)
    # pylint: enable=useless-super-delegation

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    return training_ops.resource_apply_ada_max(
        var.handle,
        m.handle,
        v.handle,
        beta_1_power,
        lr_t,
        beta_1_t,
        beta_2_t,
        self._get_hyper('epsilon', var_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)

    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    epsilon_t = self._get_hyper('epsilon', var_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_slice = array_ops.gather(m, indices)
    m_t_slice = m_slice * beta_1_t + grad * (1 - beta_1_t)
    with ops.control_dependencies([m_t_slice]):
      m_t = self._resource_scatter_update(m, indices, m_t_slice)

    # u_t = max(beta2 * u, abs(g_t))
    v = self.get_slot(var, 'v')
    v_slice = array_ops.gather(v, indices)
    v_t_slice = math_ops.maximum(v_slice * beta_2_t, math_ops.abs(grad))
    with ops.control_dependencies([v_t_slice]):
      v_t = self._resource_scatter_update(v, indices, v_t_slice)
    # theta_t = theta - lr / (1 - beta1^t) * m_t / u_t
    var_slice = -lr_t / (1 - beta_1_power) * (
        m_t_slice / (v_t_slice + epsilon_t))
    with ops.control_dependencies([var_slice]):
      var_update = self._resource_scatter_add(var, indices, var_slice)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _resource_scatter_update(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_update(
            x.handle, i, v)]):
      return x.value()
