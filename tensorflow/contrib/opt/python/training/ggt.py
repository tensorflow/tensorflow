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
"""GGT for Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class GGTOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the GGT algorithm.

  GGT has an advantage over sgd and adam on large models with poor conditioning,
  for example language models and CNNs,
  see [[ABCHSZZ 2018]](https://arxiv.org/pdf/1806.02958.pdf).
  """

  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               use_locking=False,
               name="GGT",
               window=10,
               eps=1e-4,
               svd_eps=1e-6,
               sigma_eps=1e-2):
    """Construct a new GGT optimizer.

    Initialization:

    ```
    t <- 0 (Initialize timestep)
    grad_buffer <- 0 (Initialize buffer for keeping past gradients)
    flat_grad <- 0 (Initialize flattened gradient that contains gradients of all
                    variables)
    m_0 <- 0 (Initialize 1st moment vector)
    ```

    Suppose all variables and their gradients are concatenated into vectors
    `flat_vars` and `flat_grad`. The update rule for `flat_vars`
    uses an optimization described at the beginning of section 2 of the paper:

    ```
    t <- t + 1

    m_t <- beta1 * m_{t-1} + (1 - beta1) * flat_grad
    grad_buffer[(t-1) % window, :] <- m_t

    M <- grad_buffer^T / sqrt(min(t, window))
    U, sigma, _ <- SVD(M^TM + I * svd_eps)

    sigma_sqrt_inv <- (sqrt(sigma) + sigma_eps)^(-3)
    sigma_sqrt_min <- min(sqrt(sigma))

    if sigma_sqrt_min > eps:
      new_step <- M U diag(sigma_sqrt_inv) U^T M^T m_t +
                  (m_t - M U diag(1/sigma) U^T M^T m_t) / sigma_sqrt_min
    else:
      new_step <- M U diag(sigma_sqrt_inv) U^T M^T m_t

    flat_vars <- flat_vars - learning_rate * new_step
    ```

    GGT provides the power of full-matrix adaptive regularization at a cost not
    much larger than SGD. As a result it is suited for large models where the
    gradient covariance matrix has a poor condition number that slows down first
    order methods.
    GGT uses the preconditioner from full-matrix AdaGrad, with gradient history
    attenuated exponentially as in Adam, and truncated to a window parameter.
    It has provable guarantees even for non-convex optimization that is never
    significantly worse than SGD and in some cases better.

    Args:
      learning_rate: A float hyperparameter. The learning rate.
      beta1: A float hyperparameter. The exponential decay rate for the 1st
        moment estimates.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "GGT".
      window: An integer hyperparameter. The number of first moments to keep in
        computing the adaptive preconditioner.
      eps: A float hyperparameter. Used to truncate small eigenvalues of the
        gradient covariance matrix.
      svd_eps: A float hyperparameter. Used to stabilize SVD.
      sigma_eps: A float hyperparameter. Used to regularize matrix inversion.
    """
    super(GGTOptimizer, self).__init__(use_locking, name)
    self._set_hyper("lr", learning_rate)
    self._set_hyper("beta1", beta1)
    self._set_hyper("window", window)
    self._set_hyper("eps", eps)
    self._set_hyper("svd_eps", svd_eps)
    self._set_hyper("sigma_eps", sigma_eps)

    self.index_dict = {}
    self.shape_dict = {}

  def _create_vars(self, var_list, state):
    # Construct ordered dictionary for variable dimensions, sorted by name.
    shape_dict = {}
    for v in var_list:
      shape_dict[v.name] = tensor_shape.dimension_value(np.prod(v.get_shape()))
    self.shape_dict = collections.OrderedDict(
        sorted(shape_dict.items(), key=lambda t: t[0]))

    # Assign each variable its location in flat_grad. The locations are based on
    # the order of sorted names.
    idx = 0
    for v_name, v_dim in self.shape_dict.items():
      self.index_dict[v_name] = idx
      idx += v_dim

    state.create_non_slot(
        initial_value=math_ops.cast(0., dtype=var_list[0].dtype.base_dtype),
        name="global_step")

    # Buffer for keeping past gradients.
    window = state.get_hyper("window")
    grad_buffer_init = array_ops.zeros(
        [window, idx], dtype=var_list[0].dtype.base_dtype)
    state.create_non_slot(initial_value=grad_buffer_init, name="grad_buffer")

    state.create_non_slot(
        initial_value=array_ops.zeros(
            (idx,), dtype=var_list[0].dtype.base_dtype),
        name="moment1")

    # Flattened gradient that contains gradients for all variables in the model.
    state.create_non_slot(
        initial_value=array_ops.zeros(
            (idx,), dtype=var_list[0].dtype.base_dtype),
        name="flat_grad")

  def _get_global_step(self, state=None):
    if state is None:
      state = self._get_per_graph_state()
    return state.get_non_slot("global_step")

  def _get_moment1(self, state=None):
    if state is None:
      state = self._get_per_graph_state()
    return state.get_non_slot("moment1")

  def _get_grad_buffer(self, state=None):
    if state is None:
      state = self._get_per_graph_state()
    return state.get_non_slot("grad_buffer")

  def _get_flat_grad(self, state=None):
    if state is None:
      state = self._get_per_graph_state()
    return state.get_non_slot("flat_grad")

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

  def _prepare(self, state):
    self._variables = []

  def _apply_dense(self, grad, var, state):
    self._variables.append(var)
    dim = self.shape_dict[var.name]
    start_index = self.index_dict[var.name]
    end_index = start_index + dim

    # Update flat_gradient at the index associated with the variable.
    flat_grad = self._get_flat_grad(state)
    new_flat_grad = array_ops.reshape(grad, [-1])
    flat_grad_updated = state_ops.scatter_update(
        flat_grad, math_ops.range(start_index, end_index), new_flat_grad)

    return flat_grad_updated

  def _resource_apply_dense(self, grad, var, state):
    self._variables.append(var)
    dim = self.shape_dict[var.name]
    start_index = self.index_dict[var.name]
    end_index = start_index + dim

    # Update flat_gradient at the index associated with the variable.
    flat_grad = self._get_flat_grad(state)
    new_flat_grad = array_ops.reshape(grad, [-1])
    flat_grad_updated = state_ops.scatter_update(
        flat_grad, math_ops.range(start_index, end_index), new_flat_grad)

    return flat_grad_updated

  def _finish(self, state):
    var_dtype = self._variables[0].dtype.base_dtype
    # Update global step.
    global_step = self._get_global_step(state)
    update_global_step = state_ops.assign_add(global_step, 1.)

    # Update the first moment estimate.
    beta1 = state.get_hyper("beta1", dtype=var_dtype)
    moment1 = self._get_moment1(state)
    flat_grad = self._get_flat_grad(state)
    # moment1_t := beta1 * moment1_{t-1} + (1 - beta1) * flat_grad_t
    update_moment1 = moment1.assign(beta1 * moment1 + (1. - beta1) * flat_grad)

    # Update the gradient buffer.
    window = state.get_hyper("window")
    grad_buffer = self._get_grad_buffer(state)
    next_grad_index = math_ops.floormod(
        math_ops.to_int32(update_global_step - 1.), window)
    # grad_buffer[(t-1) % window] := moment1_t
    update_grad_buffer = state_ops.scatter_update(grad_buffer, next_grad_index,
                                                  update_moment1)

    # Compute the update step.
    eps = state.get_hyper("eps", dtype=var_dtype)
    svd_eps = state.get_hyper("svd_eps", dtype=var_dtype)
    sigma_eps = state.get_hyper("sigma_eps", dtype=var_dtype)
    lr = state.get_hyper("lr", dtype=var_dtype)
    denom = math_ops.sqrt(
        math_ops.minimum(
            ops.convert_to_tensor(update_global_step),
            ops.convert_to_tensor(math_ops.cast(window, dtype=var_dtype))))
    moment1_2d = array_ops.expand_dims(update_moment1, -1)

    # m = grad_buffer^T / sqrt(min(t, window))
    # m has shape [model dimension, window], where model dimension is the sum
    # of the dimensions of the flattened variables.
    m = array_ops.transpose(math_ops.divide(update_grad_buffer, denom))

    # sigma, u, _ = SVD(m^Tm + I * svd_eps)
    mm = math_ops.matmul(m, m, transpose_a=True)
    damping = math_ops.cast(linalg_ops.eye(window), dtype=var_dtype) * svd_eps
    sigma, u, _ = linalg_ops.svd(mm + damping)
    sigma_sqrt = math_ops.sqrt(sigma)
    sigma_sqrt_min = math_ops.reduce_min(sigma_sqrt)

    # sigma_sqrt_inv = 1 / (\sqrt{sigma} + sigma_eps) ^ 3
    # We add sigma_eps to alleviate numerical instability.
    # Note that (m^Tm)^(-3/2) = u diag(sigma_sqrt_inv) u^T.
    sigma_sqrt_inv = math_ops.divide(
        math_ops.cast(1.0, dtype=var_dtype),
        math_ops.pow(sigma_sqrt + sigma_eps, 3))

    # In full matrix AdaGrad, the update step computes (mm^T)^(-1/2)g, where the
    # inversion of a model dimension by model dimension matrix is needed. To
    # speed up this computation we calculate the following instead:
    # m(m^Tm)^(-3/2)m^T moment1 = m u diag(sigma_sqrt_inv) u^T m^T moment1.
    new_step = array_ops.expand_dims(
        array_ops.zeros(flat_grad.get_shape(), dtype=var_dtype), -1)
    head = math_ops.matmul(
        m,
        math_ops.matmul(
            u,
            math_ops.matmul(
                array_ops.diag(sigma_sqrt_inv),
                math_ops.matmul(
                    u,
                    math_ops.matmul(m, moment1_2d, transpose_a=True),
                    transpose_a=True))))

    # When inverting (mm^t)^(1/2), we also add epsilon * I regularization for
    # degenerate cases. We expand ((mm^t)^(1/2) + epsilon * I)^(-1) using
    # Woodbury's identity.
    # For full derivation please see paper at
    # https://arxiv.org/pdf/1806.02958.pdf
    tail = moment1_2d - math_ops.matmul(
        m,
        math_ops.matmul(
            u,
            math_ops.matmul(
                array_ops.diag(
                    math_ops.divide(math_ops.cast(1.0, dtype=var_dtype),
                                    sigma)),
                math_ops.matmul(
                    u,
                    math_ops.matmul(m, moment1_2d, transpose_a=True),
                    transpose_a=True))))
    scaled_tail = math_ops.divide(tail, sigma_sqrt_min)

    update_new_step = control_flow_ops.cond(
        sigma_sqrt_min > eps, lambda: math_ops.add(head, scaled_tail),
        lambda: math_ops.add(new_step, head))

    # Update each variable.
    update_step = []
    for var in self._variables:
      dim = self.shape_dict[var.name]
      start_index = self.index_dict[var.name]
      end_index = start_index + dim
      var_update_correct_shape = array_ops.reshape(
          update_new_step[start_index:end_index], var.get_shape())
      var_updated = state_ops.assign_sub(var, lr * var_update_correct_shape)
      update_step.append(var_updated)

    return control_flow_ops.group(update_step)
