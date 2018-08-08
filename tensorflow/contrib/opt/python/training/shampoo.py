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

"""The Shampoo Optimizer.

Variant of Adagrad using one preconditioner matrix per variable dimension.
For details, see https://arxiv.org/abs/1802.09568
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import optimizer


def GetParam(var, timestep):
  if callable(var):
    return var(timestep)
  else:
    return var


class ShampooOptimizer(optimizer.Optimizer):
  """The Shampoo Optimizer

  Variant of Adagrad using one preconditioner matrix per variable dimension.
  For details, see https://arxiv.org/abs/1802.09568

  gbar is time-weighted accumulated gradient:
  gbar[t] = gbar_decay[t] * gbar[t-1] + gbar_weight[t] * g[t]

  mat_gbar is time-weighted accumulated gradient square:
  mat_gbar_j[t] = mat_gbar_decay[t] * mat_gbar_j[t-1]
                  + mat_gbar_weight[t] * gg_j[t]
  where if g[t] = g_abcd then gg_a[t] = g_abcd g_a'bcd (Einstein notation)

  Update rule:
  w[t+1] = w[t] - learning_rate[t] * Prod_j mat_gbar_j[t]^(-alpha/n) gbar[t]
     Again, mat_gbar_j[t]^(-alpha) gbar[t] is a tensor contraction along the
     j'th dimension of gbar[t] with the first dimension of
     mat_gbar_j[t]^(-alpha/n), where alpha is a hyperparameter,
     and n = rank of the variable.
     Prod_j represents doing this contraction for all j in 0..n-1.

  Typically learning_rate is constant, but could be time dependent by passing
  a lambda function that depends on step.
  """

  def __init__(self,
               global_step=0,
               max_matrix_size=768,
               gbar_decay=0.0,
               gbar_weight=1.0,
               mat_gbar_decay=1.0,
               mat_gbar_weight=1.0,
               learning_rate=1.0,
               svd_interval=1,
               precond_update_interval=1,
               epsilon=0.1,
               alpha=0.5,
               use_iterative_root=False,
               use_locking=False,
               name="Shampoo"):
    """Default values of the various hyper-parameters.

    gbar_decay, gbar_weight etc. can be a float or a time varying parameter.
    For time-varying parameters use e.g. "lambda T: T / (T + 1.0)"
    where the expression in the lambda is a tensorflow expression

    Args:
      global_step: tensorflow variable indicating the step.
      max_matrix_size: We do not perform SVD for matrices larger than this.
      gbar_decay:
      gbar_weight:  Used to update gbar:
            gbar[t] = gbar_decay[t] * gbar[t-1] + gbar_weight[t] * g[t]
      mat_gbar_decay:
      mat_gbar_weight:  Used to update mat_gbar:
           mat_gbar_j[t] = mat_gbar_decay[t] * mat_gbar_j[t-1]
                           + mat_gbar_weight[t] * gg_j[t]
      learning_rate: Similar to SGD
      svd_interval: We should do SVD after this many steps. Default = 1, i.e.
                    every step. Usually 20 leads to no loss of accuracy, and
                    50 or 100 is also OK. May also want more often early,
                    and less often later - set in caller as for example:
                    "svd_interval = lambda(T): tf.cond(
                        T < 2000, lambda: 20.0, lambda: 1000.0)"
      precond_update_interval: We should update the preconditioners after
                               this many steps. Default = 1. Usually less than
                               svd_interval.
      epsilon:  epsilon * I_n is added to each mat_gbar_j for stability
      alpha:  total power of the preconditioners.
      use_iterative_root: should the optimizer use SVD (faster) or the
                          iterative root method (for TPU) for finding the
                          roots of PSD matrices.
      use_locking:
      name: name of optimizer.
    """

    super(ShampooOptimizer, self).__init__(use_locking, name)

    self._global_step = math_ops.to_float(global_step)
    self._max_matrix_size = max_matrix_size
    self._gbar_decay = gbar_decay
    self._gbar_weight = gbar_weight
    self._mat_gbar_decay = mat_gbar_decay
    self._mat_gbar_weight = mat_gbar_weight
    self._learning_rate = learning_rate
    self._svd_interval = svd_interval
    self._precond_update_interval = precond_update_interval
    self._epsilon = epsilon
    self._alpha = alpha
    self._use_iterative_root = use_iterative_root
    self._name = name

  def _create_slots(self, var_list):
    for v in var_list:
      with ops.colocate_with(v):
        _ = self._zeros_slot(v, "gbar", self._name)
        shape = np.array(v.get_shape())
        for i, d in enumerate(shape):
          d_tensor = ops.convert_to_tensor(d)
          if d < self._max_matrix_size:
            mat_g_init = array_ops.zeros_like(linalg_ops.eye(d_tensor))
            if self._svd_interval > 1:
              _ = self._get_or_make_slot(v, linalg_ops.eye(d_tensor),
                                         "H_" + str(i), self._name)
          else:
            mat_g_init = array_ops.zeros([d_tensor])

          _ = self._get_or_make_slot(v, mat_g_init, "Gbar_" + str(i),
                                     self._name)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  def _apply_dense(self, grad, var):
    return self._apply_gradient(grad, var)

  def _resource_apply_sparse(self, grad_values, var, grad_indices):
    return self._apply_sparse_shared(grad_values, grad_indices, var)

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(grad.values, grad.indices, var)

  def _apply_sparse_shared(self, grad_values, grad_indices, var):
    if var.get_shape()[0] < self._max_matrix_size or self._gbar_decay != 0.0:
      # The dimension is small enough, we can make the variable dense and
      # do a dense update
      dense_grad = array_ops.scatter_nd(
          array_ops.expand_dims(grad_indices, axis=1), grad_values,
          array_ops.shape(var, out_type=grad_indices.dtype))
      return self._apply_gradient(dense_grad, var)
    return self._apply_gradient(grad_values, var, grad_indices)

  def _weighted_average(self, var, weight, weight_t, rest):
    """Computes exponential weighted average: var = weight_t * var + rest.

    Important to ensure that var does not occur in rest, otherwise
    we can get race conditions in a distributed setting.

    Args:
      var: variable to be updated
      weight: parameter to be checked. If it is a constant, we can optimize.
      weight_t: current value of parameter, used for weighting
      rest: the remaining tensor to be added

    Returns:
      updated variable.
    """
    if weight == 0.0:
      return rest       # no need to update var, we will never use it.
    if weight == 1.0:   # common case
      return state_ops.assign_add(var, rest)
    # The op below can cause race conditions in a distributed setting,
    # since computing weight_t * var + rest can take some time, during
    # which var may be set by another worker. To prevent this, it should
    # be implemented as a C++ op.
    return var.assign_add((weight_t - 1) * var + rest)

  def _update_mat_g(self, mat_g, grad, axes, mat_gbar_decay,
                    mat_gbar_weight, i):
    """Updates the cumulative outer products of the gradients.

    Args:
      mat_g: the matrix to be updated
      grad: the gradient of the variable
      axes: a list of k-1 integers 0 to k-1, except i
      mat_gbar_decay: constant for weighted average:
          mat_g = mat_g * decay + grad * weight
      mat_gbar_weight: constant for weighted average
      i: index of dimension to be updated.

    Returns:
      updated mat_g = mat_g * mat_gbar_decay + grad_outer * mat_gbar_weight

    In Einstein notation if i = 0: grad_outer_aa'= g_abcd g_a'bcd
    thus grad_outer is a matrix d_i x d_i, where d_i is the size of the
    i'th dimension of g.
    Alternate view: If mat_i(grad) is the flattening of grad to a
    d_i x (d_1d_2...d_{i-1}d_{i+1}...d_k) matrix, then
         grad_outer = mat_i(grad) mat_i(grad).transpose
    """
    grad_outer = math_ops.tensordot(grad, grad, axes=(axes, axes),
                                    name="grad_outer_" + str(i))
    return self._weighted_average(mat_g, self._mat_gbar_decay, mat_gbar_decay,
                                  mat_gbar_weight * grad_outer)

  def _compute_power_svd(self, var, mat_g, mat_g_size, alpha, mat_h_slot_name):
    """Computes mat_h = mat_g^alpha using svd. mat_g is a symmetric PSD matrix.

    Args:
      var: the variable we are updating.
      mat_g: the symmetric PSD matrix whose power it to be computed
      mat_g_size: size of mat_g
      alpha: a real number
      mat_h_slot_name: name of slot to store the power, if needed.

    Returns:
      mat_h = mat_g^alpha

    Stores mat_h in the appropriate slot, if it exists.
    Note that mat_g is PSD. So we could use linalg_ops.self_adjoint_eig.
    """
    if mat_g_size == 1:
      mat_h = math_ops.pow(mat_g + self._epsilon, alpha)
    else:
      damping = self._epsilon * linalg_ops.eye(math_ops.to_int32(mat_g_size))
      diag_d, mat_u, mat_v = linalg_ops.svd(mat_g + damping, full_matrices=True)
      mat_h = math_ops.matmul(
          mat_v * math_ops.pow(math_ops.maximum(diag_d, self._epsilon), alpha),
          array_ops.transpose(mat_u))
    if mat_h_slot_name is not None:
      return state_ops.assign(self.get_slot(var, mat_h_slot_name), mat_h)
    return mat_h

  def _compute_power_iter(self, var, mat_g, mat_g_size, alpha, mat_h_slot_name,
                          iter_count=100, epsilon=1e-6):
    """Computes mat_g^alpha, where alpha = -1/p, p a positive integer.

    We use an iterative Schur-Newton method from equation 3.2 on page 9 of:

    A Schur-Newton Method for the Matrix p-th Root and its Inverse
    by Chun-Hua Guo and Nicholas J. Higham
    SIAM Journal on Matrix Analysis and Applications,
    2006, Vol. 28, No. 3 : pp. 788-804
    https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

    Args:
      var: the variable we are updating.
      mat_g: the symmetric PSD matrix whose power it to be computed
      mat_g_size: size of mat_g.
      alpha: exponent, must be -1/p for p a positive integer.
      mat_h_slot_name: name of slot to store the power, if needed.
      iter_count: Maximum number of iterations.
      epsilon: accuracy indicator, useful for early termination.

    Returns:
      mat_g^alpha
    """

    identity = linalg_ops.eye(math_ops.to_int32(mat_g_size))

    def MatPower(mat_m, p):
      """Computes mat_m^p, for p a positive integer.

      Power p is known at graph compile time, so no need for loop and cond.
      Args:
        mat_m: a square matrix
        p: a positive integer

      Returns:
        mat_m^p
      """
      assert p == int(p) and p > 0
      power = None
      while p > 0:
        if p % 2 == 1:
          power = math_ops.matmul(mat_m, power) if power is not None else mat_m
        p //= 2
        mat_m = math_ops.matmul(mat_m, mat_m)
      return power

    def IterCondition(i, mat_m, _):
      return math_ops.logical_and(
          i < iter_count,
          math_ops.reduce_max(math_ops.abs(mat_m - identity)) > epsilon)

    def IterBody(i, mat_m, mat_x):
      mat_m_i = (1 - alpha) * identity + alpha * mat_m
      return (i + 1, math_ops.matmul(MatPower(mat_m_i, -1.0/alpha), mat_m),
              math_ops.matmul(mat_x, mat_m_i))

    if mat_g_size == 1:
      mat_h = math_ops.pow(mat_g + self._epsilon, alpha)
    else:
      damped_mat_g = mat_g + self._epsilon * identity
      z = (1 - 1 / alpha) / (2 * linalg_ops.norm(damped_mat_g))
      # The best value for z is
      # (1 - 1/alpha) * (c_max^{-alpha} - c_min^{-alpha}) /
      #                 (c_max^{1-alpha} - c_min^{1-alpha})
      # where c_max and c_min are the largest and smallest singular values of
      # damped_mat_g.
      # The above estimate assumes that c_max > c_min * 2^p. (p = -1/alpha)
      # Can replace above line by the one below, but it is less accurate,
      # hence needs more iterations to converge.
      # z = (1 - 1/alpha) / math_ops.trace(damped_mat_g)
      # If we want the method to always converge, use z = 1 / norm(damped_mat_g)
      # or z = 1 / math_ops.trace(damped_mat_g), but these can result in many
      # extra iterations.
      _, _, mat_h = control_flow_ops.while_loop(
          IterCondition, IterBody,
          [0, damped_mat_g * z, identity * math_ops.pow(z, -alpha)])
    if mat_h_slot_name is not None:
      return state_ops.assign(self.get_slot(var, mat_h_slot_name), mat_h)
    return mat_h

  def _compute_power(self, var, mat_g, mat_g_size, alpha, mat_h_slot_name=None):
    """Just a switch between the iterative power vs svd."""
    with ops.name_scope("matrix_iterative_power"):
      if self._use_iterative_root:
        return self._compute_power_iter(var, mat_g, mat_g_size, alpha,
                                        mat_h_slot_name)
      else:
        return self._compute_power_svd(var, mat_g, mat_g_size, alpha,
                                       mat_h_slot_name)

  def _apply_gradient(self, grad, var, indices=None):
    """The main function to update a variable.

    Args:
      grad: A Tensor containing gradient to apply.
      var: A Tensor containing the variable to update.
      indices: An array of integers, for sparse update.

    Returns:
      Updated variable var = var - learning_rate * preconditioner * grad

    If the gradient is dense, var and grad have the same shape.
    If the update is sparse, then the first dimension of the gradient and var
    may differ, others are all the same. In this case the indices array
    provides the set of indices of the variable which are to be updated with
    each row of the gradient.
    """
    global_step = self._global_step + 1

    # Update accumulated weighted average of gradients
    gbar = self.get_slot(var, "gbar")
    gbar_decay_t = GetParam(self._gbar_decay, global_step)
    gbar_weight_t = GetParam(self._gbar_weight, global_step)
    if indices is not None:
      # Note - the sparse update is not easily implemented, since the
      # algorithm needs all indices of gbar to be updated
      # if mat_gbar_decay != 1 or mat_gbar_decay != 0.
      # One way to make mat_gbar_decay = 1 is by rescaling.
      # If we want the update:
      #         G_{t+1} = a_{t+1} G_t + b_{t+1} w_t
      # define:
      #         r_{t+1} = a_{t+1} * r_t
      #         h_t = G_t / r_t
      # Then:
      #         h_{t+1} = h_t + (b_{t+1} / r_{t+1}) * w_t
      # So we get the mat_gbar_decay = 1 as desired.
      # We can implement this in a future version as needed.
      # However we still need gbar_decay = 0, otherwise all indices
      # of the variable will need to be updated.
      if self._gbar_decay != 0.0:
        tf_logging.warning("Not applying momentum for variable: %s" % var.name)
      gbar_updated = grad
    else:
      gbar_updated = self._weighted_average(gbar, self._gbar_decay,
                                            gbar_decay_t,
                                            gbar_weight_t * grad)

    # Update the preconditioners and compute the preconditioned gradient
    shape = var.get_shape()
    mat_g_list = []
    for i in range(len(shape)):
      mat_g_list.append(self.get_slot(var, "Gbar_" + str(i)))
    mat_gbar_decay_t = GetParam(self._mat_gbar_decay, global_step)
    mat_gbar_weight_t = GetParam(self._mat_gbar_weight, global_step)

    preconditioned_grad = gbar_updated
    v_rank = len(mat_g_list)
    neg_alpha = - GetParam(self._alpha, global_step) / v_rank
    svd_interval = GetParam(self._svd_interval, global_step)
    precond_update_interval = GetParam(self._precond_update_interval,
                                       global_step)
    for i, mat_g in enumerate(mat_g_list):
      # axes is the list of indices to reduce - everything but the current i.
      axes = list(range(i)) + list(range(i+1, v_rank))
      if shape[i] < self._max_matrix_size:
        # If the tensor size is sufficiently small perform full Shampoo update
        # Note if precond_update_interval > 1 and mat_gbar_decay_t != 1, this
        # is not strictly correct. However we will use it for now, and
        # fix if needed. (G_1 = aG + bg ==> G_n = a^n G + (1+a+..+a^{n-1})bg)

        # pylint: disable=g-long-lambda,cell-var-from-loop
        mat_g_updated = control_flow_ops.cond(
            math_ops.mod(global_step, precond_update_interval) < 1,
            lambda: self._update_mat_g(
                mat_g, grad, axes, mat_gbar_decay_t,
                mat_gbar_weight_t * precond_update_interval, i),
            lambda: mat_g)

        if self._svd_interval == 1:
          mat_h = self._compute_power(var, mat_g_updated, shape[i], neg_alpha)
        else:
          mat_h = control_flow_ops.cond(
              math_ops.mod(global_step, svd_interval) < 1,
              lambda: self._compute_power(var, mat_g_updated, shape[i],
                                          neg_alpha, "H_" + str(i)),
              lambda: self.get_slot(var, "H_" + str(i)))

        # mat_h is a square matrix of size d_i x d_i
        # preconditioned_grad is a d_i x ... x d_n x d_0 x ... d_{i-1} tensor
        # After contraction with a d_i x d_i tensor
        # it becomes a d_{i+1} x ... x d_n x d_0 x ... d_i tensor
        # (the first dimension is contracted out, and the second dimension of
        # mat_h is appended).  After going through all the indices, it becomes
        # a d_0 x ... x d_n tensor again.
        preconditioned_grad = math_ops.tensordot(preconditioned_grad, mat_h,
                                                 axes=([0], [0]),
                                                 name="precond_" + str(i))
      else:
        # Tensor size is too large -- perform diagonal Shampoo update
        grad_outer = math_ops.reduce_sum(grad * grad, axis=axes)
        if i == 0 and indices is not None:
          assert self._mat_gbar_decay == 1.0
          mat_g_updated = state_ops.scatter_add(mat_g, indices,
                                                mat_gbar_weight_t * grad_outer)
          mat_h = math_ops.pow(
              array_ops.gather(mat_g_updated, indices) + self._epsilon,
              neg_alpha)
        else:
          mat_g_updated = self._weighted_average(mat_g,
                                                 self._mat_gbar_decay,
                                                 mat_gbar_decay_t,
                                                 mat_gbar_weight_t * grad_outer)
          mat_h = math_ops.pow(mat_g_updated + self._epsilon, neg_alpha)

        # Need to do the transpose to ensure that the tensor becomes
        # a d_{i+1} x ... x d_n x d_0 x ... d_i tensor as described above.
        preconditioned_grad = array_ops.transpose(
            preconditioned_grad, perm=list(range(1, v_rank)) + [0]) * mat_h

    # Update the variable based on the Shampoo update
    learning_rate_t = GetParam(self._learning_rate, global_step)
    if indices is not None:
      var_updated = state_ops.scatter_add(
          var, indices, -learning_rate_t * preconditioned_grad)
    else:
      var_updated = state_ops.assign_sub(var,
                                         learning_rate_t * preconditioned_grad)
    return var_updated
