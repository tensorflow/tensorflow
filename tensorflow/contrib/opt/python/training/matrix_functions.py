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
"""Matrix functions contains iterative methods for M^p."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


def matrix_square_root(mat_a, mat_a_size, iter_count=100, ridge_epsilon=1e-4):
  """Iterative method to get matrix square root.

  Stable iterations for the matrix square root, Nicholas J. Higham

  Page 231, Eq 2.6b
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf

  Args:
    mat_a: the symmetric PSD matrix whose matrix square root be computed
    mat_a_size: size of mat_a.
    iter_count: Maximum number of iterations.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.

  Returns:
    mat_a^0.5
  """

  def _iter_condition(i, unused_mat_y, unused_old_mat_y, unused_mat_z,
                      unused_old_mat_z, err, old_err):
    # This method require that we check for divergence every step.
    return math_ops.logical_and(i < iter_count, err < old_err)

  def _iter_body(i, mat_y, unused_old_mat_y, mat_z, unused_old_mat_z, err,
                 unused_old_err):
    current_iterate = 0.5 * (3.0 * identity - math_ops.matmul(mat_z, mat_y))
    current_mat_y = math_ops.matmul(mat_y, current_iterate)
    current_mat_z = math_ops.matmul(current_iterate, mat_z)
    # Compute the error in approximation.
    mat_sqrt_a = current_mat_y * math_ops.sqrt(norm)
    mat_a_approx = math_ops.matmul(mat_sqrt_a, mat_sqrt_a)
    residual = mat_a - mat_a_approx
    current_err = math_ops.sqrt(math_ops.reduce_sum(residual * residual)) / norm
    return i + 1, current_mat_y, mat_y, current_mat_z, mat_z, current_err, err

  identity = linalg_ops.eye(math_ops.to_int32(mat_a_size))
  mat_a = mat_a + ridge_epsilon * identity
  norm = math_ops.sqrt(math_ops.reduce_sum(mat_a * mat_a))
  mat_init_y = mat_a / norm
  mat_init_z = identity
  init_err = norm

  _, _, prev_mat_y, _, _, _, _ = control_flow_ops.while_loop(
      _iter_condition, _iter_body, [
          0, mat_init_y, mat_init_y, mat_init_z, mat_init_z, init_err,
          init_err + 1.0
      ])
  return prev_mat_y * math_ops.sqrt(norm)


def matrix_inverse_pth_root(mat_g,
                            mat_g_size,
                            alpha,
                            iter_count=100,
                            epsilon=1e-6,
                            ridge_epsilon=1e-6):
  """Computes mat_g^alpha, where alpha = -1/p, p a positive integer.

  We use an iterative Schur-Newton method from equation 3.2 on page 9 of:

  A Schur-Newton Method for the Matrix p-th Root and its Inverse
  by Chun-Hua Guo and Nicholas J. Higham
  SIAM Journal on Matrix Analysis and Applications,
  2006, Vol. 28, No. 3 : pp. 788-804
  https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

  Args:
    mat_g: the symmetric PSD matrix whose power it to be computed
    mat_g_size: size of mat_g.
    alpha: exponent, must be -1/p for p a positive integer.
    iter_count: Maximum number of iterations.
    epsilon: accuracy indicator, useful for early termination.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.

  Returns:
    mat_g^alpha
  """

  identity = linalg_ops.eye(math_ops.to_int32(mat_g_size))

  def mat_power(mat_m, p):
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

  def _iter_condition(i, mat_m, _):
    return math_ops.logical_and(
        i < iter_count,
        math_ops.reduce_max(math_ops.abs(mat_m - identity)) > epsilon)

  def _iter_body(i, mat_m, mat_x):
    mat_m_i = (1 - alpha) * identity + alpha * mat_m
    return (i + 1, math_ops.matmul(mat_power(mat_m_i, -1.0 / alpha), mat_m),
            math_ops.matmul(mat_x, mat_m_i))

  if mat_g_size == 1:
    mat_h = math_ops.pow(mat_g + ridge_epsilon, alpha)
  else:
    damped_mat_g = mat_g + ridge_epsilon * identity
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
        _iter_condition, _iter_body,
        [0, damped_mat_g * z, identity * math_ops.pow(z, -alpha)])
  return mat_h
