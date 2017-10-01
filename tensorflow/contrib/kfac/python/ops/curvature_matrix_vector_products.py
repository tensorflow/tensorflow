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
"""Curvature matrix-vector multiplication."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


class CurvatureMatrixVectorProductComputer(object):
  """Class for computing matrix-vector products for Fishers, GGNs and Hessians.

  In other words we compute M*v where M is the matrix, v is the vector, and
  * refers to standard matrix/vector multiplication (not element-wise
  multiplication).

  The matrices are defined in terms of some differential quantity of the total
  loss function with respect to a provided list of tensors ("wrt_tensors").
  For example, the Fisher associated with a log-prob loss w.r.t. the
  parameters.

  The vecs argument to each method are lists of tensors that must be the
  size as the corresponding ones from "wrt_tensors".  They represent
  the vector being multiplied.

  "factors" of the matrix M are defined as matrices B such that B*B^T = M.
  Methods that multiply by the factor B take a "loss_inner_vecs" argument
  instead of vecs, which must be a list of tensors with shapes given by the
  corresponding XXX_inner_shapes property.

  Note that matrix-vector products are not normalized by the batch size, nor
  are any damping terms added to the results.  These things can easily be
  applied externally, if desired.

  See for example: www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf
  and https://arxiv.org/abs/1412.1193 for more information about the
  generalized Gauss-Newton, Fisher, etc., and how to compute matrix-vector
  products.
  """

  def __init__(self, losses, wrt_tensors):
    """Create a CurvatureMatrixVectorProductComputer object.

    Args:
      losses: A list of LossFunction instances whose sum defines the total loss.
      wrt_tensors: A list of Tensors to compute the differential quantities
        defining the matrices with respect to (see class description).
    """
    self._losses = losses
    self._inputs_to_losses = list(loss.inputs for loss in losses)
    self._inputs_to_losses_flat = nest.flatten(self._inputs_to_losses)
    self._wrt_tensors = wrt_tensors

  @property
  def _total_loss(self):
    return math_ops.add_n(tuple(loss.evaluate() for loss in self._losses))

  # Jacobian multiplication functions:
  # NOTE: These implementations use tf.gradients and thus aren't actually
  # computing partial derivatives, but total derivatives instead (despite what
  # the documentation for tf.gradients says).  Because we require partial
  # derivatives for Jacobians this implementation will only be correct if the
  # partial derivatives are equal to the full derivatives.  This happens as long
  # as the elements of wrt_tensors don't depend on each other in the graph.  If
  # these tensors are standard neural network parameters this will be true.
  def _multiply_jacobian(self, vecs):
    """Multiply vecs by the Jacobian of losses."""
    jacobian_vecs_flat = utils.fwd_gradients(
        self._inputs_to_losses_flat, self._wrt_tensors, grad_xs=vecs)
    return nest.pack_sequence_as(self._inputs_to_losses, jacobian_vecs_flat)

  def _multiply_jacobian_transpose(self, loss_vecs):
    """Multiply vecs by the transpose Jacobian of losses."""
    loss_vecs_flat = nest.flatten(loss_vecs)
    return gradients_impl.gradients(
        self._inputs_to_losses_flat, self._wrt_tensors, grad_ys=loss_vecs_flat)

  # Losses Fisher/Hessian multiplication functions:
  def _multiply_loss_fisher(self, loss_vecs):
    """Multiply loss_vecs by Fisher of total loss."""
    return tuple(
        loss.multiply_fisher(loss_vec)
        for loss, loss_vec in zip(self._losses, loss_vecs))

  def _multiply_loss_fisher_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of Fisher of total loss."""
    return tuple(
        loss.multiply_fisher_factor(loss_vec)
        for loss, loss_vec in zip(self._losses, loss_inner_vecs))

  def _multiply_loss_fisher_factor_transpose(self, loss_vecs):
    """Multiply loss_vecs by transpose factor of Fisher of total loss."""
    return tuple(
        loss.multiply_fisher_factor_transpose(loss_vec)
        for loss, loss_vec in zip(self._losses, loss_vecs))

  def _multiply_loss_hessian(self, loss_vecs):
    """Multiply loss_vecs by Hessian of total loss."""
    return tuple(
        loss.multiply_hessian(loss_vec)
        for loss, loss_vec in zip(self._losses, loss_vecs))

  def _multiply_loss_hessian_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of Hessian of total loss."""
    return tuple(
        loss.multiply_hessian_factor(loss_vec)
        for loss, loss_vec in zip(self._losses, loss_inner_vecs))

  def _multiply_loss_hessian_factor_transpose(self, loss_vecs):
    """Multiply loss_vecs by transpose factor of Hessian of total loss."""
    return tuple(
        loss.multiply_hessian_factor_transpose(loss_vec)
        for loss, loss_vec in zip(self._losses, loss_vecs))

  # Matrix-vector product functions:
  def multiply_fisher(self, vecs):
    """Multiply vecs by Fisher of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    loss_fisher_jacobian_vecs = self._multiply_loss_fisher(jacobian_vecs)
    return self._multiply_jacobian_transpose(loss_fisher_jacobian_vecs)

  def multiply_fisher_factor_transpose(self, vecs):
    """Multiply vecs by transpose of factor of Fisher of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    return self._multiply_loss_fisher_factor_transpose(jacobian_vecs)

  def multiply_fisher_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of Fisher of total loss."""
    fisher_factor_transpose_vecs = self._multiply_loss_fisher_factor_transpose(
        loss_inner_vecs)
    return self._multiply_jacobian_transpose(fisher_factor_transpose_vecs)

  def multiply_hessian(self, vecs):
    """Multiply vecs by Hessian of total loss."""
    return gradients_impl.gradients(
        gradients_impl.gradients(self._total_loss, self._wrt_tensors),
        self._wrt_tensors,
        grad_ys=vecs)

  def multiply_generalized_gauss_newton(self, vecs):
    """Multiply vecs by generalized Gauss-Newton of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    loss_hessian_jacobian_vecs = self._multiply_loss_hessian(jacobian_vecs)
    return self._multiply_jacobian_transpose(loss_hessian_jacobian_vecs)

  def multiply_generalized_gauss_newton_factor_transpose(self, vecs):
    """Multiply vecs by transpose of factor of GGN of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    return self._multiply_loss_hessian_factor_transpose(jacobian_vecs)

  def multiply_generalized_gauss_newton_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of GGN of total loss."""
    hessian_factor_transpose_vecs = (
        self._multiply_loss_hessian_factor_transpose(loss_inner_vecs))
    return self._multiply_jacobian_transpose(hessian_factor_transpose_vecs)

  # Shape properties for multiply_XXX_factor methods:
  @property
  def fisher_factor_inner_shapes(self):
    """Shapes required by multiply_fisher_factor."""
    return tuple(loss.fisher_factor_inner_shape for loss in self._losses)

  @property
  def generalized_gauss_newton_factor_inner_shapes(self):
    """Shapes required by multiply_generalized_gauss_newton_factor."""
    return tuple(loss.hessian_factor_inner_shape for loss in self._losses)
