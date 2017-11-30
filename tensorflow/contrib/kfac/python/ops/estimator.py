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
"""Defines the high-level Fisher estimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.util import nest


class FisherEstimator(object):
  """Fisher estimator class supporting various approximations of the Fisher."""

  def __init__(self,
               variables,
               cov_ema_decay,
               damping,
               layer_collection,
               estimation_mode="gradients"):
    """Create a FisherEstimator object.

    Args:
      variables: A list of the variables for which to estimate the Fisher. This
          must match the variables registered in layer_collection (if it is not
          None).
      cov_ema_decay: The decay factor used when calculating the covariance
          estimate moving averages.
      damping: The damping factor used to stabilize training due to errors in
          the local approximation with the Fisher information matrix, and to
          regularize the update direction by making it closer to the gradient.
          (Higher damping means the update looks more like a standard gradient
          update - see Tikhonov regularization.)
      layer_collection: The layer collection object, which holds the fisher
          blocks, kronecker factors, and losses associated with the
          graph.
      estimation_mode: The type of estimator to use for the Fishers.  Can be
          'gradients', 'empirical', 'curvature_propagation', or 'exact'.
          (Default: 'gradients').  'gradients' is the basic estimation approach
          from the original K-FAC paper.  'empirical' computes the 'empirical'
          Fisher information matrix (which uses the data's distribution for the
          targets, as opposed to the true Fisher which uses the model's
          distribution) and requires that each registered loss have specified
          targets. 'curvature_propagation' is a method which estimates the
          Fisher using self-products of random 1/-1 vectors times "half-factors"
          of the Fisher, as described here: https://arxiv.org/abs/1206.6464 .
          Finally, 'exact' is the obvious generalization of Curvature
          Propagation to compute the exact Fisher (modulo any additional
          diagonal or Kronecker approximations) by looping over one-hot vectors
          for each coordinate of the output instead of using 1/-1 vectors.  It
          is more expensive to compute than the other three options by a factor
          equal to the output dimension, roughly speaking.

    Raises:
      ValueError: If no losses have been registered with layer_collection.
    """

    self._variables = variables
    self._damping = damping
    self._estimation_mode = estimation_mode
    self._layers = layer_collection
    self._layers.create_subgraph()
    self._check_registration(variables)
    setup = self._setup(cov_ema_decay)
    self.cov_update_op, self.inv_update_op, self.inv_updates_dict = setup

  @property
  def variables(self):
    return self._variables

  @property
  def damping(self):
    return self._damping

  def _apply_transformation(self, vecs_and_vars, transform):
    """Applies an block-wise transformation to the corresponding vectors.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.
      transform: A function of the form f(fb, vec), where vec is the vector
          to transform and fb is its corresponding block in the matrix, that
          returns the transformed vector.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """

    vecs = utils.SequenceDict((var, vec) for vec, var in vecs_and_vars)

    trans_vecs = utils.SequenceDict()

    for params, fb in self._layers.fisher_blocks.items():
      trans_vecs[params] = transform(fb, vecs[params])

    return [(trans_vecs[var], var) for _, var in vecs_and_vars]

  def multiply_inverse(self, vecs_and_vars):
    """Multiplies the vecs by the corresponding (damped) inverses of the blocks.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """

    return self._apply_transformation(vecs_and_vars,
                                      lambda fb, vec: fb.multiply_inverse(vec))

  def multiply(self, vecs_and_vars):
    """Multiplies the vectors by the corresponding (damped) blocks.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """

    return self._apply_transformation(vecs_and_vars,
                                      lambda fb, vec: fb.multiply(vec))

  def _check_registration(self, variables):
    """Checks that all variable uses have been registered properly.

    Args:
      variables: List of variables.

    Raises:
      ValueError: If any registered variables are not included in the list.
      ValueError: If any variable in the list is not registered.
      ValueError: If any variable in the list is registered with the wrong
          number of "uses" in the subgraph recorded (vs the number of times that
          variable is actually used in the subgraph).
    """
    # Note that overlapping parameters (i.e. those that share variables) will
    # be caught by layer_collection.LayerParametersDict during registration.

    reg_use_map = self._layers.get_use_count_map()

    error_messages = []

    for var in variables:
      total_uses = self._layers.subgraph.variable_uses(var)
      reg_uses = reg_use_map[var]

      if reg_uses == 0:
        error_messages.append("Variable {} not registered.".format(var))
      elif (not math.isinf(reg_uses)) and reg_uses != total_uses:
        error_messages.append(
            "Variable {} registered with wrong number of uses ({} "
            "vs {} actual).".format(var, reg_uses, total_uses))

    num_get_vars = len(reg_use_map)

    if num_get_vars > len(variables):
      error_messages.append("{} registered variables were not included in list."
                            .format(num_get_vars - len(variables)))

    if error_messages:
      error_messages = [
          "Found the following errors with variable registration:"
      ] + error_messages
      raise ValueError("\n\t".join(error_messages))

  def _setup(self, cov_ema_decay):
    """Sets up the various operations.

    Args:
      cov_ema_decay: The decay factor used when calculating the covariance
          estimate moving averages.

    Returns:
      A triple (covs_update_op, invs_update_op, inv_updates_dict), where
      covs_update_op is the grouped Op to update all the covariance estimates,
      invs_update_op is the grouped Op to update all the inverses, and
      inv_updates_dict is a dict mapping Op names to individual inverse updates.

    Raises:
      ValueError: If estimation_mode was improperly specified at construction.
    """
    damping = self.damping

    fisher_blocks_list = self._layers.get_blocks()

    tensors_to_compute_grads = [
        fb.tensors_to_compute_grads() for fb in fisher_blocks_list
    ]
    tensors_to_compute_grads_flat = nest.flatten(tensors_to_compute_grads)

    if self._estimation_mode == "gradients":
      grads_flat = gradients_impl.gradients(self._layers.total_sampled_loss(),
                                            tensors_to_compute_grads_flat)
      grads_all = nest.pack_sequence_as(tensors_to_compute_grads, grads_flat)
      grads_lists = tuple((grad,) for grad in grads_all)

    elif self._estimation_mode == "empirical":
      grads_flat = gradients_impl.gradients(self._layers.total_loss(),
                                            tensors_to_compute_grads_flat)
      grads_all = nest.pack_sequence_as(tensors_to_compute_grads, grads_flat)
      grads_lists = tuple((grad,) for grad in grads_all)

    elif self._estimation_mode == "curvature_prop":
      loss_inputs = list(loss.inputs for loss in self._layers.losses)
      loss_inputs_flat = nest.flatten(loss_inputs)

      transformed_random_signs = list(loss.multiply_fisher_factor(
          utils.generate_random_signs(loss.fisher_factor_inner_shape))
                                      for loss in self._layers.losses)

      transformed_random_signs_flat = nest.flatten(transformed_random_signs)

      grads_flat = gradients_impl.gradients(loss_inputs_flat,
                                            tensors_to_compute_grads_flat,
                                            grad_ys
                                            =transformed_random_signs_flat)
      grads_all = nest.pack_sequence_as(tensors_to_compute_grads, grads_flat)
      grads_lists = tuple((grad,) for grad in grads_all)

    elif self._estimation_mode == "exact":
      # Loop over all coordinates of all losses.
      grads_all = []
      for loss in self._layers.losses:
        for index in np.ndindex(*loss.fisher_factor_inner_static_shape[1:]):
          transformed_one_hot = loss.multiply_fisher_factor_replicated_one_hot(
              index)
          grads_flat = gradients_impl.gradients(loss.inputs,
                                                tensors_to_compute_grads_flat,
                                                grad_ys=transformed_one_hot)
          grads_all.append(nest.pack_sequence_as(tensors_to_compute_grads,
                                                 grads_flat))

      grads_lists = zip(*grads_all)

    else:
      raise ValueError("Unrecognized value {} for estimation_mode.".format(
          self._estimation_mode))

    for grads_list, fb in zip(grads_lists, fisher_blocks_list):
      fb.instantiate_factors(grads_list, damping)

    cov_updates = [
        factor.make_covariance_update_op(cov_ema_decay)
        for factor in self._layers.get_factors()
    ]
    inv_updates = {
        op.name: op
        for factor in self._layers.get_factors()
        for op in factor.make_inverse_update_ops()
    }

    return control_flow_ops.group(*cov_updates), control_flow_ops.group(
        *inv_updates.values()), inv_updates
