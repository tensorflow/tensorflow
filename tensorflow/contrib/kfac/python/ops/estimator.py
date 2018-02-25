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

import contextlib
import itertools

import numpy as np

from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.util import nest


class _DeviceContextGenerator(object):
  """Class for generating device contexts in a round-robin fashion."""

  def __init__(self, devices):
    """Creates a _DeviceContextGenerator object.

    Example usage:

    ```python
    dcg = _DeviceContextGenerator(['/gpu:0', 'gpu:1'])
    with dcg():
      # All operations in this context will be placed on GPU 0
      ...
    with dcg():
      # All operations in this context will be placed on GPU 1
      ...
    ```

    Args:
      devices: An iterable of device strings (or None). Successive calls to
          __call__ will give contexts which place devices on these devices in
          a round-robin fashion.
    """
    self._cycle = None if devices is None else itertools.cycle(devices)

  @contextlib.contextmanager
  def __call__(self):
    """Returns a context manager specifying the default device."""
    if self._cycle is None:
      yield
    else:
      with tf_ops.device(next(self._cycle)):
        yield


class FisherEstimator(object):
  """Fisher estimator class supporting various approximations of the Fisher.

  Attributes:
    cov_update_thunks: list of no-arg functions. Executing a function adds
      covariance update ops for a single FisherFactor to the graph.
    cov_update_ops: List of Ops. Running an op updates covariance matrices for a
      single FisherFactor.
    cov_update_op: Op. Running updates covariance matrices for all
      FisherFactors.
    inv_update_thunks: list of no-arg functions.  Executing a function adds
      inverse update ops for a single FisherFactor to the graph.
    inv_update_ops: List of Ops. Running an op updates inverse matrices for a
      single FisherFactor.
    inv_update_op: Op. Running updates inverse matrices for all FisherFactors.
  """

  def __init__(self,
               damping_fn,
               variables,
               cov_ema_decay,
               layer_collection,
               estimation_mode="gradients",
               colocate_gradients_with_ops=True,
               cov_devices=None,
               inv_devices=None):
    """Create a FisherEstimator object.

    Args:
      damping_fn: Function, accepts no arguments and returns damping value.
      variables: A list of the variables for which to estimate the Fisher. This
          must match the variables registered in layer_collection (if it is not
          None).
      cov_ema_decay: The decay factor used when calculating the covariance
          estimate moving averages.
      layer_collection: The layer collection object, which holds the fisher
          blocks, kronecker factors, and losses associated with the
          graph.
      estimation_mode: The type of estimator to use for the Fishers.  Can be
          'gradients', 'empirical', 'curvature_prop', or 'exact'.
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
      colocate_gradients_with_ops: Whether we should request gradients be
          colocated with their respective ops. (Default: True)
      cov_devices: Iterable of device strings (e.g. '/gpu:0'). Covariance
          computations will be placed on these devices in a round-robin fashion.
          Can be None, which means that no devices are specified.
      inv_devices: Iterable of device strings (e.g. '/gpu:0'). Inversion
          computations will be placed on these devices in a round-robin fashion.
          Can be None, which means that no devices are specified.

    Raises:
      ValueError: If no losses have been registered with layer_collection.
    """
    self._damping_fn = damping_fn
    self._cov_ema_decay = cov_ema_decay
    self._variables = variables
    self._estimation_mode = estimation_mode
    self._layers = layer_collection
    self._layers.create_subgraph()
    self._layers.check_registration(variables)
    self._gradient_fns = {
        "gradients": self._get_grads_lists_gradients,
        "empirical": self._get_grads_lists_empirical,
        "curvature_prop": self._get_grads_lists_curvature_prop,
        "exact": self._get_grads_lists_exact
    }
    self._colocate_gradients_with_ops = colocate_gradients_with_ops

    # TODO(b/70674513): Factor device placement outside of this class.
    self._cov_device_context_generator = _DeviceContextGenerator(cov_devices)
    if inv_devices == cov_devices:
      self._inv_device_context_generator = self._cov_device_context_generator
    else:
      self._inv_device_context_generator = _DeviceContextGenerator(inv_devices)

    self._instantiate_factors()

    self.cov_update_thunks = [
        self._create_cov_update_thunk(factor)
        for factor in self._layers.get_factors()
    ]
    self.cov_update_ops = [thunk() for thunk in self.cov_update_thunks]
    self.cov_update_op = control_flow_ops.group(
        self.cov_update_ops, name="cov_update_op")

    self.inv_update_thunks = [
        self._create_inv_update_thunk(factor)
        for factor in self._layers.get_factors()
    ]
    self.inv_update_ops = [thunk() for thunk in self.inv_update_thunks]
    self.inv_update_op = control_flow_ops.group(
        self.inv_update_ops, name="inv_update_op")

  @property
  def variables(self):
    return self._variables

  @property
  def damping(self):
    return self._damping_fn()

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

  def _instantiate_factors(self):
    """Instantiates FisherFactors' variables.

    Raises:
      ValueError: If estimation_mode was improperly specified at construction.
    """
    fisher_blocks_list = self._layers.get_blocks()
    tensors_to_compute_grads = [
        fb.tensors_to_compute_grads() for fb in fisher_blocks_list
    ]

    try:
      grads_lists = self._gradient_fns[self._estimation_mode](
          tensors_to_compute_grads)
    except KeyError:
      raise ValueError("Unrecognized value {} for estimation_mode.".format(
          self._estimation_mode))

    # TODO(b/68033310): This loop round-robins the "concat" operations which
    # gather the inputs for the cov_updates. In future, we might do these
    # computations locally then communicate the results, which would require a
    # modification to this code.
    for grads_list, fb in zip(grads_lists, fisher_blocks_list):
      with self._cov_device_context_generator():
        fb.instantiate_factors(grads_list, self.damping)

  def _create_cov_update_thunk(self, factor):
    """Constructs a covariance update thunk for a single FisherFactor."""

    def thunk():
      with tf_ops.name_scope(
          "create_cov_update_thunk", values=[self._cov_ema_decay]):
        return factor.make_covariance_update_op(self._cov_ema_decay)

    return thunk

  def _create_inv_update_thunk(self, factor):
    """Constructs an inverse update thunk for a single FisherFactor."""

    def thunk():
      with tf_ops.name_scope("create_inv_update_thunk"):
        with self._inv_device_context_generator():
          return control_flow_ops.group(factor.make_inverse_update_ops())

    return thunk

  def _get_grads_lists_gradients(self, tensors):
    grads_flat = gradients_impl.gradients(
        self._layers.total_sampled_loss(),
        nest.flatten(tensors),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_grads_lists_empirical(self, tensors):
    grads_flat = gradients_impl.gradients(
        self._layers.total_loss(),
        nest.flatten(tensors),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_transformed_random_signs(self):
    transformed_random_signs = []
    for loss in self._layers.losses:
      transformed_random_signs.append(
          loss.multiply_fisher_factor(
              utils.generate_random_signs(loss.fisher_factor_inner_shape)))
    return transformed_random_signs

  def _get_grads_lists_curvature_prop(self, tensors):
    loss_inputs = list(loss.inputs for loss in self._layers.losses)
    transformed_random_signs = self._get_transformed_random_signs()
    grads_flat = gradients_impl.gradients(
        nest.flatten(loss_inputs),
        nest.flatten(tensors),
        grad_ys=nest.flatten(transformed_random_signs),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_grads_lists_exact(self, tensors):
    """No docstring required."""
    # Loop over all coordinates of all losses.
    grads_all = []
    for loss in self._layers.losses:
      for index in np.ndindex(*loss.fisher_factor_inner_static_shape[1:]):
        transformed_one_hot = loss.multiply_fisher_factor_replicated_one_hot(
            index)
        grads_flat = gradients_impl.gradients(
            loss.inputs,
            nest.flatten(tensors),
            grad_ys=transformed_one_hot,
            colocate_gradients_with_ops=self._colocate_gradients_with_ops)
        grads_all.append(nest.pack_sequence_as(tensors, grads_flat))
    return zip(*grads_all)
