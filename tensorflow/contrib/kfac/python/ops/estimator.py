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
from tensorflow.python.ops import variable_scope
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


def _make_thunk_on_device(func, device):
  def thunk():
    with tf_ops.device(device):
      return func()
  return thunk


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
               variables,
               cov_ema_decay,
               damping,
               layer_collection,
               exps=(-1,),
               estimation_mode="gradients",
               colocate_gradients_with_ops=True,
               name="FisherEstimator"):
    """Create a FisherEstimator object.

    Args:
      variables: A list of the variables for which to estimate the Fisher. This
          must match the variables registered in layer_collection (if it is not
          None).
      cov_ema_decay: The decay factor used when calculating the covariance
          estimate moving averages.
      damping: float. The damping factor used to stabilize training due to
          errors in the local approximation with the Fisher information matrix,
          and to regularize the update direction by making it closer to the
          gradient. (Higher damping means the update looks more like a standard
          gradient update - see Tikhonov regularization.)
      layer_collection: The layer collection object, which holds the fisher
          blocks, kronecker factors, and losses associated with the
          graph.
      exps: List of floats or ints. These represent the different matrix
          powers of the approximate Fisher that the FisherEstimator will be able
          to multiply vectors by. If the user asks for a matrix power other
          one of these (or 1, which is always supported), there will be a
          failure. (Default: (-1,))
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
      name: A string. A name given to this estimator, which is added to the
          variable scope when constructing variables and ops.
          (Default: "FisherEstimator")
    Raises:
      ValueError: If no losses have been registered with layer_collection.
    """
    self._variables = variables
    self._cov_ema_decay = cov_ema_decay
    self._damping = damping
    self._estimation_mode = estimation_mode
    self._layers = layer_collection
    self._gradient_fns = {
        "gradients": self._get_grads_lists_gradients,
        "empirical": self._get_grads_lists_empirical,
        "curvature_prop": self._get_grads_lists_curvature_prop,
        "exact": self._get_grads_lists_exact
    }
    self._colocate_gradients_with_ops = colocate_gradients_with_ops

    self._made_vars = False
    self._exps = exps

    self._name = name

  @property
  def variables(self):
    return self._variables

  @property
  def damping(self):
    return self._damping

  @property
  def blocks(self):
    """All registered FisherBlocks."""
    return self._layers.get_blocks()

  @property
  def factors(self):
    """All registered FisherFactors."""
    return self._layers.get_factors()

  @property
  def name(self):
    return self._name

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
    return self.multiply_matpower(-1, vecs_and_vars)

  def multiply(self, vecs_and_vars):
    """Multiplies the vectors by the corresponding (damped) blocks.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """
    return self.multiply_matpower(1, vecs_and_vars)

  def multiply_matpower(self, exp, vecs_and_vars):
    """Multiplies the vecs by the corresponding matrix powers of the blocks.

    Args:
      exp: A float representing the power to raise the blocks by before
        multiplying it by the vector.
      vecs_and_vars: List of (vector, variable) pairs.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """
    fcn = lambda fb, vec: fb.multiply_matpower(vec, exp)
    return self._apply_transformation(vecs_and_vars, fcn)

  def _instantiate_factors(self):
    """Instantiates FisherFactors' variables.

    Raises:
      ValueError: If estimation_mode was improperly specified at construction.
    """
    blocks = self.blocks
    tensors_to_compute_grads = [
        block.tensors_to_compute_grads() for block in blocks
    ]

    try:
      grads_lists = self._gradient_fns[self._estimation_mode](
          tensors_to_compute_grads)
    except KeyError:
      raise ValueError("Unrecognized value {} for estimation_mode.".format(
          self._estimation_mode))

    for grads_list, block in zip(grads_lists, blocks):
      block.instantiate_factors(grads_list, self.damping)

  def _check_vars_unmade_and_set_made_flag(self):
    if self._made_vars:
      raise Exception("Already made variables.")
    self._made_vars = True

  def made_vars(self):
    return self._made_vars

  def _register_matrix_functions(self):
    for exp in self._exps:
      for block in self.blocks:
        block.register_matpower(exp)

  def _finalize_layer_collection(self):
    self._layers.create_subgraph()
    self._layers.check_registration(self.variables)
    self._instantiate_factors()
    self._register_matrix_functions()

  def make_ops_and_vars(self, scope=None):
    """Make ops and vars with no specific device placement.

    See make_ops_and_vars_round_robin for further details.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All variables will be created,
        and all ops will execute, inside of a variable scope of the given
        name. (Default: None)
    Returns:
      cov_update_ops: List of ops that compute the cov updates. Corresponds
        one-to-one with the list of factors given by the "factors" property.
      cov_update_op: cov_update_ops grouped into a single op.
      inv_update_ops: List of ops that compute the inv updates. Corresponds
        one-to-one with the list of factors given by the "factors" property.
      inv_update_op: inv_update_ops grouped into a single op.
      cov_update_thunks: Thunks that make the ops in cov_update_ops.
      inv_update_thunks: Thunks that make the ops in inv_update_ops.
    """
    return self.make_ops_and_vars_round_robin(scope=scope)

  # TODO(b/70674513): Factor device placement outside of this class.
  def make_ops_and_vars_round_robin(self, scope=None, cov_devices=None,
                                    inv_devices=None):
    """Make ops and vars with a round-robin device placement strategy.

    For each factor, all of that factor's cov variables and their associated
    update ops will be placed on a particular device.  A new device is chosen
    for each factor by cycling through list of devices in the cov_devices
    argument. If cov_devices is None then no explicit device placement occurs.

    An analogous strategy is followed for inverse update ops, with the list of
    devices being given by the inv_devices argument.

    Inverse variables on the other hand are not placed on any specific device
    (they will just use the current the device placement context, whatever
    that happens to be).  The idea is that the inverse variable belong where
    they will be accessed most often, which is the device that actually applies
    the preconditioner to the gradient. The user will be responsible for setting
    the device context for this.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All variables will be created,
        and all ops will execute, inside of a variable scope of the given
        name. (Default: None)
      cov_devices: Iterable of device strings (e.g. '/gpu:0'). Covariance
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.
      inv_devices: Iterable of device strings (e.g. '/gpu:0'). Inversion
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.

    Returns:
      cov_update_ops: List of ops that compute the cov updates. Corresponds
        one-to-one with the list of factors given by the "factors" property.
      cov_update_op: cov_update_ops grouped into a single op.
      inv_update_ops: List of ops that compute the inv updates. Corresponds
        one-to-one with the list of factors given by the "factors" property.
      inv_update_op: inv_update_ops grouped into a single op.
      cov_update_thunks: Thunks that make the ops in cov_update_ops.
      inv_update_thunks: Thunks that make the ops in inv_update_ops.
    """
    (cov_update_thunks,
     inv_update_thunks) = self.make_vars_and_create_op_thunks_round_robin(
         scope=scope,
         cov_devices=cov_devices,
         inv_devices=inv_devices)
    cov_update_ops = [thunk() for thunk in cov_update_thunks]
    inv_update_ops = [thunk() for thunk in inv_update_thunks]

    scope = self.name if scope is None else scope
    with variable_scope.variable_scope(scope):
      cov_update_op = control_flow_ops.group(cov_update_ops,
                                             name="cov_update_op")
      inv_update_op = control_flow_ops.group(inv_update_ops,
                                             name="inv_update_op")

    return (cov_update_ops, cov_update_op, inv_update_ops, inv_update_op,
            cov_update_thunks, inv_update_thunks)

  def make_vars_and_create_op_thunks_round_robin(self,
                                                 scope=None,
                                                 cov_devices=None,
                                                 inv_devices=None):
    """Make vars and create op thunks w/ a round-robin device placement strat.

    For each factor, all of that factor's cov variables and their associated
    update ops will be placed on a particular device.  A new device is chosen
    for each factor by cycling through list of devices in the cov_devices
    argument. If cov_devices is None then no explicit device placement occurs.

    An analogous strategy is followed for inverse update ops, with the list of
    devices being given by the inv_devices argument.

    Inverse variables on the other hand are not placed on any specific device
    (they will just use the current the device placement context, whatever
    that happens to be).  The idea is that the inverse variable belong where
    they will be accessed most often, which is the device that actually applies
    the preconditioner to the gradient. The user will be responsible for setting
    the device context for this.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All variables will be created,
        and all thunks will execute, inside of a variable scope of the given
        name. (Default: None)
      cov_devices: Iterable of device strings (e.g. '/gpu:0'). Covariance
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.
      inv_devices: Iterable of device strings (e.g. '/gpu:0'). Inversion
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.
    Returns:
      cov_update_thunks: List of cov update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
      inv_update_thunks: List of inv update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
    """

    (cov_variable_thunks_raw, cov_update_thunks_raw, inv_variable_thunks_raw,
     inv_update_thunks_raw) = self.create_ops_and_vars_thunks(scope=scope)

    if cov_devices:
      cov_update_thunks = []
      for cov_variable_thunk, cov_update_thunk, device in zip(
          cov_variable_thunks_raw, cov_update_thunks_raw,
          itertools.cycle(cov_devices)):
        with tf_ops.device(device):
          cov_variable_thunk()
        cov_update_thunks.append(_make_thunk_on_device(cov_update_thunk,
                                                       device))
    else:
      for cov_variable_thunk in cov_variable_thunks_raw:
        cov_variable_thunk()
      cov_update_thunks = cov_update_thunks_raw

    for inv_variable_thunk in inv_variable_thunks_raw:
      inv_variable_thunk()

    if inv_devices:
      inv_update_thunks = []
      for inv_update_thunk, device in zip(inv_update_thunks_raw,
                                          itertools.cycle(inv_devices)):
        inv_update_thunks.append(_make_thunk_on_device(inv_update_thunk,
                                                       device))
    else:
      inv_update_thunks = inv_update_thunks_raw

    return cov_update_thunks, inv_update_thunks

  def create_ops_and_vars_thunks(self, scope=None):
    """Create thunks that make the ops and vars on demand.

    This function returns 4 lists of thunks: cov_variable_thunks,
    cov_update_thunks, inv_variable_thunks, and inv_update_thunks.

    The length of each list is the number of factors and the i-th element of
    each list corresponds to the i-th factor (given by the "factors" property).

    Note that the execution of these thunks must happen in a certain
    partial order.  The i-th element of cov_variable_thunks must execute
    before the i-th element of cov_update_thunks (and also the i-th element
    of inv_update_thunks).  Similarly, the i-th element of inv_variable_thunks
    must execute before the i-th element of inv_update_thunks.

    TL;DR (oversimplified): Execute the thunks according to the order that
    they are returned.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All thunks will execute inside
        of a variable scope of the given name. (Default: None)
    Returns:
      cov_variable_thunks: A list of thunks that make the cov variables.
      cov_update_thunks: A list of thunks that make the cov update ops.
      inv_variable_thunks: A list of thunks that make the inv variables.
      inv_update_thunks: A list of thunks that make the inv update ops.
    """
    self._check_vars_unmade_and_set_made_flag()

    self._finalize_layer_collection()

    scope = self.name if scope is None else scope

    cov_variable_thunks = [
        self._create_cov_variable_thunk(factor, scope)
        for factor in self.factors
    ]
    cov_update_thunks = [
        self._create_cov_update_thunk(factor, scope) for factor in self.factors
    ]
    inv_variable_thunks = [
        self._create_inv_variable_thunk(factor, scope)
        for factor in self.factors
    ]
    inv_update_thunks = [
        self._create_inv_update_thunk(factor, scope) for factor in self.factors
    ]

    return (cov_variable_thunks, cov_update_thunks,
            inv_variable_thunks, inv_update_thunks)

  def _create_cov_variable_thunk(self, factor, scope):
    """Constructs a covariance variable thunk for a single FisherFactor."""

    def thunk():
      with variable_scope.variable_scope(scope):
        return factor.instantiate_cov_variables()

    return thunk

  def _create_cov_update_thunk(self, factor, scope):
    """Constructs a covariance update thunk for a single FisherFactor."""

    def thunk():
      with variable_scope.variable_scope(scope):
        return factor.make_covariance_update_op(self._cov_ema_decay)

    return thunk

  def _create_inv_variable_thunk(self, factor, scope):
    """Constructs a inverse variable thunk for a single FisherFactor."""

    def thunk():
      with variable_scope.variable_scope(scope):
        return factor.instantiate_inv_variables()

    return thunk

  def _create_inv_update_thunk(self, factor, scope):
    """Constructs an inverse update thunk for a single FisherFactor."""

    def thunk():
      with variable_scope.variable_scope(scope):
        return control_flow_ops.group(factor.make_inverse_update_ops())

    return thunk

  def _get_grads_lists_gradients(self, tensors):
    # Passing in a list of loss values is better than passing in the sum as
    # the latter creates unnessesary ops on the default device
    grads_flat = gradients_impl.gradients(
        self._layers.eval_losses_on_samples(),
        nest.flatten(tensors),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_grads_lists_empirical(self, tensors):
    # Passing in a list of loss values is better than passing in the sum as
    # the latter creates unnessesary ops on the default device
    grads_flat = gradients_impl.gradients(
        self._layers.eval_losses(),
        nest.flatten(tensors),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_transformed_random_signs(self):
    transformed_random_signs = []
    for loss in self._layers.losses:
      with tf_ops.colocate_with(self._layers.loss_colocation_ops[loss]):
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
      with tf_ops.colocate_with(self._layers.loss_colocation_ops[loss]):
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
