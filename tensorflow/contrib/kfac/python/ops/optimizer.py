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
"""The KFAC optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

# pylint disable=long-line
from tensorflow.contrib.kfac.python.ops import curvature_matrix_vector_products as cmvp
from tensorflow.contrib.kfac.python.ops import estimator as est
# pylint enable=long-line

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import gradient_descent


class KfacOptimizer(gradient_descent.GradientDescentOptimizer):
  """The KFAC Optimizer (https://arxiv.org/abs/1503.05671)."""

  def __init__(self,
               learning_rate,
               cov_ema_decay,
               damping,
               layer_collection,
               var_list=None,
               momentum=0.9,
               momentum_type="regular",
               norm_constraint=None,
               name="KFAC",
               estimation_mode="gradients",
               colocate_gradients_with_ops=True,
               batch_size=None,
               cov_devices=None,
               inv_devices=None):
    """Initializes the KFAC optimizer with the given settings.

    Args:
      learning_rate: The base learning rate for the optimizer.  Should probably
          be set to 1.0 when using momentum_type = 'qmodel', but can still be
          set lowered if desired (effectively lowering the trust in the
          quadratic model.)
      cov_ema_decay: The decay factor used when calculating the covariance
          estimate moving averages.
      damping: The damping factor used to stabilize training due to errors in
          the local approximation with the Fisher information matrix, and to
          regularize the update direction by making it closer to the gradient.
          If damping is adapted during training then this value is used for
          initializing damping varaible.
          (Higher damping means the update looks more like a standard gradient
          update - see Tikhonov regularization.)
      layer_collection: The layer collection object, which holds the fisher
          blocks, kronecker factors, and losses associated with the
          graph.  The layer_collection cannot be modified after KfacOptimizer's
          initialization.
      var_list: Optional list or tuple of variables to train. Defaults to the
          list of variables collected in the graph under the key
          `GraphKeys.TRAINABLE_VARIABLES`.
      momentum: The momentum decay constant to use. Only applies when
          momentum_type is 'regular' or 'adam'. (Default: 0.9)
      momentum_type: The type of momentum to use in this optimizer, one of
          'regular', 'adam', or 'qmodel'. (Default: 'regular')
      norm_constraint: float or Tensor. If specified, the update is scaled down
          so that its approximate squared Fisher norm v^T F v is at most the
          specified value. May only be used with momentum type 'regular'.
          (Default: None)
      name: The name for this optimizer. (Default: 'KFAC')
      estimation_mode: The type of estimator to use for the Fishers.  Can be
          'gradients', 'empirical', 'curvature_propagation', or 'exact'.
          (Default: 'gradients'). See the doc-string for FisherEstimator for
          more a more detailed description of these options.
      colocate_gradients_with_ops: Whether we should request gradients we
          compute in the estimator be colocated with their respective ops.
          (Default: True)
      batch_size: The size of the mini-batch. Only needed when momentum_type
          == 'qmodel' or when automatic adjustment is used.  (Default: None)
      cov_devices: Iterable of device strings (e.g. '/gpu:0'). Covariance
          computations will be placed on these devices in a round-robin fashion.
          Can be None, which means that no devices are specified. Only used
          with (soon-to-be-depcrecated "convenience" properties).
      inv_devices: Iterable of device strings (e.g. '/gpu:0'). Inversion
          computations will be placed on these devices in a round-robin fashion.
          Can be None, which means that no devices are specified. Only used
          with (soon-to-be-depcrecated "convenience" properties).

    Raises:
      ValueError: If the momentum type is unsupported.
      ValueError: If clipping is used with momentum type other than 'regular'.
      ValueError: If no losses have been registered with layer_collection.
      ValueError: If momentum is non-zero and momentum_type is not 'regular'
          or 'adam'.
    """

    variables = var_list
    if variables is None:
      variables = tf_variables.trainable_variables()

    # Parameters to be passed to the Fisher estimator:
    self._variables = variables
    self._cov_ema_decay = cov_ema_decay
    self._layers = layer_collection
    self._estimation_mode = estimation_mode
    self._colocate_gradients_with_ops = colocate_gradients_with_ops
    self._cov_devices = cov_devices
    self._inv_devices = inv_devices

    # The below paramaters are required only if damping needs to be adapated.
    # These parameters can be set by calling
    # set_damping_adaptation_params() explicitly.
    self._damping_adaptation_decay = 0.95
    self._damping_adaptation_interval = 5
    # Check section 6.5 KFAC paper. omega(1) = pow(damping decay, interval)
    self._omega = (
        self._damping_adaptation_decay**self._damping_adaptation_interval)
    self._adapt_damping = False
    self._min_damping = 1e-5
    self._prev_train_batch = None
    self._is_chief = False
    self._loss_fn = None
    self._damping_constant = damping
    self._damping = None
    self._rho = None
    self._prev_loss = None
    self._q_model_change = None
    self._update_damping_op = None

    momentum_type = momentum_type.lower()
    legal_momentum_types = ["regular", "adam", "qmodel"]

    if momentum_type not in legal_momentum_types:
      raise ValueError("Unsupported momentum type {}. Must be one of {}."
                       .format(momentum_type, legal_momentum_types))
    if momentum_type != "regular" and norm_constraint is not None:
      raise ValueError("Update clipping is only supported with momentum "
                       "type 'regular'.")
    if momentum_type not in ["regular", "adam"] and momentum != 0:
      raise ValueError("Momentum must be unspecified if using a momentum_type "
                       "other than 'regular' or 'adam'.")

    # Extra parameters of the optimizer
    self._momentum = momentum
    self._momentum_type = momentum_type
    self._norm_constraint = norm_constraint
    self._batch_size = batch_size

    with variable_scope.variable_scope(name):
      self._fisher_est = est.FisherEstimator(
          self._variables,
          self._cov_ema_decay,
          self.damping,
          self._layers,
          exps=(-1,),
          estimation_mode=self._estimation_mode,
          colocate_gradients_with_ops=self._colocate_gradients_with_ops)

    super(KfacOptimizer, self).__init__(learning_rate, name=name)

  def set_damping_adaptation_params(self,
                                    is_chief,
                                    prev_train_batch,
                                    loss_fn,
                                    min_damping=1e-5,
                                    damping_adaptation_decay=0.99,
                                    damping_adaptation_interval=5):
    """Sets parameters required to adapt damping during training.

    When called, enables damping adaptation according to the Levenberg-Marquardt
    style rule described in Section 6.5 of "Optimizing Neural Networks with
    Kronecker-factored Approximate Curvature".

    Note that this function creates Tensorflow variables which store a few
    scalars and are accessed by the ops which update the damping (as part
    of the training op returned by the minimize() method).

    Args:
      is_chief: `Boolean`, `True` if the worker is chief.
      prev_train_batch: Training data used to minimize loss in the previous
        step. This will be used to evaluate loss by calling
        `loss_fn(prev_train_batch)`.
      loss_fn: `function` that takes as input training data tensor and returns
        a scalar loss.
      min_damping: `float`(Optional), Minimum value the damping parameter
        can take. Default value 1e-5.
      damping_adaptation_decay: `float`(Optional), The `damping` parameter is
        multipled by the `damping_adaptation_decay` every
        `damping_adaptation_interval` number of iterations. Default value 0.99.
      damping_adaptation_interval: `int`(Optional), Number of steps in between
        updating the `damping` parameter. Default value 5.

    Raises:
      ValueError: If `set_damping_adaptation_params` is already called and the
        the `adapt_damping` is `True`.
    """
    if self._adapt_damping:
      raise ValueError("Damping adaptation parameters already set.")

    with variable_scope.variable_scope(self.get_name()):
      self._adapt_damping = True
      self._is_chief = is_chief
      self._prev_train_batch = prev_train_batch
      self._loss_fn = loss_fn
      self._damping_adaptation_decay = damping_adaptation_decay
      self._damping_adaptation_interval = damping_adaptation_interval
      self._omega = (
          self._damping_adaptation_decay**self._damping_adaptation_interval)
      self._min_damping = min_damping

      self._rho = variable_scope.get_variable(
          "rho", shape=(), dtype=dtypes.float32, trainable=False)  # LM ratio.
      self._prev_loss = variable_scope.get_variable(
          "prev_loss", shape=(), dtype=dtypes.float32, trainable=False)
      self._q_model_change = variable_scope.get_variable(
          "q_model_change", shape=(), dtype=dtypes.float32, trainable=False)
      self._damping = variable_scope.get_variable(
          "damping", initializer=self._damping_constant, trainable=False)

  @property
  def cov_update_thunks(self):
    self._maybe_make_and_save_everything()
    return self._cov_update_thunks

  @property
  def cov_update_ops(self):
    self._maybe_make_and_save_everything()
    return self._cov_update_ops

  @property
  def cov_update_op(self):
    self._maybe_make_and_save_everything()
    return self._cov_update_op

  @property
  def inv_update_thunks(self):
    self._maybe_make_and_save_everything()
    return self._inv_update_thunks

  @property
  def inv_update_ops(self):
    self._maybe_make_and_save_everything()
    return self._inv_update_ops

  @property
  def inv_update_op(self):
    self._maybe_make_and_save_everything()
    return self._inv_update_op

  @property
  def variables(self):
    return self._variables

  @property
  def damping(self):
    if self._damping:
      return self._damping
    else:
      return self._damping_constant

  @property
  def damping_adaptation_interval(self):
    return self._damping_adaptation_interval

  def _maybe_make_and_save_everything(self):
    if not self._fisher_est.made_vars():
      warnings.warn("These convenience properties will be depcrecated soon. "
                    "Please use explicit op/thunk creation methods instead "
                    "(e.g. make_ops_and_vars_round_robin, etc).",
                    DeprecationWarning)
      (self._cov_update_ops, self._cov_update_op, self._inv_update_ops,
       self._inv_update_op, self._cov_update_thunks,
       self._inv_update_thunks) = self.make_ops_and_vars_round_robin(
           cov_devices=self._cov_devices,
           inv_devices=self._inv_devices)

  def make_ops_and_vars(self):
    """Make ops and vars with no specific device placement.

    See make_ops_and_vars_round_robin for details.

    Returns:
      cov_update_ops: List of ops that compute the cov updates. Corresponds
        one-to-one with the list of factors given by the "factors" property.
      cov_update_op: cov_update_ops grouped into a single op.
      inv_update_ops: List of ops that compute the inv updates. Corresponds
        one-to-one with the list of factors given by the "factors" property.
      cov_update_op: cov_update_ops grouped into a single op.
      inv_update_op: inv_update_ops grouped into a single op.
    """
    with variable_scope.variable_scope(self.get_name()):
      return self._fisher_est.make_ops_and_vars()

  def make_ops_and_vars_round_robin(self, cov_devices=None, inv_devices=None):
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
      cov_update_op: cov_update_ops grouped into a single op.
      inv_update_op: inv_update_ops grouped into a single op.
      cov_update_thunks: Thunks that make the ops in cov_update_ops.
      inv_update_thunks: Thunks that make the ops in inv_update_ops.
    """
    with variable_scope.variable_scope(self.get_name()):
      return self._fisher_est.make_ops_and_vars_round_robin(
          cov_devices=cov_devices, inv_devices=inv_devices)

  def make_vars_and_create_op_thunks_round_robin(self,
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
    scope = self.get_name() + "/" + self._fisher_est.name
    return self._fisher_est.make_vars_and_create_op_thunks_round_robin(
        scope=scope, cov_devices=cov_devices, inv_devices=inv_devices)

  def ops_and_vars_thunks(self):
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

    Returns:
      cov_variable_thunks: A list of thunks that make the cov variables.
      cov_update_thunks: A list of thunks that make the cov update ops.
      inv_variable_thunks: A list of thunks that make the inv variables.
      inv_update_thunks: A list of thunks that make the inv update ops.
    """
    scope = self.get_name() + "/" + self._fisher_est.name
    return self._fisher_est.ops_and_vars_thunks(scope=scope)

  def minimize(self, *args, **kwargs):
    # Should this variable scope encompass everything below?  Or will the super-
    # class make another copy of the same name scope?
    with variable_scope.variable_scope(self.get_name()):
      kwargs["var_list"] = kwargs.get("var_list") or self.variables
      if set(kwargs["var_list"]) != set(self.variables):
        raise ValueError("var_list doesn't match with set of Fisher-estimating "
                         "variables.")
      if self._adapt_damping and self._is_chief:
        global_step = kwargs.get("global_step", None)
        if not global_step:
          raise KeyError("global_step needs to be passed to optimizer.minimize "
                         "if damping parameter is adapted.")
        update_damping_op = self._update_damping(self._prev_train_batch,
                                                 global_step)
        with ops.control_dependencies([update_damping_op]):
          loss = args[0]
          loss_assign_op = state_ops.assign(self._prev_loss, loss)
          train_op = super(KfacOptimizer, self).minimize(*args, **kwargs)
          return control_flow_ops.group(loss_assign_op, train_op)
      else:
        return super(KfacOptimizer, self).minimize(*args, **kwargs)

  def compute_gradients(self, *args, **kwargs):
    # args[1] could be our var_list
    if len(args) > 1:
      var_list = args[1]
    else:
      kwargs["var_list"] = kwargs.get("var_list") or self.variables
      var_list = kwargs["var_list"]
    if set(var_list) != set(self.variables):
      raise ValueError("var_list doesn't match with set of Fisher-estimating "
                       "variables.")
    return super(KfacOptimizer, self).compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    """Applies gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      *args: Additional arguments for super.apply_gradients.
      **kwargs: Additional keyword arguments for super.apply_gradients.

    Returns:
      An `Operation` that applies the specified gradients.
    """
    self._maybe_make_and_save_everything()

    # In Python 3, grads_and_vars can be a zip() object which can only be
    # iterated over once. By converting it to a list, we ensure that it can be
    # iterated over more than once.
    grads_and_vars = list(grads_and_vars)

    # Compute step.
    steps_and_vars = self._compute_update_steps(grads_and_vars)

    # Update trainable variables with this step.
    return super(KfacOptimizer, self).apply_gradients(steps_and_vars, *args,
                                                      **kwargs)

  def _squared_fisher_norm(self, grads_and_vars, precon_grads_and_vars):
    """Computes the squared (approximate) Fisher norm of the updates.

    This is defined as v^T F v, where F is the approximate Fisher matrix
    as computed by the estimator, and v = F^{-1} g, where g is the gradient.
    This is computed efficiently as v^T g.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      precon_grads_and_vars: List of (preconditioned gradient, variable) pairs.
        Must be the result of calling `self._fisher_est.multiply_inverse`
        on `grads_and_vars`.

    Returns:
      Scalar representing the squared norm.

    Raises:
      ValueError: if the two list arguments do not contain the same variables,
        in the same order.
    """
    for (_, gvar), (_, pgvar) in zip(grads_and_vars, precon_grads_and_vars):
      if gvar is not pgvar:
        raise ValueError("The variables referenced by the two arguments "
                         "must match.")
    terms = [
        math_ops.reduce_sum(grad * pgrad)
        for (grad, _), (pgrad, _) in zip(grads_and_vars, precon_grads_and_vars)
    ]
    return math_ops.reduce_sum(terms)

  def _update_clip_coeff(self, grads_and_vars, precon_grads_and_vars):
    """Computes the scale factor for the update to satisfy the norm constraint.

    Defined as min(1, sqrt(c / r^T F r)), where c is the norm constraint,
    F is the approximate Fisher matrix, and r is the update vector, i.e.
    -alpha * v, where alpha is the learning rate, and v is the preconditioned
    gradient.

    This is based on Section 5 of Ba et al., Distributed Second-Order
    Optimization using Kronecker-Factored Approximations. Note that they
    absorb the learning rate alpha (which they denote eta_max) into the formula
    for the coefficient, while in our implementation, the rescaling is done
    before multiplying by alpha. Hence, our formula differs from theirs by a
    factor of alpha.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      precon_grads_and_vars: List of (preconditioned gradient, variable) pairs.
        Must be the result of calling `self._fisher_est.multiply_inverse`
        on `grads_and_vars`.

    Returns:
      Scalar representing the coefficient which should be applied to the
      preconditioned gradients to satisfy the norm constraint.
    """
    sq_norm_grad = self._squared_fisher_norm(grads_and_vars,
                                             precon_grads_and_vars)
    sq_norm_up = sq_norm_grad * self._learning_rate**2
    return math_ops.minimum(1.,
                            math_ops.sqrt(self._norm_constraint / sq_norm_up))

  def _clip_updates(self, grads_and_vars, precon_grads_and_vars):
    """Rescales the preconditioned gradients to satisfy the norm constraint.

    Rescales the preconditioned gradients such that the resulting update r
    (after multiplying by the learning rate) will satisfy the norm constraint.
    This constraint is that r^T F r <= C, where F is the approximate Fisher
    matrix, and C is the norm_constraint attribute. See Section 5 of
    Ba et al., Distributed Second-Order Optimization using Kronecker-Factored
    Approximations.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      precon_grads_and_vars: List of (preconditioned gradient, variable) pairs.
        Must be the result of calling `self._fisher_est.multiply_inverse`
        on `grads_and_vars`.

    Returns:
      List of (rescaled preconditioned gradient, variable) pairs.
    """
    coeff = self._update_clip_coeff(grads_and_vars, precon_grads_and_vars)
    return [(pgrad * coeff, var) for pgrad, var in precon_grads_and_vars]

  def _compute_prev_updates(self, variables):
    """Computes previous updates as negative velocities scaled by learning rate.

    Args:
      variables: List of variables in the graph that the update will be
          applied to.

    Returns:
      List of previous updates applied to the `variables`.
    """
    return list(
        -1 * self._learning_rate * self._zeros_slot(var, "velocity", self._name)
        for var in variables)

  def _compute_qmodel_hyperparams(self, precon_grads, prev_updates, grads,
                                  variables):
    """Compute optimal update hyperparameters from the quadratic model.

    More specifically, if L is the loss we minimize a quadratic approximation
    of L(theta + d) which we denote by qmodel(d) with
    d = alpha*precon_grad + mu*prev_update with respect to alpha and mu, where

      qmodel(d) = (1/2) * d^T * B * d + grad^T*d + L(theta) .

    Unlike in the KL clipping approach we use the non-approximated quadratic
    model where the curvature matrix C is the true Fisher on the current
    mini-batch (computed without any approximations beyond mini-batch sampling),
    with the usual Tikhonov damping/regularization applied,

      C = F + damping * I

    See Section 7 of https://arxiv.org/abs/1503.05671 for a derivation of
    the formula.  See Appendix C for a discussion of the trick of using
    a factorized Fisher matrix to more efficiently compute the required
    vector-matrix-vector products.

    Note that the elements of all 4 lists passed to this function must
    be in correspondence with each other.

    Args:
      precon_grads: List of preconditioned gradients.
      prev_updates: List of updates computed at the previous iteration.
      grads: List of gradients.
      variables: List of variables in the graph that the update will be
          applied to. (Note that this function doesn't actually apply the
          update.)

    Returns:
      (alpha, mu, qmodel_change), where alpha and mu are chosen to optimize the
      quadratic model, and
      qmodel_change = qmodel(alpha*precon_grad + mu*prev_update) - qmodel(0)
                    = qmodel(alpha*precon_grad + mu*prev_update) - L(theta).
    """

    cmvpc = cmvp.CurvatureMatrixVectorProductComputer(self._layers.losses,
                                                      variables)

    # compute the matrix-vector products with the transposed Fisher factor
    fft_precon_grads = cmvpc.multiply_fisher_factor_transpose(precon_grads)
    fft_prev_updates = cmvpc.multiply_fisher_factor_transpose(prev_updates)

    batch_size = math_ops.cast(
        self._batch_size, dtype=fft_precon_grads[0].dtype)

    # compute the entries of the 2x2 matrix
    m_11 = (
        _inner_product_list(fft_precon_grads, fft_precon_grads) / batch_size +
        self.damping * _inner_product_list(precon_grads, precon_grads))

    m_21 = (
        _inner_product_list(fft_prev_updates, fft_precon_grads) / batch_size +
        self.damping * _inner_product_list(prev_updates, precon_grads))

    m_22 = (
        _inner_product_list(fft_prev_updates, fft_prev_updates) / batch_size +
        self.damping * _inner_product_list(prev_updates, prev_updates))

    def non_zero_prevupd_case():
      r"""Computes optimal (alpha, mu) given non-zero previous update.

      We solve the full 2x2 linear system. See Martens & Grosse (2015),
      Section 7, definition of $\alpha^*$ and $\mu^*$.

      Returns:
        (alpha, mu, qmodel_change), where alpha and mu are chosen to optimize
        the quadratic model, and
        qmodel_change = qmodel(alpha*precon_grad + mu*prev_update) - qmodel(0).
      """
      m = ops.convert_to_tensor([[m_11, m_21], [m_21, m_22]])

      c = ops.convert_to_tensor([[_inner_product_list(grads, precon_grads)],
                                 [_inner_product_list(grads, prev_updates)]])

      sol = -1. * _two_by_two_solve(m, c)
      alpha = sol[0]
      mu = sol[1]
      qmodel_change = 0.5 * math_ops.reduce_sum(sol * c)

      return alpha, mu, qmodel_change

    def zero_prevupd_case():
      r"""Computes optimal (alpha, mu) given all-zero previous update.

      The linear system reduces to 1x1. See Martens & Grosse (2015),
      Section 6.4, definition of $\alpha^*$.

      Returns:
        (alpha, 0.0, qmodel_change), where alpha is chosen to optimize the
        quadratic model, and
        qmodel_change = qmodel(alpha*precon_grad) - qmodel(0)
      """
      m = m_11
      c = _inner_product_list(grads, precon_grads)

      alpha = -c / m
      mu = 0.0
      qmodel_change = 0.5 * alpha * c

      return alpha, mu, qmodel_change

    return control_flow_ops.cond(
        math_ops.equal(m_22, 0.0), zero_prevupd_case, non_zero_prevupd_case)

  def _assign_q_model_change(self, q_model_change):
    """Assigns `q_model_change` to `self._q_model_change` if damping is adapted.

    Note only the chief worker does the assignment.

    Args:
      q_model_change: Scalar tensor of type `float32`.

    Returns:
      If `adapt_damping` is `True` then returns an assign op, Otherwise returns
      a no_op().
    """
    if self._adapt_damping and self._is_chief:
      q_model_assign_op = state_ops.assign(self._q_model_change, q_model_change)
    else:
      q_model_assign_op = control_flow_ops.no_op()
    return q_model_assign_op

  def _compute_qmodel_hyperparams_wrapper(self, grads_and_vars,
                                          precon_grads_and_vars):
    """Wrapper function for `self._compute_qmodel_hyperparams`.

    Constructs a list of preconditioned gradients and variables. Also creates a
    op to asssign the computed q model change to `self._q_model_change`.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      precon_grads_and_vars: List of (preconditioned gradients, variable)
        pairs.

    Returns:
      (alpha, mu, q_model_assign_op), where alpha and mu are chosen to optimize
      the quadratic model, `q_model_assign_op` assigns the computed q model
      change to `self._q_model_change`.
    """
    precon_grads = list(
        precon_grad for (precon_grad, _) in precon_grads_and_vars)
    grads = list(grad for (grad, _) in grads_and_vars)
    variables = list(var for (_, var) in grads_and_vars)
    prev_updates = self._compute_prev_updates(variables)
    # Compute optimal velocity update parameters according to quadratic model
    alpha, mu, q_model_change = self._compute_qmodel_hyperparams(
        precon_grads, prev_updates, grads, variables)

    return alpha, mu, self._assign_q_model_change(q_model_change)

  def _compute_update_steps(self, grads_and_vars):
    """Computes the update steps for the variables given the gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      A list of tuple (assign_op ,var) where `assign_op` assigns the update
      steps to `var`.
    """

    if self._momentum_type == "regular":
      # Compute "preconditioned" gradient.
      precon_grads_and_vars = self._fisher_est.multiply_inverse(grads_and_vars)

      # Apply "KL clipping" if asked for.
      if self._norm_constraint is not None:
        precon_grads_and_vars = self._clip_updates(grads_and_vars,
                                                   precon_grads_and_vars)

      # Update the velocity with this and return it as the step.
      if self._adapt_damping and self._is_chief:
        _, _, q_model_assign_op = self._compute_qmodel_hyperparams_wrapper(
            grads_and_vars, precon_grads_and_vars)
        with ops.control_dependencies([q_model_assign_op]):
          return self._update_velocities(precon_grads_and_vars, self._momentum)
      else:
        return self._update_velocities(precon_grads_and_vars, self._momentum)
    elif self._momentum_type == "adam":
      # Update velocity.
      velocities_and_vars = self._update_velocities(grads_and_vars,
                                                    self._momentum)
      # Return "preconditioned" velocity vector as the step.
      return self._fisher_est.multiply_inverse(velocities_and_vars)

    elif self._momentum_type == "qmodel":
      # Compute "preconditioned" gradient.
      precon_grads_and_vars = self._fisher_est.multiply_inverse(grads_and_vars)

      # Compute optimal velocity update parameters according to quadratic model
      alpha, mu, q_model_assign_op = self._compute_qmodel_hyperparams_wrapper(
          grads_and_vars, precon_grads_and_vars)

      with ops.control_dependencies([q_model_assign_op]):
        return self._update_velocities(
            precon_grads_and_vars, mu, vec_coeff=-alpha)

  def _update_velocities(self, vecs_and_vars, decay, vec_coeff=1.0):
    """Updates the velocities of the variables with the given vectors.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.
      decay: How much to decay the old velocity by.  This is often referred to
        as the 'momentum constant'.
      vec_coeff: Coefficient to apply to the vectors before adding them to the
        velocity.

    Returns:
      A list of (velocity, var) indicating the new velocity for each var.
    """

    def _update_velocity(vec, var):
      velocity = self._zeros_slot(var, "velocity", self._name)
      with ops.colocate_with(velocity):
        # NOTE(mattjj): read/modify/write race condition not suitable for async.

        # Compute the new velocity for this variable.
        new_velocity = decay * velocity + vec_coeff * vec

        # Save the updated velocity.
        return (array_ops.identity(velocity.assign(new_velocity)), var)

    # Go through variable and update its associated part of the velocity vector.
    return [_update_velocity(vec, var) for vec, var in vecs_and_vars]

  # TODO(b/73448937): Move all update damping code to a separate class/function.
  def _update_damping(self, prev_batch, global_step):
    """Adapts damping parameter. Check KFAC (Section 6.5) for the details.

    The damping parameter is updated according to the Levenberg-Marquardt rule
    every `self._damping_adaptation_interval` iterations.

    Args:
      prev_batch: Tensor or tuple of tensors which can be passed to
        `self._loss_fn` to evaluate loss.
      global_step: `Variable` which keeps track of number of times the training
        variables have been updated.
    Returns:
      A `tf.cond` op which updates the damping parameter.
    """
    def compute_damping():
      """"Adapts damping parameter based on "reduction ratio".

      Reduction ratio captures how closely the quadratic approximation to the
      loss function approximates the actual loss within a trust region. The
      damping update tries to make the damping as small as possible while
      maintaining the property that the quadratic model remains a good local
      approximation to the loss function.

      Returns:
        An Op to assign newly computed damping value to `self._damping`.
      """
      prev_batch_loss = self._loss_fn(prev_batch)
      with ops.control_dependencies([prev_batch_loss]):
        rho_assign = self._rho.assign(
            (prev_batch_loss - self._prev_loss) / self._q_model_change)
        with ops.control_dependencies([rho_assign]):
          new_damping = control_flow_ops.case(
              [(self._rho < 0.25, lambda: self.damping / self._omega),
               (self._rho > 0.75, lambda: self.damping * self._omega)],
              lambda: self.damping)
          with ops.control_dependencies([new_damping]):
            new_damping_min = math_ops.maximum(new_damping, self._min_damping)
            return control_flow_ops.group(self._damping.assign(new_damping_min))

    return control_flow_ops.cond(
        math_ops.equal(
            math_ops.mod(global_step + 1, self._damping_adaptation_interval),
            0), compute_damping, control_flow_ops.no_op)


def _inner_product_list(list1, list2):
  return math_ops.add_n(
      [math_ops.reduce_sum(elt1 * elt2) for elt1, elt2 in zip(list1, list2)])


def _two_by_two_solve(m, c):
  # it might be better just to crank out the exact formula for 2x2 inverses
  return math_ops.matmul(linalg_ops.matrix_inverse(m), c)
