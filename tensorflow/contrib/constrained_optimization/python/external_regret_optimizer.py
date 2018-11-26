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
"""Defines `AdditiveExternalRegretOptimizer`.

This optimizer minimizes a `ConstrainedMinimizationProblem` by introducing
Lagrange multipliers, and using `tf.train.Optimizer`s to jointly optimize over
the model parameters and Lagrange multipliers.

For the purposes of constrained optimization, at least in theory,
external-regret minimization suffices if the `ConstrainedMinimizationProblem`
we're optimizing doesn't have any `proxy_constraints`, while swap-regret
minimization should be used if `proxy_constraints` are present.

For more specifics, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization".
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

The formulation used by the AdditiveExternalRegretOptimizer--which is simply the
usual Lagrangian formulation--can be found in Definition 1, and is discussed in
Section 3. This optimizer is most similar to Algorithm 3 in Appendix C.3, with
the two differences being that it uses proxy constraints (if they're provided)
in the update of the model parameters, and uses `tf.train.Optimizer`s, instead
of SGD, for the "inner" updates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.constrained_optimization.python import constrained_optimizer

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer as train_optimizer


def _project_multipliers_wrt_euclidean_norm(multipliers, radius):
  """Projects its argument onto the feasible region.

  The feasible region is the set of all vectors with nonnegative elements that
  sum to at most `radius`.

  Args:
    multipliers: 1d tensor, the Lagrange multipliers to project.
    radius: float, the radius of the feasible region.

  Returns:
    The 1d tensor that results from projecting `multipliers` onto the feasible
      region w.r.t. the Euclidean norm.

  Raises:
    ValueError: if the `multipliers` tensor is not floating-point, does not have
      a fully-known shape, or is not one-dimensional.
  """
  if not multipliers.dtype.is_floating:
    raise ValueError("multipliers must have a floating-point dtype")
  multipliers_shape = multipliers.get_shape()
  if multipliers_shape.ndims is None:
    raise ValueError("multipliers must have known shape")
  if multipliers_shape.ndims != 1:
    raise ValueError(
        "multipliers must be one dimensional (instead is %d-dimensional)" %
        multipliers_shape.ndims)
  dimension = multipliers_shape.dims[0].value
  if dimension is None:
    raise ValueError("multipliers must have fully-known shape")

  def while_loop_condition(iteration, multipliers, inactive, old_inactive):
    """Returns false if the while loop should terminate."""
    del multipliers  # Needed by the body, but not the condition.
    not_done = (iteration < dimension)
    not_converged = standard_ops.reduce_any(
        standard_ops.not_equal(inactive, old_inactive))
    return standard_ops.logical_and(not_done, not_converged)

  def while_loop_body(iteration, multipliers, inactive, old_inactive):
    """Performs one iteration of the projection."""
    del old_inactive  # Needed by the condition, but not the body.
    iteration += 1
    scale = standard_ops.minimum(
        0.0,
        (radius - standard_ops.reduce_sum(multipliers)) / standard_ops.maximum(
            1.0, standard_ops.reduce_sum(inactive)))
    multipliers = multipliers + (scale * inactive)
    new_inactive = standard_ops.cast(multipliers > 0, multipliers.dtype)
    multipliers = multipliers * new_inactive
    return (iteration, multipliers, new_inactive, inactive)

  iteration = standard_ops.constant(0)
  inactive = standard_ops.ones_like(multipliers, dtype=multipliers.dtype)

  # We actually want a do-while loop, so we explicitly call while_loop_body()
  # once before tf.while_loop().
  iteration, multipliers, inactive, old_inactive = while_loop_body(
      iteration, multipliers, inactive, inactive)
  iteration, multipliers, inactive, old_inactive = control_flow_ops.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, multipliers, inactive, old_inactive),
      name="euclidean_projection")

  return multipliers


@six.add_metaclass(abc.ABCMeta)
class _ExternalRegretOptimizer(constrained_optimizer.ConstrainedOptimizer):
  """Base class representing an `_ExternalRegretOptimizer`.

  This class contains most of the logic for performing constrained
  optimization, minimizing external regret for the constraints player. What it
  *doesn't* do is keep track of the internal state (the Lagrange multipliers).
  Instead, the state is accessed via the _initial_state(),
  _lagrange_multipliers(), _constraint_grad_and_var() and _projection_op()
  methods.

  The reason for this is that we want to make it easy to implement different
  representations of the internal state.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by `_ExternalRegretOptimizer`s--which is simply the usual
  Lagrangian formulation--can be found in Definition 1, and is discussed in
  Section 3. Such optimizers are most similar to Algorithm 3 in Appendix C.3.
  """

  def __init__(self, optimizer, constraint_optimizer=None):
    """Constructs a new `_ExternalRegretOptimizer`.

    The difference between `optimizer` and `constraint_optimizer` (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the Lagrange multipliers. If no
    `constraint_optimizer` is provided, then `optimizer` is used for both.

    Args:
      optimizer: tf.train.Optimizer, used to optimize the objective and
        proxy_constraints portion of the ConstrainedMinimizationProblem. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers.
      constraint_optimizer: optional tf.train.Optimizer, used to optimize the
        Lagrange multipliers.

    Returns:
      A new `_ExternalRegretOptimizer`.
    """
    super(_ExternalRegretOptimizer, self).__init__(optimizer=optimizer)
    self._constraint_optimizer = constraint_optimizer

  @property
  def constraint_optimizer(self):
    """Returns the `tf.train.Optimizer` used for the Lagrange multipliers."""
    return self._constraint_optimizer

  @abc.abstractmethod
  def _initial_state(self, num_constraints):
    pass

  @abc.abstractmethod
  def _lagrange_multipliers(self, state):
    pass

  @abc.abstractmethod
  def _constraint_grad_and_var(self, state, gradient):
    pass

  @abc.abstractmethod
  def _projection_op(self, state, name=None):
    pass

  def _minimize_constrained(self,
                            minimization_problem,
                            global_step=None,
                            var_list=None,
                            gate_gradients=train_optimizer.Optimizer.GATE_OP,
                            aggregation_method=None,
                            colocate_gradients_with_ops=False,
                            name=None,
                            grad_loss=None):
    """Returns an `Operation` for minimizing the constrained problem.

    The `optimizer` constructor parameter will be used to update the model
    parameters, while the Lagrange multipliers will be updated using
    `constrained_optimizer` (if provided) or `optimizer` (if not).

    Args:
      minimization_problem: ConstrainedMinimizationProblem, the problem to
        optimize.
      global_step: as in `tf.train.Optimizer`'s `minimize` method.
      var_list: as in `tf.train.Optimizer`'s `minimize` method.
      gate_gradients: as in `tf.train.Optimizer`'s `minimize` method.
      aggregation_method: as in `tf.train.Optimizer`'s `minimize` method.
      colocate_gradients_with_ops: as in `tf.train.Optimizer`'s `minimize`
        method.
      name: as in `tf.train.Optimizer`'s `minimize` method.
      grad_loss: as in `tf.train.Optimizer`'s `minimize` method.

    Raises:
      ValueError: If the minimization_problem tensors have different dtypes.

    Returns:
      `Operation`, the train_op.
    """
    objective = minimization_problem.objective

    constraints = minimization_problem.constraints
    proxy_constraints = minimization_problem.proxy_constraints
    if proxy_constraints is None:
      proxy_constraints = constraints

    # Make sure that the objective, constraints and proxy constraints all have
    # the same dtype.
    if (objective.dtype.base_dtype != constraints.dtype.base_dtype or
        objective.dtype.base_dtype != proxy_constraints.dtype.base_dtype):
      raise ValueError("objective, constraints and proxy_constraints must "
                       "have the same dtype")

    # Flatten both constraints tensors to 1d.
    num_constraints = minimization_problem.num_constraints
    constraints = standard_ops.reshape(constraints, shape=(num_constraints,))
    proxy_constraints = standard_ops.reshape(
        proxy_constraints, shape=(num_constraints,))

    # We use a lambda to initialize the state so that, if this function call is
    # inside the scope of a tf.control_dependencies() block, the dependencies
    # will not be applied to the initializer.
    state = standard_ops.Variable(
        lambda: self._initial_state(num_constraints),
        trainable=False,
        name="external_regret_optimizer_state")

    multipliers = self._lagrange_multipliers(state)
    loss = (
        objective + standard_ops.tensordot(
            standard_ops.cast(multipliers, proxy_constraints.dtype),
            proxy_constraints, 1))
    multipliers_gradient = standard_ops.cast(constraints, multipliers.dtype)

    update_ops = []
    if self.constraint_optimizer is None:
      # If we don't have a separate constraint_optimizer, then we use
      # self._optimizer for both the update of the model parameters, and that of
      # the internal state.
      grads_and_vars = self.optimizer.compute_gradients(
          loss,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          grad_loss=grad_loss)
      grads_and_vars.append(
          self._constraint_grad_and_var(state, multipliers_gradient))
      update_ops.append(
          self.optimizer.apply_gradients(grads_and_vars, name="update"))
    else:
      # If we have a separate constraint_optimizer, then we use self._optimizer
      # for the update of the model parameters, and self._constraint_optimizer
      # for that of the internal state.
      grads_and_vars = self.optimizer.compute_gradients(
          loss,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          grad_loss=grad_loss)
      multiplier_grads_and_vars = [
          self._constraint_grad_and_var(state, multipliers_gradient)
      ]

      gradients = [
          gradient for gradient, _ in grads_and_vars + multiplier_grads_and_vars
          if gradient is not None
      ]
      with ops.control_dependencies(gradients):
        update_ops.append(
            self.optimizer.apply_gradients(grads_and_vars, name="update"))
        update_ops.append(
            self.constraint_optimizer.apply_gradients(
                multiplier_grads_and_vars, name="optimizer_state_update"))

    with ops.control_dependencies(update_ops):
      if global_step is None:
        # If we don't have a global step, just project, and we're done.
        return self._projection_op(state, name=name)
      else:
        # If we have a global step, then we need to increment it in addition to
        # projecting.
        projection_op = self._projection_op(state, name="project")
        with ops.colocate_with(global_step):
          global_step_op = state_ops.assign_add(
              global_step, 1, name="global_step_increment")
        return control_flow_ops.group(projection_op, global_step_op, name=name)


class AdditiveExternalRegretOptimizer(_ExternalRegretOptimizer):
  """A `ConstrainedOptimizer` based on external-regret minimization.

  This `ConstrainedOptimizer` uses the given `tf.train.Optimizer`s to jointly
  minimize over the model parameters, and maximize over Lagrange multipliers,
  with the latter maximization using additive updates and an algorithm that
  minimizes external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer--which is simply the usual Lagrangian
  formulation--can be found in Definition 1, and is discussed in Section 3. It
  is most similar to Algorithm 3 in Appendix C.3, with the two differences being
  that it uses proxy constraints (if they're provided) in the update of the
  model parameters, and uses `tf.train.Optimizer`s, instead of SGD, for the
  "inner" updates.
  """

  def __init__(self,
               optimizer,
               constraint_optimizer=None,
               maximum_multiplier_radius=None):
    """Constructs a new `AdditiveExternalRegretOptimizer`.

    Args:
      optimizer: tf.train.Optimizer, used to optimize the objective and
        proxy_constraints portion of ConstrainedMinimizationProblem. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multipliers.
      constraint_optimizer: optional tf.train.Optimizer, used to optimize the
        Lagrange multipliers.
      maximum_multiplier_radius: float, an optional upper bound to impose on the
        sum of the Lagrange multipliers.

    Returns:
      A new `AdditiveExternalRegretOptimizer`.

    Raises:
      ValueError: If the maximum_multiplier_radius parameter is nonpositive.
    """
    super(AdditiveExternalRegretOptimizer, self).__init__(
        optimizer=optimizer, constraint_optimizer=constraint_optimizer)

    if maximum_multiplier_radius and (maximum_multiplier_radius <= 0.0):
      raise ValueError("maximum_multiplier_radius must be strictly positive")

    self._maximum_multiplier_radius = maximum_multiplier_radius

  def _initial_state(self, num_constraints):
    # For an AdditiveExternalRegretOptimizer, the internal state is simply a
    # tensor of Lagrange multipliers with shape (m,), where m is the number of
    # constraints.
    #
    # FUTURE WORK: make the dtype a parameter.
    return standard_ops.zeros((num_constraints,), dtype=dtypes.float32)

  def _lagrange_multipliers(self, state):
    return state

  def _constraint_grad_and_var(self, state, gradient):
    # TODO(acotter): tf.colocate_with(), if colocate_gradients_with_ops is True?
    return (-gradient, state)

  def _projection_op(self, state, name=None):
    with ops.colocate_with(state):
      if self._maximum_multiplier_radius:
        projected_multipliers = _project_multipliers_wrt_euclidean_norm(
            state, self._maximum_multiplier_radius)
      else:
        projected_multipliers = standard_ops.maximum(state, 0.0)
      return state_ops.assign(state, projected_multipliers, name=name)
