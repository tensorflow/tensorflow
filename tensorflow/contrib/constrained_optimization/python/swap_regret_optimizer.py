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
"""Defines `{Additive,Multiplicative}SwapRegretOptimizer`s.

These optimizers minimize a `ConstrainedMinimizationProblem` by using a
swap-regret minimizing algorithm (either SGD or multiplicative weights) to learn
what weights should be associated with the objective function and constraints.
These algorithms do *not* use Lagrange multipliers, but the idea is similar.
The main differences between the formulation used here, and the standard
Lagrangian formulation, are that (i) the objective function is weighted, in
addition to the constraints, and (ii) we learn a matrix of weights, instead of a
vector.

For the purposes of constrained optimization, at least in theory,
external-regret minimization suffices if the `ConstrainedMinimizationProblem`
we're optimizing doesn't have any `proxy_constraints`, while swap-regret
minimization should be used if `proxy_constraints` are present.

For more specifics, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization".
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

The formulation used by both of the SwapRegretOptimizers can be found in
Definition 2, and is discussed in Section 4. The
`MultiplicativeSwapRegretOptimizer` is most similar to Algorithm 2 in Section 4,
with the difference being that it uses `tf.train.Optimizer`s, instead of SGD,
for the "inner" updates. The `AdditiveSwapRegretOptimizer` differs further in
that it performs additive (instead of multiplicative) updates of the stochastic
matrix.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math

import six

from tensorflow.contrib.constrained_optimization.python import constrained_optimizer

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer as train_optimizer


def _maximal_eigenvector_power_method(matrix,
                                      epsilon=1e-6,
                                      maximum_iterations=100):
  """Returns the maximal right-eigenvector of `matrix` using the power method.

  Args:
    matrix: 2D Tensor, the matrix of which we will find the maximal
      right-eigenvector.
    epsilon: nonnegative float, if two iterations of the power method differ (in
      L2 norm) by no more than epsilon, we will terminate.
    maximum_iterations: nonnegative int, if we perform this many iterations, we
      will terminate.

  Result:
    The maximal right-eigenvector of `matrix`.

  Raises:
    ValueError: If the `matrix` tensor is not floating-point, or if the
      `epsilon` or `maximum_iterations` parameters violate their bounds.
  """
  if not matrix.dtype.is_floating:
    raise ValueError("multipliers must have a floating-point dtype")
  if epsilon <= 0.0:
    raise ValueError("epsilon must be strictly positive")
  if maximum_iterations <= 0:
    raise ValueError("maximum_iterations must be strictly positive")

  def while_loop_condition(iteration, eigenvector, old_eigenvector):
    """Returns false if the while loop should terminate."""
    not_done = (iteration < maximum_iterations)
    not_converged = (standard_ops.norm(eigenvector - old_eigenvector) > epsilon)
    return standard_ops.logical_and(not_done, not_converged)

  def while_loop_body(iteration, eigenvector, old_eigenvector):
    """Performs one iteration of the power method."""
    del old_eigenvector  # Needed by the condition, but not the body.
    iteration += 1
    # We need to use tf.matmul() and tf.expand_dims(), instead of
    # tf.tensordot(), since the former will infer the shape of the result, while
    # the latter will not (tf.while_loop() needs the shapes).
    new_eigenvector = standard_ops.matmul(
        matrix, standard_ops.expand_dims(eigenvector, 1))[:, 0]
    new_eigenvector /= standard_ops.norm(new_eigenvector)
    return (iteration, new_eigenvector, eigenvector)

  iteration = standard_ops.constant(0)
  eigenvector = standard_ops.ones_like(matrix[:, 0])
  eigenvector /= standard_ops.norm(eigenvector)

  # We actually want a do-while loop, so we explicitly call while_loop_body()
  # once before tf.while_loop().
  iteration, eigenvector, old_eigenvector = while_loop_body(
      iteration, eigenvector, eigenvector)
  iteration, eigenvector, old_eigenvector = control_flow_ops.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, eigenvector, old_eigenvector),
      name="power_method")

  return eigenvector


def _project_stochastic_matrix_wrt_euclidean_norm(matrix):
  """Projects its argument onto the set of left-stochastic matrices.

  This algorithm is O(n^3) at worst, where `matrix` is n*n. It can be done in
  O(n^2 * log(n)) time by sorting each column (and maybe better with a different
  algorithm), but the algorithm implemented here is easier to implement in
  TensorFlow.

  Args:
    matrix: 2d square tensor, the matrix to project.

  Returns:
    The 2d square tensor that results from projecting `matrix` onto the set of
      left-stochastic matrices w.r.t. the Euclidean norm applied column-wise
      (i.e. the Frobenius norm).

  Raises:
    ValueError: if the `matrix` tensor is not floating-point, does not have a
      fully-known shape, or is not two-dimensional and square.
  """
  if not matrix.dtype.is_floating:
    raise ValueError("multipliers must have a floating-point dtype")
  matrix_shape = matrix.get_shape()
  if matrix_shape.ndims is None:
    raise ValueError("matrix must have known shape")
  if matrix_shape.ndims != 2:
    raise ValueError(
        "matrix must be two dimensional (instead is %d-dimensional)" %
        matrix_shape.ndims)
  if matrix_shape[0] != matrix_shape[1]:
    raise ValueError("matrix must be square (instead has shape (%d,%d))" %
                     (matrix_shape[0], matrix_shape[1]))
  dimension = matrix_shape[0].value
  if dimension is None:
    raise ValueError("matrix must have fully-known shape")

  def while_loop_condition(iteration, matrix, inactive, old_inactive):
    """Returns false if the while loop should terminate."""
    del matrix  # Needed by the body, but not the condition.
    not_done = (iteration < dimension)
    not_converged = standard_ops.reduce_any(
        standard_ops.not_equal(inactive, old_inactive))
    return standard_ops.logical_and(not_done, not_converged)

  def while_loop_body(iteration, matrix, inactive, old_inactive):
    """Performs one iteration of the projection."""
    del old_inactive  # Needed by the condition, but not the body.
    iteration += 1
    scale = (1.0 - standard_ops.reduce_sum(
        matrix, axis=0, keepdims=True)) / standard_ops.maximum(
            1.0, standard_ops.reduce_sum(inactive, axis=0, keepdims=True))
    matrix = matrix + (scale * inactive)
    new_inactive = standard_ops.cast(matrix > 0, matrix.dtype)
    matrix = matrix * new_inactive
    return (iteration, matrix, new_inactive, inactive)

  iteration = standard_ops.constant(0)
  inactive = standard_ops.ones_like(matrix, dtype=matrix.dtype)

  # We actually want a do-while loop, so we explicitly call while_loop_body()
  # once before tf.while_loop().
  iteration, matrix, inactive, old_inactive = while_loop_body(
      iteration, matrix, inactive, inactive)
  iteration, matrix, inactive, old_inactive = control_flow_ops.while_loop(
      while_loop_condition,
      while_loop_body,
      loop_vars=(iteration, matrix, inactive, old_inactive),
      name="euclidean_projection")

  return matrix


def _project_log_stochastic_matrix_wrt_kl_divergence(log_matrix):
  """Projects its argument onto the set of log-left-stochastic matrices.

  Args:
    log_matrix: 2d square tensor, the element-wise logarithm of the matrix to
      project.

  Returns:
    The 2d square tensor that results from projecting exp(`matrix`) onto the set
      of left-stochastic matrices w.r.t. the KL-divergence applied column-wise.
  """

  # For numerical reasons, make sure that the largest matrix element is zero
  # before exponentiating.
  log_matrix = log_matrix - standard_ops.reduce_max(
      log_matrix, axis=0, keepdims=True)
  log_matrix = log_matrix - standard_ops.log(
      standard_ops.reduce_sum(
          standard_ops.exp(log_matrix), axis=0, keepdims=True))
  return log_matrix


@six.add_metaclass(abc.ABCMeta)
class _SwapRegretOptimizer(constrained_optimizer.ConstrainedOptimizer):
  """Base class representing a `_SwapRegretOptimizer`.

  This class contains most of the logic for performing constrained optimization,
  minimizing swap regret for the constraints player. What it *doesn't* do is
  keep track of the internal state (the stochastic matrix).  Instead, the state
  is accessed via the _initial_state(), _stochastic_matrix(),
  _constraint_grad_and_var() and _projection_op() methods.

  The reason for this is that we want to make it easy to implement different
  representations of the internal state. For example, for additive updates, it's
  most natural to store the stochastic matrix directly, whereas for
  multiplicative updates, it's most natural to store its element-wise logarithm.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by `_SwapRegretOptimizer`s can be found in Definition 2,
  and is discussed in Section 4. Such optimizers are most similar to Algorithm
  2 in Section 4. Most notably, the internal state is a left-stochastic matrix
  of shape (m+1,m+1), where m is the number of constraints.
  """

  def __init__(self, optimizer, constraint_optimizer=None):
    """Constructs a new `_SwapRegretOptimizer`.

    The difference between `optimizer` and `constraint_optimizer` (if the latter
    is provided) is that the former is used for learning the model parameters,
    while the latter us used for the update to the constraint/objective weight
    matrix (the analogue of Lagrange multipliers). If no `constraint_optimizer`
    is provided, then `optimizer` is used for both.

    Args:
      optimizer: tf.train.Optimizer, used to optimize the objective and
        proxy_constraints portion of ConstrainedMinimizationProblem. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multiplier analogues.
      constraint_optimizer: optional tf.train.Optimizer, used to optimize the
        Lagrange multiplier analogues.

    Returns:
      A new `_SwapRegretOptimizer`.
    """
    super(_SwapRegretOptimizer, self).__init__(optimizer=optimizer)
    self._constraint_optimizer = constraint_optimizer

  @property
  def constraint_optimizer(self):
    """Returns the `tf.train.Optimizer` used for the matrix."""
    return self._constraint_optimizer

  @abc.abstractmethod
  def _initial_state(self, num_constraints):
    pass

  @abc.abstractmethod
  def _stochastic_matrix(self, state):
    pass

  def _distribution(self, state):
    distribution = _maximal_eigenvector_power_method(
        self._stochastic_matrix(state))
    distribution = standard_ops.abs(distribution)
    distribution /= standard_ops.reduce_sum(distribution)
    return distribution

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
    parameters, while the constraint/objective weight matrix (the analogue of
    Lagrange multipliers) will be updated using `constrained_optimizer` (if
    provided) or `optimizer` (if not). Whether the matrix updates are additive
    or multiplicative depends on the derived class.

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
        name="swap_regret_optimizer_state")

    zero_and_constraints = standard_ops.concat(
        (standard_ops.zeros((1,), dtype=constraints.dtype), constraints),
        axis=0)
    objective_and_proxy_constraints = standard_ops.concat(
        (standard_ops.expand_dims(objective, 0), proxy_constraints), axis=0)

    distribution = self._distribution(state)
    loss = standard_ops.tensordot(
        standard_ops.cast(distribution, objective_and_proxy_constraints.dtype),
        objective_and_proxy_constraints, 1)
    matrix_gradient = standard_ops.matmul(
        standard_ops.expand_dims(
            standard_ops.cast(zero_and_constraints, distribution.dtype), 1),
        standard_ops.expand_dims(distribution, 0))

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
          self._constraint_grad_and_var(state, matrix_gradient))
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
      matrix_grads_and_vars = [
          self._constraint_grad_and_var(state, matrix_gradient)
      ]

      gradients = [
          gradient for gradient, _ in grads_and_vars + matrix_grads_and_vars
          if gradient is not None
      ]
      with ops.control_dependencies(gradients):
        update_ops.append(
            self.optimizer.apply_gradients(grads_and_vars, name="update"))
        update_ops.append(
            self.constraint_optimizer.apply_gradients(
                matrix_grads_and_vars, name="optimizer_state_update"))

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


class AdditiveSwapRegretOptimizer(_SwapRegretOptimizer):
  """A `ConstrainedOptimizer` based on swap-regret minimization.

  This `ConstrainedOptimizer` uses the given `tf.train.Optimizer`s to jointly
  minimize over the model parameters, and maximize over constraint/objective
  weight matrix (the analogue of Lagrange multipliers), with the latter
  maximization using additive updates and an algorithm that minimizes swap
  regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer can be found in Definition 2, and is
  discussed in Section 4. It is most similar to Algorithm 2 in Section 4, with
  the differences being that it uses `tf.train.Optimizer`s, instead of SGD, for
  the "inner" updates, and performs additive (instead of multiplicative) updates
  of the stochastic matrix.
  """

  def __init__(self, optimizer, constraint_optimizer=None):
    """Constructs a new `AdditiveSwapRegretOptimizer`.

    Args:
      optimizer: tf.train.Optimizer, used to optimize the objective and
        proxy_constraints portion of ConstrainedMinimizationProblem. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multiplier analogues.
      constraint_optimizer: optional tf.train.Optimizer, used to optimize the
        Lagrange multiplier analogues.

    Returns:
      A new `AdditiveSwapRegretOptimizer`.
    """
    # TODO(acotter): add a parameter determining the initial values of the
    # matrix elements (like initial_multiplier_radius in
    # MultiplicativeSwapRegretOptimizer).
    super(AdditiveSwapRegretOptimizer, self).__init__(
        optimizer=optimizer, constraint_optimizer=constraint_optimizer)

  def _initial_state(self, num_constraints):
    # For an AdditiveSwapRegretOptimizer, the internal state is a tensor of
    # shape (m+1,m+1), where m is the number of constraints, representing a
    # left-stochastic matrix.
    dimension = num_constraints + 1
    # Initialize by putting all weight on the objective, and none on the
    # constraints.
    return standard_ops.concat(
        (standard_ops.ones(
            (1, dimension)), standard_ops.zeros((dimension - 1, dimension))),
        axis=0)

  def _stochastic_matrix(self, state):
    return state

  def _constraint_grad_and_var(self, state, gradient):
    # TODO(acotter): tf.colocate_with(), if colocate_gradients_with_ops is True?
    return (-gradient, state)

  def _projection_op(self, state, name=None):
    with ops.colocate_with(state):
      return state_ops.assign(
          state,
          _project_stochastic_matrix_wrt_euclidean_norm(state),
          name=name)


class MultiplicativeSwapRegretOptimizer(_SwapRegretOptimizer):
  """A `ConstrainedOptimizer` based on swap-regret minimization.

  This `ConstrainedOptimizer` uses the given `tf.train.Optimizer`s to jointly
  minimize over the model parameters, and maximize over constraint/objective
  weight matrix (the analogue of Lagrange multipliers), with the latter
  maximization using multiplicative updates and an algorithm that minimizes swap
  regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer can be found in Definition 2, and is
  discussed in Section 4. It is most similar to Algorithm 2 in Section 4, with
  the difference being that it uses `tf.train.Optimizer`s, instead of SGD, for
  the "inner" updates.
  """

  def __init__(self,
               optimizer,
               constraint_optimizer=None,
               minimum_multiplier_radius=1e-3,
               initial_multiplier_radius=None):
    """Constructs a new `MultiplicativeSwapRegretOptimizer`.

    Args:
      optimizer: tf.train.Optimizer, used to optimize the objective and
        proxy_constraints portion of ConstrainedMinimizationProblem. If
        constraint_optimizer is not provided, this will also be used to optimize
        the Lagrange multiplier analogues.
      constraint_optimizer: optional tf.train.Optimizer, used to optimize the
        Lagrange multiplier analogues.
      minimum_multiplier_radius: float, each element of the matrix will be lower
        bounded by `minimum_multiplier_radius` divided by one plus the number of
        constraints.
      initial_multiplier_radius: float, the initial value of each element of the
        matrix associated with a constraint (i.e. excluding those elements
        associated with the objective) will be `initial_multiplier_radius`
        divided by one plus the number of constraints. Defaults to the value of
        `minimum_multiplier_radius`.

    Returns:
      A new `MultiplicativeSwapRegretOptimizer`.

    Raises:
      ValueError: If the two radius parameters are inconsistent.
    """
    super(MultiplicativeSwapRegretOptimizer, self).__init__(
        optimizer=optimizer, constraint_optimizer=constraint_optimizer)

    if (minimum_multiplier_radius <= 0.0) or (minimum_multiplier_radius >= 1.0):
      raise ValueError("minimum_multiplier_radius must be in the range (0,1)")
    if initial_multiplier_radius is None:
      initial_multiplier_radius = minimum_multiplier_radius
    elif (initial_multiplier_radius <
          minimum_multiplier_radius) or (minimum_multiplier_radius > 1.0):
      raise ValueError("initial_multiplier_radius must be in the range "
                       "[minimum_multiplier_radius,1]")

    self._minimum_multiplier_radius = minimum_multiplier_radius
    self._initial_multiplier_radius = initial_multiplier_radius

  def _initial_state(self, num_constraints):
    # For a MultiplicativeSwapRegretOptimizer, the internal state is a tensor of
    # shape (m+1,m+1), where m is the number of constraints, representing the
    # element-wise logarithm of a left-stochastic matrix.
    dimension = num_constraints + 1
    # Initialize by putting as much weight as possible on the objective, and as
    # little as possible on the constraints.
    log_initial_one = math.log(1.0 - (self._initial_multiplier_radius *
                                      (dimension - 1) / (dimension)))
    log_initial_zero = math.log(self._initial_multiplier_radius / dimension)
    # FUTURE WORK: make the dtype a parameter.
    return standard_ops.concat(
        (standard_ops.constant(
            log_initial_one, dtype=dtypes.float32, shape=(1, dimension)),
         standard_ops.constant(
             log_initial_zero,
             dtype=dtypes.float32,
             shape=(dimension - 1, dimension))),
        axis=0)

  def _stochastic_matrix(self, state):
    return standard_ops.exp(state)

  def _constraint_grad_and_var(self, state, gradient):
    # TODO(acotter): tf.colocate_with(), if colocate_gradients_with_ops is True?
    return (-gradient, state)

  def _projection_op(self, state, name=None):
    with ops.colocate_with(state):
      # Gets the dimension of the state (num_constraints + 1)--all of these
      # assertions are of things that should be impossible, since the state
      # passed into this method will have the same shape as that returned by
      # _initial_state().
      state_shape = state.get_shape()
      assert state_shape is not None
      assert state_shape.ndims == 2
      assert state_shape[0] == state_shape[1]
      dimension = state_shape[0].value
      assert dimension is not None

      minimum_log_multiplier = standard_ops.log(
          self._minimum_multiplier_radius / standard_ops.to_float(dimension))

      return state_ops.assign(
          state,
          standard_ops.maximum(
              _project_log_stochastic_matrix_wrt_kl_divergence(state),
              minimum_log_multiplier),
          name=name)
