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
"""Defines base class for `ConstrainedOptimizer`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.training import optimizer as train_optimizer


@six.add_metaclass(abc.ABCMeta)
class ConstrainedOptimizer(object):
  """Base class representing a constrained optimizer.

  A ConstrainedOptimizer wraps a tf.train.Optimizer (or more than one), and
  applies it to a ConstrainedMinimizationProblem. Unlike a tf.train.Optimizer,
  which takes a tensor to minimize as a parameter to its minimize() method, a
  constrained optimizer instead takes a ConstrainedMinimizationProblem.
  """

  def __init__(self, optimizer):
    """Constructs a new `ConstrainedOptimizer`.

    Args:
      optimizer: tf.train.Optimizer, used to optimize the
        ConstraintedMinimizationProblem.

    Returns:
      A new `ConstrainedOptimizer`.
    """
    self._optimizer = optimizer

  @property
  def optimizer(self):
    """Returns the `tf.train.Optimizer` used for optimization."""
    return self._optimizer

  def minimize_unconstrained(self,
                             minimization_problem,
                             global_step=None,
                             var_list=None,
                             gate_gradients=train_optimizer.Optimizer.GATE_OP,
                             aggregation_method=None,
                             colocate_gradients_with_ops=False,
                             name=None,
                             grad_loss=None):
    """Returns an `Op` for minimizing the unconstrained problem.

    Unlike `minimize_constrained`, this function ignores the `constraints` (and
    `proxy_constraints`) portion of the minimization problem entirely, and only
    minimizes `objective`.

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

    Returns:
      TensorFlow Op.
    """
    return self.optimizer.minimize(
        minimization_problem.objective,
        global_step=global_step,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        name=name,
        grad_loss=grad_loss)

  @abc.abstractmethod
  def minimize_constrained(self,
                           minimization_problem,
                           global_step=None,
                           var_list=None,
                           gate_gradients=train_optimizer.Optimizer.GATE_OP,
                           aggregation_method=None,
                           colocate_gradients_with_ops=False,
                           name=None,
                           grad_loss=None):
    """Returns an `Op` for minimizing the constrained problem.

    Unlike `minimize_unconstrained`, this function attempts to find a solution
    that minimizes the `objective` portion of the minimization problem while
    satisfying the `constraints` portion.

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

    Returns:
      TensorFlow Op.
    """
    pass

  def minimize(self,
               minimization_problem,
               unconstrained_steps=None,
               global_step=None,
               var_list=None,
               gate_gradients=train_optimizer.Optimizer.GATE_OP,
               aggregation_method=None,
               colocate_gradients_with_ops=False,
               name=None,
               grad_loss=None):
    """Returns an `Op` for minimizing the constrained problem.

    This method combines the functionality of `minimize_unconstrained` and
    `minimize_constrained`. If global_step < unconstrained_steps, it will
    perform an unconstrained update, and if global_step >= unconstrained_steps,
    it will perform a constrained update.

    The reason for this functionality is that it may be best to initialize the
    constrained optimizer with an approximate optimum of the unconstrained
    problem.

    Args:
      minimization_problem: ConstrainedMinimizationProblem, the problem to
        optimize.
      unconstrained_steps: int, number of steps for which we should perform
        unconstrained updates, before transitioning to constrained updates.
      global_step: as in `tf.train.Optimizer`'s `minimize` method.
      var_list: as in `tf.train.Optimizer`'s `minimize` method.
      gate_gradients: as in `tf.train.Optimizer`'s `minimize` method.
      aggregation_method: as in `tf.train.Optimizer`'s `minimize` method.
      colocate_gradients_with_ops: as in `tf.train.Optimizer`'s `minimize`
        method.
      name: as in `tf.train.Optimizer`'s `minimize` method.
      grad_loss: as in `tf.train.Optimizer`'s `minimize` method.

    Returns:
      TensorFlow Op.

    Raises:
      ValueError: If unconstrained_steps is provided, but global_step is not.
    """

    def unconstrained_fn():
      """Returns an `Op` for minimizing the unconstrained problem."""
      return self.minimize_unconstrained(
          minimization_problem=minimization_problem,
          global_step=global_step,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          name=name,
          grad_loss=grad_loss)

    def constrained_fn():
      """Returns an `Op` for minimizing the constrained problem."""
      return self.minimize_constrained(
          minimization_problem=minimization_problem,
          global_step=global_step,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          name=name,
          grad_loss=grad_loss)

    if unconstrained_steps is not None:
      if global_step is None:
        raise ValueError(
            "global_step cannot be None if unconstrained_steps is provided")
      unconstrained_steps_tensor = ops.convert_to_tensor(unconstrained_steps)
      dtype = unconstrained_steps_tensor.dtype
      return control_flow_ops.cond(
          standard_ops.cast(global_step, dtype) < unconstrained_steps_tensor,
          true_fn=unconstrained_fn,
          false_fn=constrained_fn)
    else:
      return constrained_fn()
