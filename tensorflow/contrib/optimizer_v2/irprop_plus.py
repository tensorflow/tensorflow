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

"""Implementation of iRprop+"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops


class IRpropPlusOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the iRprop+ algorithm.

  The Rprop (resilient backpropagation) algorithms are efficient gradient-based
  optimization algorithms. They require hardly any hyperparameter tuning.

  In Rprop, the direction of each objective variable update is given by the sign
  of the partial derivative. The amount of the change is decoupled
  from the absolute value of the partial derivativsadasdase. It is determined by a
  step-size parameter, which is individually adapted for each objective
  variable.

  Rprop was originally proposed by Riedmiller and Braun in the article
  [A direct adaptive method for faster backpropagation learning:
  the RPROP algorithm](https://doi.org/10.1109/ICNN.1993.298623).
  The original Rprop algorithm uses weight-backtracking. It retracts the update
  of an objective variable if the update caused a change in sign of the
  corresponding partial derivative. The implememented Rprop variant, which is
  called iRprop+ and is described in the article
  [Empirical evaluation of the improved Rprop learning algorithms](https://doi.org/10.1016/S0925-2312(01)00700-7)
  only retracts an update if additionally the overall error increased.
  The TensorFlow implementation is described in the article
  [Resilient Backpropagation (Rprop) for Batch-learning in TensorFlow](https://openreview.net/forum?id=r1R0o7yDz).

  **The Rprop algorithms are recommended for batch learning, _not_ for
  mini-batch learning.** The iRprop+ (improved Rprop with weight-backtracking)
  algorithm is empirically found to be faster and more robust than the
  [standard variant](RpropMinusOptimizer.md).
  See [Resilient Backpropagation (Rprop) for Batch-learning in
  TensorFlow](https://openreview.net/forum?id=r1R0o7yDz)
  for details and references.
  """

  def __init__(self,
               eta_minus=0.5,
               eta_plus=1.2,
               delta_zero=0.5,
               delta_min=1e-6,
               delta_max=50,
               use_locking=False, name="IRpropPlusOptimizer"):
    """Constructs a new IRpropPlusOptimizer object.

    The pseudocode of the algorithm can be found in the articles
    [Empirical evaluation of the improved Rprop learning algorithms](https://doi.org/10.1016/S0925-2312(01)00700-7)
    and [Resilient Backpropagation (Rprop) for Batch-learning in TensorFlow](https://openreview.net/forum?id=r1R0o7yDz).

    Initialization:

    ```
    old_grad <- 0 (Initialize the gradient from the previous iteration g{t-1})
    delta_update <- delta_zero (Initialize step-size)
    t <- 0 (Initialize iteration counter)
    ```

    The following update rule is performed for each individual objective
    variable (e.g., weight), where `g{t}` denotes the partial derivative of the
    objective function with respect to the objective variable at iteration `t`:

    ```
    t <- t + 1
    grad_sign <- sign(g{t} * g{t-1})
    if (grad_sign > 0)
      delta_update{t} <- min(eta_plus * delta_update{t-1}, delta_max)
      variable{t+1} <- variable{t} -sign(g{t}) * delta_update{t}
    else if (grad_sign < 0)
      delta_update{t} <- max(eta_minus * delta_update{t-1}, delta_min)
      if (error{t} > error{t-1})
        variable{t+1} <- variable{t-1}
      g{t} = 0
    else
      variable{t+1} <- variable{t} - sign(g{t}) * delta_update{t}

    ```

    Args:
      eta_minus: Step-size decrease factor.
      eta_plus: Step-size increase factor.
      delta_zero: Initial step-size.
      delta_min: Lower bound on step-size.
      delta_max: Upper bound on step-size.
      use_locking: If True, use locks for update operations.
      name: Optional name for the operations created when applying gradients.
         Defaults to "IRpropPlusOptimizer".
   """

    super(IRpropPlusOptimizer, self).__init__(use_locking, name)

    # Init parameters
    self._set_hyper("eta_minus", eta_minus)
    self._set_hyper("eta_plus", eta_plus)
    self._set_hyper("delta_zero", delta_zero)
    self._set_hyper("delta_min", delta_min)
    self._set_hyper("delta_max", delta_max)

    # Error auxiliary var
    self._error = None

  def _create_vars(self, var_list, state):

    for v in var_list:
      state.zeros_slot(v, "old_grad")

      init_step = math_ops.add(
          array_ops.zeros_like(v),
          state.get_hyper("delta_zero", v.dtype.base_dtype))
      state.create_slot_with_initializer(v, init_step, v.get_shape(),
                                         v.dtype.base_dtype, "delta_update")
      # Error tensors
      state.create_non_slot(array_ops.zeros_like(v), "error")
      state.create_non_slot(array_ops.zeros_like(v), "old_error")

  def _get_error_values(self, state=None):
    if state is None:
      state = self._get_per_graph_state()
    return (state.get_non_slot("error"),
            state.get_non_slot("old_error"))


  def _apply_dense(self, grad, var, state):
    old_grad = state.get_slot(var, "old_grad")
    delta_update = state.get_slot(var, "delta_update")

    error, old_error = self._get_error_values(state)
    # Update the error E(t) passed in apply_gradients or minimize
    update_err = error.assign(math_ops.cast(self._error, var.dtype.base_dtype),
                              use_locking=self._use_locking)

    with ops.control_dependencies([update_err]):
      return training_ops.apply_i_rprop_plus(
          var,
          old_grad,
          delta_update,
          state.get_hyper("eta_minus", var.dtype.base_dtype),
          state.get_hyper("eta_plus", var.dtype.base_dtype),
          state.get_hyper("delta_min", var.dtype.base_dtype),
          state.get_hyper("delta_max", var.dtype.base_dtype),
          math_ops.cast(error, var.dtype.base_dtype),
          math_ops.cast(old_error, var.dtype.base_dtype),
          grad, use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var, state):
    old_grad = state.get_slot(var, "old_grad")
    delta_update = state.get_slot(var, "delta_update")
    error, old_error = self._get_error_values(state)
    update_err = error.assign(math_ops.cast(self._error, var.dtype.base_dtype),
                              use_locking=self._use_locking)

    with ops.control_dependencies([update_err]):
      return training_ops.resource_apply_i_rprop_plus(
          var.handle,
          old_grad.handle,
          delta_update.handle,
          state.get_hyper("eta_minus", var.dtype.base_dtype),
          state.get_hyper("eta_plus", var.dtype.base_dtype),
          state.get_hyper("delta_min", var.dtype.base_dtype),
          state.get_hyper("delta_max", var.dtype.base_dtype),
          math_ops.cast(error, var.dtype.base_dtype),
          math_ops.cast(old_error, var.dtype.base_dtype),
          grad, use_locking=self._use_locking)

  def minimize(self,
               loss,
               global_step=None,
               var_list=None,
               gate_gradients=optimizer_v2.OptimizerV2.GATE_OP,
               aggregation_method=None,
               name=None,
               grad_loss=None,
               stop_gradients=None,
               scale_loss_by_num_replicas=None):
    """Add operations to minimize `loss` by updating `var_list`.
    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function. The loss argument has to be passed as a 0-D Tensor.
    Args:
      loss: A 0-D `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in
        the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.
    Raises:
      ValueError: If some of the variables are not `Variable` objects.
      ValueError: If the `loss` is not a 0-D tensor (scalar).
    @compatibility(eager)
    When eager execution is enabled, `loss` should be a Python function that
    takes elements of `var_list` as arguments and computes the value to be
    minimized. If `var_list` is `None`, `loss` should take no arguments.
    Minimization (and gradient computation) is done with respect to the
    elements of `var_list` if not `None`, else with respect to any trainable
    variables created during the execution of the `loss` function.
    `gate_gradients`, `aggregation_method`, `colocate_gradients_with_ops` and
    `grad_loss` are ignored when eager execution is enabled.
    @end_compatibility
    """
    # Override method from base class, the loss is required in the update step
    if callable(loss):
      self._error = loss()
    else:
      self._error = loss

    return super(IRpropPlusOptimizer, self).minimize(
        loss,
        global_step=global_step,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        name=name,
        grad_loss=grad_loss,
        stop_gradients=stop_gradients,
        scale_loss_by_num_replicas=scale_loss_by_num_replicas)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=optimizer_v2.OptimizerV2.GATE_OP,
                        aggregation_method=None,
                        grad_loss=None,
                        stop_gradients=None,
                        scale_loss_by_num_replicas=None):
    # eager compatibility
    if callable(loss):
      self._error = loss()
    else:
      self._error = loss

    return super(IRpropPlusOptimizer, self).compute_gradients(
        loss=loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        grad_loss=grad_loss,
        stop_gradients=stop_gradients,
        scale_loss_by_num_replicas=scale_loss_by_num_replicas)

  def _finish(self, state):
    error, old_error = self._get_error_values(state)
    # Update the old error E(t-1) <- E(t)
    update_old_error = old_error.assign(error, use_locking=self._use_locking)
    return control_flow_ops.group(update_old_error)
