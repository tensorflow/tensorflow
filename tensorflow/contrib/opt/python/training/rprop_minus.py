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
"""Implementation of Rprop-"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.training import training_ops


class RpropMinusOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the Rprop- algorithm.

  The Rprop (resilient backpropagation) algorithms are efficient gradient-based
  optimization algorithms. They require hardly any hyperparameter tuning.

  In Rprop, the direction of each objective variable update is given by the sign
  of the partial derivative. The amount of the change is decoupled
  from the absolute value of the partial derivative. It is determined by a step-size
  parameter, which is individually adapted for each objective variable.

  Rprop was originally proposed by Riedmiller and Braun in the paper
  [A direct adaptive method for faster backpropagation learning: the RPROP algorithm](https://doi.org/10.1109/ICNN.1993.298623).
  The Rprop variant implemented here is described in Riedmiller's article
  [Advanced supervised learning in multi-layer perceptrons — From backpropagation to adaptive learning algorithms](https://doi.org/10.1016/0920-5489(94)90017-5),
  and is referred to as Rprop- (Rprop without weight-backtracking) in the
  article [Empirical evaluation of the improved Rprop learning algorithms](https://doi.org/10.1016/S0925-2312(01)00700-7).

  **The Rprop algorithms are recommended for batch learning, _not_
  for mini-batch learning.** The variant
  [iRprop+ (improved Rprop with weight-backtracking)
   algorithm](IRpropPlusOptimizer.md)
  is empirically found to be faster and more robust than Rprop-.
  See [Resilient Backpropagation (Rprop) for Batch-learning in TensorFlow](https://openreview.net/forum?id=r1R0o7yDz)
  for details and references.
  """

  def __init__(self,
               eta_minus=0.5,
               eta_plus=1.2,
               delta_zero=0.5,
               delta_min=1e-6,
               delta_max=50,
               use_locking=False, name="RpropMinusOptimizer"):
    """Constructs a new RpropMinusOptimizer object.

    Initialization:

    ```
    old_grad <- 0 (Initialize the gradient from the previous timestep g{t-1})
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
    else if (grad_sign < 0)
      delta_update{t} <- max(eta_minus * delta_update{t-1}, delta_min)
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
         Defaults to "RpropMinusOptimizer".
    """
    super(RpropMinusOptimizer, self).__init__(use_locking, name)

    # Init parameters
    self._set_hyper("eta_minus", eta_minus)
    self._set_hyper("eta_plus", eta_plus)
    self._set_hyper("delta_zero", delta_zero)
    self._set_hyper("delta_min", delta_min)
    self._set_hyper("delta_max", delta_max)

  def _create_vars(self, var_list, state):
    for v in var_list:
      state.zeros_slot(v, "old_grad")

      init_step = math_ops.add(
          array_ops.zeros_like(v),
          state.get_hyper("delta_zero", v.dtype.base_dtype))
      state.create_slot_with_initializer(v, init_step, v.get_shape(),
                                         v.dtype.base_dtype, "delta_update")

  def _apply_dense(self, grad, var, state):
    old_grad = state.get_slot(var, "old_grad")
    delta_update = state.get_slot(var, "delta_update")

    return training_ops.apply_rprop_minus(
        var,
        old_grad,
        delta_update,
        state.get_hyper("eta_minus", var.dtype.base_dtype),
        state.get_hyper("eta_plus", var.dtype.base_dtype),
        state.get_hyper("delta_min", var.dtype.base_dtype),
        state.get_hyper("delta_max", var.dtype.base_dtype),
        grad, use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var, state):
    old_grad = state.get_slot(var, "old_grad")
    delta_update = state.get_slot(var, "delta_update")

    return training_ops.resource_apply_rprop_minus(
        var.handle,
        old_grad.handle,
        delta_update.handle,
        state.get_hyper("eta_minus", var.dtype.base_dtype),
        state.get_hyper("eta_plus", var.dtype.base_dtype),
        state.get_hyper("delta_min", var.dtype.base_dtype),
        state.get_hyper("delta_max", var.dtype.base_dtype),
        grad, use_locking=self._use_locking)
