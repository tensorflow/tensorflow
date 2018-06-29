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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class RpropMinusOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Rprop- update.

  In the Rprop algorithms the direction of each weight update is given by
  the sign of the partial derivative `g{t}`. The step size is decoupled
  from the absolute value of the gradient, and is adjusted according to a
  local adaptive heuristic.

  See [Igel and Husken, 2003],
  [Empirical evaluation of the improved Rprop learning algorithms](https://pdfs.semanticscholar.org/ead3/a65b0c851a15ac6805b2ea7af13fa10c3fae.pdf).
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
    delta_update <- delta_zero (Initialize initial step size)
    t <- 0 (Initialize timestep)
    ```

    The following update rule is performed for each individual weight:

    ```
    t <- t + 1
    grad_sign <- sign(g{t} * g{t-1})
    if (grad_sign > 0)
      delta_update{t} <- min(eta_plus * delta_update{t-1}, delta_max)
    else if (grad_sign < 0)
      delta_update{t} <- max(eta_minus * delta_update{t-1}, delta_min)
    variable{t+1} <- variable{t} -sign(g{t}) * delta_update{t}
    ```

    Args:
      eta_minus: step size decrease multiplier
      eta_plus: step size increase multiplier
      delta_zero: initial step size quantity
      delta_min: lower bound on the step size
      delta_max: upper bound on the step size
      use_locking: If True, use locks for update operations.
      name: Optional name for the operations created iwhen applying gradients.
         Defaults to "RpropMinusOptimizer".
    """
    super(RpropMinusOptimizer, self).__init__(use_locking, name)

    # init parameters
    self._eta_minus = eta_minus
    self._eta_plus = eta_plus
    self._delta_zero = delta_zero
    self._delta_min = delta_min
    self._delta_max = delta_max

    # Tensor versions of the constructor arguments, created in _prepare().
    self._eta_minus_t = None
    self._eta_plus_t = None
    self._delta_min_t = None
    self._delta_max_t = None

  def _create_slots(self, var_list):
    # create slots for the gradient at (t-1) and
    # the step size "delta_update"
    for v in var_list:
      # gradient from the previous step
      self._zeros_slot(v, "old_grad", self._name)

      # delta update slot
      init_step = math_ops.add(array_ops.zeros(
          v.get_shape().as_list(), dtype=v.dtype.base_dtype), self._delta_zero)
      self._get_or_make_slot(v, init_step, "delta_update", self._name)

  def _prepare(self):
    self._eta_minus_t = ops.convert_to_tensor(self._eta_minus, name="eta_minus")
    self._eta_plus_t = ops.convert_to_tensor(self._eta_plus, name="eta_plus")
    self._delta_min_t = ops.convert_to_tensor(self._delta_min, name="delta_min")
    self._delta_max_t = ops.convert_to_tensor(self._delta_max, name="delta_max")

  def _apply_dense(self, grad, var):
    old_grad = self.get_slot(var, "old_grad")
    delta_update = self.get_slot(var, "delta_update")

    eta_minus_t = math_ops.cast(self._eta_minus_t, var.dtype.base_dtype)
    eta_plus_t = math_ops.cast(self._eta_plus_t, var.dtype.base_dtype)
    delta_min_t = math_ops.cast(self._delta_min_t, var.dtype.base_dtype)
    delta_max_t = math_ops.cast(self._delta_max_t, var.dtype.base_dtype)

    return training_ops.apply_rprop_minus(
        var,
        old_grad,
        delta_update,
        eta_minus_t,
        eta_plus_t,
        delta_min_t,
        delta_max_t,
        grad, use_locking=self._use_locking).op
