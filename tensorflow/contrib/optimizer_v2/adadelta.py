# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Adadelta for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.training import training_ops


class AdadeltaOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the Adadelta algorithm.

  See [M. D. Zeiler](http://arxiv.org/abs/1212.5701)
  ([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))
  """

  def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-8,
               use_locking=False, name="Adadelta"):
    """Construct a new Adadelta optimizer.

    Some of the args below are hyperparameters, where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate.
        To match the exact form in the original paper use 1.0.
      rho: A float hyperparameter. The decay rate.
      epsilon: A float hyperparameter. A constant epsilon used to better
        condition the grad update.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adadelta".
    """
    super(AdadeltaOptimizer, self).__init__(use_locking, name)
    self._set_hyper("learning_rate", learning_rate)
    self._set_hyper("rho", rho)
    self._set_hyper("epsilon", epsilon)

  def _create_vars(self, var_list, state):
    for v in var_list:
      state.zeros_slot(v, "accum")
      state.zeros_slot(v, "accum_update")

  def _apply_dense(self, grad, var, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.apply_adadelta(
        var,
        accum,
        accum_update,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.resource_apply_adadelta(
        var.handle,
        accum.handle,
        accum_update.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.sparse_apply_adadelta(
        var,
        accum,
        accum_update,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, state):
    accum = state.get_slot(var, "accum")
    accum_update = state.get_slot(var, "accum_update")
    return training_ops.resource_sparse_apply_adadelta(
        var.handle,
        accum.handle,
        accum_update.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("rho", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad,
        indices,
        use_locking=self._use_locking)
