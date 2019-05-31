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

"""Adagrad optimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops


class AdagradOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the Adagrad algorithm.

  See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  or this
  [intro](https://ppasupat.github.io/a9online/uploads/proximal_notes.pdf).
  """

  def __init__(self, learning_rate, initial_accumulator_value=0.1,
               use_locking=False, name="Adagrad"):
    """Construct a new Adagrad optimizer.

    The learning_rate arg below is a hyperparameter, where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.
    """
    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    super(AdagradOptimizer, self).__init__(use_locking, name)
    self._set_hyper("learning_rate", learning_rate)

    self._initial_accumulator_value = initial_accumulator_value

  def _create_vars(self, var_list, state):
    for v in var_list:
      dtype = v.dtype.base_dtype
      if v.get_shape().is_fully_defined():
        init = init_ops.constant_initializer(
            self._initial_accumulator_value, dtype=dtype)
      else:

        def init(v=v, dtype=dtype):
          # Use a Tensor instead of initializer if variable does not have
          # static shape.
          init_constant = gen_array_ops.fill(
              array_ops.shape(v), self._initial_accumulator_value)
          return math_ops.cast(init_constant, dtype)

      state.create_slot_with_initializer(v, init, v.get_shape(), dtype,
                                         "accumulator")

  def _apply_dense(self, grad, var, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.apply_adagrad(
        var,
        acc,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.resource_apply_adagrad(
        var.handle,
        acc.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.sparse_apply_adagrad(
        var,
        acc,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, state):
    acc = state.get_slot(var, "accumulator")
    return training_ops.resource_sparse_apply_adagrad(
        var.handle,
        acc.handle,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad,
        indices,
        use_locking=self._use_locking)
