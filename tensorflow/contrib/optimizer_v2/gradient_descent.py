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

"""Momentum for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops


class GradientDescentOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the gradient descent algorithm."""

  def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
    """Construct a new gradient descent optimizer.

    The learning rate arg below is a hyperparameter where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(GradientDescentOptimizer, self).__init__(use_locking, name)
    self._set_hyper("learning_rate", learning_rate)

  def _apply_dense(self, grad, var, state):
    return training_ops.apply_gradient_descent(
        var,
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle, state):
    lr = state.get_hyper("learning_rate", grad.dtype.base_dtype)
    return training_ops.resource_apply_gradient_descent(
        handle.handle, lr, grad, use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
                                               state):
    lr = state.get_hyper("learning_rate", grad.dtype.base_dtype)
    return resource_variable_ops.resource_scatter_add(handle.handle, indices,
                                                      -grad * lr)

  def _apply_sparse_duplicate_indices(self, grad, var, state):
    delta = ops.IndexedSlices(
        grad.values * state.get_hyper("learning_rate", var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)
