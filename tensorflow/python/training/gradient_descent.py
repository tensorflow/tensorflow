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

"""GradientDescent for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.GradientDescentOptimizer"])
class GradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.
  """

  def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
    """Construct a new gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".

    @compatibility(eager)
    When eager execution is enabled, `learning_rate` can be a callable that
    takes no arguments and returns the actual value to use. This can be useful
    for changing these values across different invocations of optimizer
    functions.
    @end_compatibility
    """
    super(GradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._learning_rate_tensor = None

  def _apply_dense(self, grad, var):
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return resource_variable_ops.resource_scatter_add(
        handle.handle, indices, -grad * self._learning_rate)

  def _apply_sparse_duplicate_indices(self, grad, var):
    delta = ops.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = ops.convert_to_tensor(
        learning_rate, name="learning_rate")
