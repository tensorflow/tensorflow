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

"""ProximalGradientDescent for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
# pylint: disable=unused-import
from tensorflow.python.ops import math_ops
# pylint: enable=unused-import
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.ProximalGradientDescentOptimizer"])
class ProximalGradientDescentOptimizer(optimizer.Optimizer):
  # pylint: disable=line-too-long
  """Optimizer that implements the proximal gradient descent algorithm.

  References:
    Efficient Learning using Forward-Backward Splitting:
      [Duchi et al., 2009](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting)
      ([pdf](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf))
  """

  def __init__(self, learning_rate, l1_regularization_strength=0.0,
               l2_regularization_strength=0.0, use_locking=False,
               name="ProximalGradientDescent"):
    """Construct a new proximal gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(ProximalGradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    self._l1_regularization_strength_tensor = None
    self._l2_regularization_strength_tensor = None

  def _apply_dense(self, grad, var):
    return training_ops.apply_proximal_gradient_descent(
        var,
        self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    return training_ops.resource_apply_proximal_gradient_descent(
        var.handle,
        self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    return training_ops.sparse_apply_proximal_gradient_descent(
        var,
        self._learning_rate_tensor,
        self._l1_regularization_strength_tensor,
        self._l2_regularization_strength_tensor,
        grad.values,
        grad.indices,
        use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices):
    return training_ops.resource_sparse_apply_proximal_gradient_descent(
        var.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
        grad,
        indices,
        use_locking=self._use_locking)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength, name="l1_regularization_strength")
    self._l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength, name="l2_regularization_strength")
