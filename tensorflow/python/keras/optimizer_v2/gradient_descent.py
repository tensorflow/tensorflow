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

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops


class SGD(optimizer_v2.OptimizerV2):
  """Stochastic gradient descent optimizer.

  Computes:

  ```
  variable -= learning_rate * gradient
  ```

  Some of the args below are hyperparameters, where a hyperparameter is
  defined as a scalar Tensor, a regular Python value, or a callable (which
  will be evaluated when `apply_gradients` is called) returning a scalar
  Tensor or a Python value.

  @compatibility(eager)
  When eager execution is enabled, learning_rate can be a callable that takes
  no arguments and returns the actual value to use. This can be useful for
  changing these values across different invocations of optimizer functions.
  @end_compatibility

  Arguments:
      learning_rate: float hyperparameter >= 0. Learning rate.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'SGD'.
  """

  def __init__(self,
               learning_rate=0.001,
               momentum=None,
               nesterov=False,
               name="SGD"):
    super(SGD, self).__init__(name)
    self._set_hyper("learning_rate", learning_rate)

  def _apply_dense(self, grad, var):
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._get_hyper("learning_rate"), var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    return training_ops.resource_apply_gradient_descent(
        var.handle,
        math_ops.cast(self._get_hyper("learning_rate"), var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
    return resource_variable_ops.resource_scatter_add(
        var.handle, indices, -grad * math_ops.cast(
            self._get_hyper("learning_rate"), var.dtype.base_dtype))

  def _apply_sparse_duplicate_indices(self, grad, var):
    delta = ops.IndexedSlices(
        grad.values * math_ops.cast(
            self._get_hyper("learning_rate"), var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def get_config(self):
    config = super(SGD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
    })
    return config
