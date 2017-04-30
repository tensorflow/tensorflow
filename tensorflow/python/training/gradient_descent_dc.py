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

"""GradientDescentDC for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class GradientDescentOptimizerDC(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm with delay compensation.
  See [Zheng, Shuxin, et al., 2016](https://arxiv.org/abs/1609.08326)
  ([pdf](https://arxiv.org/pdf/1609.08326.pdf)).
  """

  def __init__(self, learning_rate, variance_parameter, use_locking=False, name="GradientDescentDC"):
    """Construct a new gradient descent optimizer with delay compensation.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      variance_parameter: A Tensor or a floating point value. The lambda value to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescentDC".
    """
    super(GradientDescentOptimizerDC, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._lambda = variance_parameter

  def _apply_dense(self, grad, var):
    var_bak = var
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.add(grad, math_ops.multiply(math_ops.multiply(grad, math_ops.cast(self._lambda_tensor, grad.dtype.base_dtype)), math_ops.subtract(var, var_bak))),
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    var_bak = handle.handle
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        math_ops.add(grad, math_ops.multiply(math_ops.multiply(grad, math_ops.cast(self._lambda_tensor, grad.dtype.base_dtype)), math_ops.subtract(handle.handle, var_bak))),
        use_locking=self._use_locking)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._lambda_tensor = ops.convert_to_tensor(self._lambda,
                                                      name="lambda")
