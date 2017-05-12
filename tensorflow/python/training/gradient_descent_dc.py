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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

class GradientDescentDCOptimizer(optimizer.Optimizer):

    """Optimizer that implements the gradient descent algorithm with delay
    compensation.

    See [Zheng, Shuxin, et al., 2016](https://arxiv.org/abs/1609.08326)
    ([pdf](https://arxiv.org/pdf/1609.08326.pdf)).
    """

    def __init__(self, learning_rate, variance_parameter, num_workers=1,
                 use_locking=False, name="GradientDescentDC"):
        """Construct a new gradient descent optimizer with delay compensation.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning
            rate to use.
          variance_parameter: A Tensor or a floating point value. The lambda
            value to use.
          num_workers: A value to indicate number of workers computing gradients
            asynchronously.
          use_locking: If True use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "GradientDescentDC".
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be positive: %s" % num_workers)
        super(GradientDescentDCOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._lambda = variance_parameter
        self._num_workers = num_workers

    def _create_slots(self, var_list):
        """Initialize slots for all the vars of each worker to store
            the previous values of it
        """
        for index in range(self._num_workers):
            for v in var_list:
                var2 = array_ops.identity(v.initialized_value())
                self._get_or_make_slot(v, var2, "var_bak_{0}".format(index),
                                       self._name)

    def _apply_dense(self, grad, var, worker_index=0):
        # Get previous value of the variable from the slot
        var_bak = self.get_slot(var, "var_bak_{0}".format(worker_index))
        return training_ops.apply_gradient_descent_dc(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad,
            math_ops.cast(self._lambda_tensor, var.dtype.base_dtype),
            var_bak,
            use_locking=self._use_locking).op

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                           name="learning_rate")
        self._lambda_tensor = ops.convert_to_tensor(self._lambda,
                                                    name="lambda")
