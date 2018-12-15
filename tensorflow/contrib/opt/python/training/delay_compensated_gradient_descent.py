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

"""DelayCompensatedGradientDescentOptimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import array_ops

GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2


class DelayCompensatedGradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements the DelayCompensatedGradientDescent algorithm.

  See [Zheng, Shuxin, et al., 2016](https://arxiv.org/abs/1609.08326)
  ([pdf](https://arxiv.org/pdf/1609.08326.pdf)).
  """

  def __init__(self, learning_rate, variance_parameter=2.0, num_workers=1,
               use_locking=False, name="DelayCompensatedGradientDescentOptimizer"):

    """Construct a gradient descent optimizer with delay compensation.

    It is cricial to note the `num_workers` in constructor and `worker_index` in
    `minimize()` and `apply_gradients()`.

    Contrast to AdaMaxamOptimizer, the sparse implementation of this algorithm
    (used when the gradient is an IndexedSlices object, typically because of
    `tf.gather` or an embedding lookup in the forward pass) only updates
    variable slices and corresponding `shadow_t` term when that part of
    the variable was used in the forward pass. This means that the sparse
    behavior is contrast to the dense behavior (similar to some momentum
    implementations which ignore momentum unless a variable slice was actually
    used).

    Args:
      learning_rate: A Tensor or a floating point value. The learning rate.
      variance_parameter: A Tensor or a floating point value.
        The variance control parameter.
      num_workers: A int value. The number of workers.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "DelayCompensatedGradientDescentOptimizer".
    """
    num_workers = self._call_if_callable(num_workers)
    if num_workers <= 0:
      raise ValueError("num_workers must be positive: %s" % num_workers)
    super(DelayCompensatedGradientDescentOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._lambda = variance_parameter
    self._num_workers = num_workers
    self._learning_rate_tensor = None
    self._lambda_tensor = None
    self._use_locking = use_locking

  def _create_slots(self, var_list):
    for index in range(self._num_workers):
      for v in var_list:
        self._zeros_slot(v, "shadow_{0}".format(index), self._name)


  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    lambda_ = self._call_if_callable(self._lambda)

    self._learning_rate_tensor = ops.convert_to_tensor(lr, name="learning_rate")
    self._lambda_tensor = ops.convert_to_tensor(lambda_, name="lambda")

  def _apply_dense(self, grad, var):

    shadow = self.get_slot(var, "shadow_{0}".format(self.worker_index))
    return training_ops.apply_delay_compensated_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._lambda_tensor, grad.dtype.base_dtype),
        shadow,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):

    shadow = self.get_slot(var, "shadow_{0}".format(self.worker_index))
    return training_ops.resource_apply_delay_compensated_gradient_descent(
        var.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._lambda_tensor, grad.dtype.base_dtype),
        shadow.handle,
        use_locking=self._use_locking)

  def _apply_sparse_shared(self, grad, var, indices):

    shadow = self.get_slot(var, "shadow_{0}".format(self.worker_index))
    # if shadow is None:
    #   raise ValueError("None shadow with index = " + str(self.worker_index) + " and var = " + str(var))
    lambda_ = math_ops.cast(self._lambda_tensor, var.dtype.base_dtype)
    lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)

    var_slice = array_ops.gather(var, indices)
    shadow_slice = array_ops.gather(shadow, indices)

    var_scaled_g_values = lr * (grad + lambda_ * grad * grad * (var_slice - shadow_slice))

    var_t = state_ops.scatter_add(var, indices, -var_scaled_g_values, use_locking=self._use_locking)

    with ops.control_dependencies([var_t]):
      shadow_t = state_ops.assign(shadow, var_t)

    return control_flow_ops.group(*[var_t, shadow_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices)

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(
        grad, var, indices)



  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None, worker_index=0):
    self.worker_index = worker_index
    return super(DelayCompensatedGradientDescentOptimizer, self).minimize(loss=loss, global_step=global_step, var_list=var_list,
               gate_gradients=gate_gradients, aggregation_method=aggregation_method,
               colocate_gradients_with_ops=colocate_gradients_with_ops, name=name,
               grad_loss=grad_loss)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None, worker_index=0):
    self.worker_index = worker_index
    return super(DelayCompensatedGradientDescentOptimizer, self).apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step, name=name)
