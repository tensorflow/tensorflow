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
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class MomentumOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Momentum algorithm.

  Computes (if `use_nesterov = False`):
  
  ```
  accumulation = momentum * accumulation + gradient
  variable -= learning_rate * accumulation
  ```

  Note that in the dense version of this algorithm, `accumulation` is updated
  and applied regardless of a gradient's value, whereas the sparse version (when
  the gradient is an `IndexedSlices`, typically because of `tf.gather` or an
  embedding) only updates variable slices and corresponding `accumulation` terms
  when that part of the variable was used in the forward pass.
  """

  def __init__(self, learning_rate, momentum,
               use_locking=False, name="Momentum", use_nesterov=False):
    """Construct a new Momentum optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
      use_nesterov: If `True` use Nesterov Momentum.
        See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf)

    """
    super(MomentumOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._use_nesterov = use_nesterov

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._momentum_tensor = ops.convert_to_tensor(self._momentum,
                                                  name="momentum")

  def _apply_dense(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov).op

  def _resource_apply_dense(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.resource_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov)

  def _apply_sparse(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.sparse_apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values, grad.indices,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov).op

  def _resource_apply_sparse(self, grad, var, indices):
    mom = self.get_slot(var, "momentum")
    return training_ops.resource_sparse_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad, indices,
        math_ops.cast(self._momentum_tensor, grad.dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov)
