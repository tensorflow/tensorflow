# Copyright 2015 Google Inc. All Rights Reserved.
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
from tensorflow.python.ops import constant_op
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class MomentumOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Momentum algorithm.

  @@__init__
  """

  def __init__(self, learning_rate, momentum,
               use_locking=False, name="Momentum"):
    """Construct a new Momentum optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
    """
    super(MomentumOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._momentum = momentum

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
        self._learning_rate_tensor, grad, self._momentum_tensor,
        use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.sparse_apply_momentum(
        var, mom,
        self._learning_rate_tensor, grad.values, grad.indices,
        self._momentum_tensor, use_locking=self._use_locking).op
