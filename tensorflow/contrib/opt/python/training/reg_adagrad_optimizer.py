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
"""RegAdagrad for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops
from tensorflow.python.training import adagrad
from tensorflow.python.training import training_ops
from tensorflow.python.util import tf_contextlib


class RegAdagradOptimizer(adagrad.AdagradOptimizer):
  """RegAdagrad: Adagrad with updates that optionally skip updating the slots.

  This is meant to address the problem of additional regularization terms in the
  loss function affecting learning rate decay and causing hyper-param
  entanglement. Example usage:

    loss = tf.nn.cross_entropy(x, labels)
    reg_loss = reg_strength * tf.reduce_sum(x * x)
    opt = tf.contrib.opt.RegAdagradOptimizer(learning_rate)
    loss_update = opt.minimize(loss)
    with opt.avoid_updating_slots():
      reg_update = opt.minimize(reg_loss)
    total_update = tf.group([loss_update, reg_update])

    # ...

    sess.run(total_update, ...)
  """

  def __init__(self,
               learning_rate,
               initial_accumulator_value=0.1,
               use_locking=False,
               name="RegAdagrad"):
    super(RegAdagradOptimizer, self).__init__(
        learning_rate,
        initial_accumulator_value=initial_accumulator_value,
        use_locking=use_locking,
        name=name)
    self._should_update_slots = True

  @tf_contextlib.contextmanager
  def avoid_updating_slots(self):
    old = self._should_update_slots
    self._should_update_slots = False
    try:
      yield
    finally:
      self._should_update_slots = old

  def _apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return training_ops.apply_adagrad(
        var,
        acc,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking,
        update_slots=self._should_update_slots)

  def _resource_apply_dense(self, grad, var, update_slots=True):
    acc = self.get_slot(var, "accumulator")
    return training_ops.resource_apply_adagrad(
        var.handle,
        acc.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking,
        update_slots=self._should_update_slots)

  def _apply_sparse(self, grad, var, update_slots=True):
    acc = self.get_slot(var, "accumulator")
    return training_ops.sparse_apply_adagrad(
        var,
        acc,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking,
        update_slots=self._should_update_slots)

  def _resource_apply_sparse(self, grad, var, indices, update_slots=True):
    acc = self.get_slot(var, "accumulator")
    return training_ops.resource_sparse_apply_adagrad(
        var.handle,
        acc.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad,
        indices,
        use_locking=self._use_locking,
        update_slots=self._should_update_slots)
