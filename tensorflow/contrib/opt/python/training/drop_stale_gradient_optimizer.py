# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Wrapper optimizer for checking and dropping stale gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_util


class DropStaleGradientOptimizer(optimizer.Optimizer):
  """Wrapper optimizer that checks and drops stale gradient.

  This optimizer records the global step for each worker before computing
  gradients and compares it with the global step at the time of applying the
  gradients. If the difference is larger than a threshold, it will drop all
  the computed gradients.
  """

  def __init__(self,
               opt,
               staleness,
               use_locking=False,
               name="DropStaleGradient"):
    """Constructs a new DropStaleGradientOptimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
           gradients. Must be one of the Optimizer classes.
      staleness: The maximum staleness allowed for the optimizer.
      use_locking: If `True` use locks for clip update operations.
      name: Optional name prefix for the operations created when applying
            gradients. Defaults to "DropStaleGradient".
    """
    super(DropStaleGradientOptimizer, self).__init__(use_locking, name)
    self._opt = opt
    self._staleness = staleness

  def compute_gradients(self, loss, *args, **kwargs):
    # Record current global step for worker.
    with ops.colocate_with(loss):
      self._local_step = training_util.get_global_step().read_value() + 0

    with ops.control_dependencies([self._local_step]):
      loss = gen_array_ops.identity(loss)
      return self._opt.compute_gradients(loss, *args, **kwargs)

  def get_slot(self, *args, **kwargs):
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    return self._opt.get_slot_names(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    gradients = []
    # Number of stale gradients.
    with ops.colocate_with(global_step):
      stale_counter = variable_scope.get_variable(
          "stale_counter", [],
          initializer=init_ops.zeros_initializer(),
          trainable=False)

    def _AcceptGradientOp():
      with ops.control_dependencies(
          [self._opt.apply_gradients(
              grads_and_vars, global_step=global_step, name=name)]):
        return gen_array_ops.identity(0.0)

    def _DropGradientOp():
      return gen_array_ops.identity(1.0)

    for grad_and_var in grads_and_vars:
      grad = grad_and_var[0]
      if isinstance(grad, ops.Tensor):
        gradients.append(grad)
      elif grad is not None:
        gradients.append(grad.op)

    with ops.control_dependencies(gradients), ops.colocate_with(global_step):
      staleness = gen_array_ops.reshape(
          global_step.read_value() - self._local_step, shape=())

    conditional_update = stale_counter.assign_add(control_flow_ops.cond(
        gen_math_ops.less_equal(staleness, self._staleness),
        _AcceptGradientOp, _DropGradientOp))

    summary.scalar(
        "Gradient staleness percentage", stale_counter / (math_ops.cast(
            global_step.read_value() + 1, dtypes.float32)))
    return conditional_update
