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

"""Adagrad for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops


class Adagrad(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adagrad algorithm.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  Initialization:

  $$accum_g_0 := initial_accumulator_value$$

  $$t := t + 1$$
  $$accum_g_t := accum_g_{t-1} + g * g$$
  $$theta_t := theta_{t-1} - lr * g / (\sqrt{accum_g_t} + \epsilon)$$

  References
    See [paper]
      (http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    or this
      [intro](https://ppasupat.github.io/a9online/uploads/proximal_notes.pdf).
  """

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               name='Adagrad'):
    """Construct a new Adagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      epsilon: A floating point value.
        Starting value for the accumulators, must be positive.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the `initial_accumulator_value` or `epsilon` is invalid.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate` can be a callable that
    takes no arguments and returns the actual value to use. This can be useful
    for changing these values across different invocations of optimizer
    functions.
    @end_compatibility
    """
    if initial_accumulator_value <= 0.0:
      raise ValueError('initial_accumulator_value must be positive: %s' %
                       initial_accumulator_value)
    if epsilon < 1e-7:
      raise ValueError('epsilon must be larger than 1e-7: %s' % epsilon)
    super(Adagrad, self).__init__(name)
    self._set_hyper('learning_rate', learning_rate)
    self._initial_accumulator_value = initial_accumulator_value
    self._set_hyper('epsilon', epsilon)

  def _create_slots(self, var_list):
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = init_ops.constant_initializer(
          self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, 'accumulator', init)

  def _init_constant_op(self, v, dtype):
    def init():
      # Use a Tensor instead of initializer if variable does not have
      # static shape.
      init_constant = gen_array_ops.fill(array_ops.shape(v),
                                         self._initial_accumulator_value)
      return math_ops.cast(init_constant, dtype)
    return init

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    learning_rate = math_ops.cast(self._get_hyper('learning_rate'), var_dtype)
    epsilon = math_ops.cast(self._get_hyper('epsilon'), var_dtype)
    acc = self.get_slot(var, 'accumulator')

    acc_t = state_ops.assign_add(
        acc, math_ops.square(grad), use_locking=self._use_locking)
    var_update = state_ops.assign_sub(
        var, learning_rate * grad / (math_ops.sqrt(acc_t) + epsilon))
    return var_update

  def _resource_apply_sparse(self, grad, var, indices):

    def _resource_scatter_add(x, i, v):
      with ops.control_dependencies(
          [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
        return x.value()

    var_dtype = var.dtype.base_dtype
    learning_rate = math_ops.cast(self._get_hyper('learning_rate'), var_dtype)
    epsilon = math_ops.cast(self._get_hyper('epsilon'), var_dtype)
    acc = self.get_slot(var, 'accumulator')

    acc_t = _resource_scatter_add(acc, indices, math_ops.square(grad))
    acc_t_slice = array_ops.gather(acc_t, indices)
    var_update = _resource_scatter_add(
        var, indices,
        -learning_rate * grad / (math_ops.sqrt(acc_t_slice) + epsilon))
    return var_update

  def get_config(self):
    config = super(Adagrad, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'initial_accumulator_value': self._initial_accumulator_value,
        'epsilon': self._serialize_hyperparameter('epsilon'),
    })
    return config
