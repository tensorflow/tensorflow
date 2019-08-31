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

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Adagrad')
class Adagrad(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adagrad algorithm.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  Initialization:
  $$accum_{g_0} := \text{initial_accumulator_value}$$

  Update step:
  $$t := t + 1$$
  $$accum_{g_t} := accum_{g_{t-1}} + g^2$$
  $$\theta_t := \theta_{t-1} - lr * g / (\sqrt{accum_{g_t}} + \epsilon)$$

  References:

  * [Paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  * [Introduction]
    (https://ppasupat.github.io/a9online/uploads/proximal_notes.pdf).
  """

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               name='Adagrad',
               **kwargs):
    """Construct a new Adagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be non-negative.
      epsilon: A small floating point value to avoid zero denominator.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    Raises:
      ValueError: If the `initial_accumulator_value` or `epsilon` is invalid.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate` can be a callable that
    takes no arguments and returns the actual value to use. This can be useful
    for changing these values across different invocations of optimizer
    functions.
    @end_compatibility
    """
    if initial_accumulator_value < 0.0:
      raise ValueError('initial_accumulator_value must be non-negative: %s' %
                       initial_accumulator_value)
    if epsilon is None:
      epsilon = backend_config.epsilon()
    super(Adagrad, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._initial_accumulator_value = initial_accumulator_value
    self.epsilon = epsilon or backend_config.epsilon()

  def _create_slots(self, var_list):
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = init_ops.constant_initializer(
          self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, 'accumulator', init)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adagrad, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(dict(
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        neg_lr_t=-apply_state[(var_device, var_dtype)]['lr_t'],
        zero=array_ops.zeros((), dtype=dtypes.int64)
    ))

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(Adagrad, self).set_weights(weights)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates an optimizer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An optimizer instance.
    """
    if 'initial_accumulator_value' not in config:
      config['initial_accumulator_value'] = 0.
    if 'lr' in config:
      config['learning_rate'] = config.pop('lr')
    return cls(**config)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    acc = self.get_slot(var, 'accumulator')
    if compat.forward_compatible(2019, 8, 20):
      return training_ops.resource_apply_adagrad_v2(
          var.handle,
          acc.handle,
          coefficients['lr_t'],
          coefficients['epsilon'],
          grad,
          use_locking=self._use_locking)

    acc_t = state_ops.assign_add(
        acc, math_ops.square(grad), use_locking=self._use_locking)
    var_update = state_ops.assign_sub(
        var, coefficients['lr_t'] * grad /
        (math_ops.sqrt(acc_t) + coefficients['epsilon']))
    return var_update

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    acc = self.get_slot(var, 'accumulator')
    if compat.forward_compatible(2019, 8, 20):
      return training_ops.resource_sparse_apply_adagrad_v2(
          var.handle,
          acc.handle,
          coefficients['lr_t'],
          coefficients['epsilon'],
          grad,
          indices,
          use_locking=self._use_locking)
    with ops.control_dependencies([
        resource_variable_ops.resource_scatter_add(acc.handle, indices,
                                                   math_ops.square(grad))
    ]):
      acc_t_slice = acc.sparse_read(indices)
    var_update = resource_variable_ops.resource_scatter_add(
        var.handle, indices, coefficients['neg_lr_t'] * grad /
        (math_ops.sqrt(acc_t_slice) + coefficients['epsilon']))
    return var_update

  def get_config(self):
    config = super(Adagrad, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'initial_accumulator_value': self._initial_accumulator_value,
        'epsilon': self.epsilon,
    })
    return config
