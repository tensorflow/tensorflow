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
"""Adadelta optimizer implementation."""
# pylint: disable=g-classes-have-attributes

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Adadelta')
class Adadelta(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adadelta algorithm.

  Adadelta optimization is a stochastic gradient descent method that is based on
  adaptive learning rate per dimension to address two drawbacks:

  - The continual decay of learning rates throughout training.
  - The need for a manually selected global learning rate.

  Adadelta is a more robust extension of Adagrad that adapts learning rates
  based on a moving window of gradient updates, instead of accumulating all
  past gradients. This way, Adadelta continues learning even when many updates
  have been done. Compared to Adagrad, in the original version of Adadelta you
  don't have to set an initial learning rate. In this version, the initial
  learning rate can be set, as in most other Keras optimizers.

  Args:
    learning_rate: Initial value for the learning rate:
      either a floating point value,
      or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
      Defaults to 0.001.
      Note that `Adadelta` tends to benefit from higher initial learning rate
      values compared to other optimizers.
      To match the exact form in the original paper, use 1.0.
    rho: A `Tensor` or a floating point value. The decay rate.
    epsilon: Small floating point value used to maintain numerical stability.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Adadelta"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm and represents
      the maximum norm of each parameter;
      `"clipvalue"` (float) clips gradient by value and represents the
      maximum absolute value of each parameter.

  Reference:
    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               rho=0.95,
               epsilon=1e-7,
               name='Adadelta',
               **kwargs):
    super(Adadelta, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('rho', rho)
    self.epsilon = epsilon or backend_config.epsilon()

  def _create_slots(self, var_list):
    # Separate for-loops to respect the ordering of slot variables from v1.
    for v in var_list:
      self.add_slot(v, 'accum_grad')
    for v in var_list:
      self.add_slot(v, 'accum_var')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adadelta, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(
        dict(
            epsilon=ops.convert_to_tensor_v2_with_dispatch(
                self.epsilon, var_dtype),
            rho=array_ops.identity(self._get_hyper('rho', var_dtype))))

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(Adadelta, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
    return gen_training_ops.ResourceApplyAdadelta(
        var=var.handle,
        accum=accum_grad.handle,
        accum_update=accum_var.handle,
        lr=coefficients['lr_t'],
        rho=coefficients['rho'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
    return gen_training_ops.ResourceSparseApplyAdadelta(
        var=var.handle,
        accum=accum_grad.handle,
        accum_update=accum_var.handle,
        lr=coefficients['lr_t'],
        rho=coefficients['rho'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(Adadelta, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'rho': self._serialize_hyperparameter('rho'),
        'epsilon': self.epsilon,
    })
    return config
