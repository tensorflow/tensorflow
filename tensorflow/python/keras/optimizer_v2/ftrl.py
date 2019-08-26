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
"""Ftrl-proximal for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.Ftrl')
class Ftrl(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the FTRL algorithm.

  See Algorithm 1 of this [paper](
  https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
  This version has support for both online L2 (the L2 penalty given in the paper
  above) and shrinkage-type L2 (which is the addition of an L2 penalty to the
  loss function).

  Initialization:
  $$t = 0$$
  $$n_{0} = 0$$
  $$\sigma_{0} = 0$$
  $$z_{0} = 0$$

  Update ($$i$$ is variable index):
  $$t = t + 1$$
  $$n_{t,i} = n_{t-1,i} + g_{t,i}^{2}$$
  $$\sigma_{t,i} = (\sqrt{n_{t,i}} - \sqrt{n_{t-1,i}}) / \alpha$$
  $$z_{t,i} = z_{t-1,i} + g_{t,i} - \sigma_{t,i} * w_{t,i}$$
  $$w_{t,i} = - ((\beta+\sqrt{n+{t}}) / \alpha + \lambda_{2})^{-1} * (z_{i} -
               sgn(z_{i}) * \lambda_{1}) if \abs{z_{i}} > \lambda_{i} else 0$$

  Check the documentation for the l2_shrinkage_regularization_strength
  parameter for more details when shrinkage is enabled, where gradient is
  replaced with gradient_with_shrinkage.
  """

  def __init__(self,
               learning_rate=0.001,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               name='Ftrl',
               l2_shrinkage_regularization_strength=0.0,
               **kwargs):
    r"""Construct a new FTRL optimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for
        a fixed learning rate.
      initial_accumulator_value: The starting value for accumulators.
        Only zero or positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Ftrl".
      l2_shrinkage_regularization_strength: A float value, must be greater than
        or equal to zero. This differs from L2 above in that the L2 above is a
        stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
        The FTRL formulation can be written as:
        w_{t+1} = argmin_w(\hat{g}_{1:t}w + L1*||w||_1 + L2*||w||_2^2), where
        \hat{g} = g + (2*L2_shrinkage*w), and g is the gradient of the loss
        function w.r.t. the weights w.
        Specifically, in the absence of L1 regularization, it is equivalent to
        the following update rule:
        w_{t+1} = w_t - lr_t / (1 + 2*L2*lr_t) * g_t -
                  2*L2_shrinkage*lr_t / (1 + 2*L2*lr_t) * w_t
        where lr_t is the learning rate at t.
        When input is sparse shrinkage will only happen on the active weights.\
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    Raises:
      ValueError: If one of the arguments is invalid.

    References
      See [paper]
        (https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)
    """
    super(Ftrl, self).__init__(name, **kwargs)

    if initial_accumulator_value < 0.0:
      raise ValueError(
          'initial_accumulator_value %f needs to be positive or zero' %
          initial_accumulator_value)
    if learning_rate_power > 0.0:
      raise ValueError('learning_rate_power %f needs to be negative or zero' %
                       learning_rate_power)
    if l1_regularization_strength < 0.0:
      raise ValueError(
          'l1_regularization_strength %f needs to be positive or zero' %
          l1_regularization_strength)
    if l2_regularization_strength < 0.0:
      raise ValueError(
          'l2_regularization_strength %f needs to be positive or zero' %
          l2_regularization_strength)
    if l2_shrinkage_regularization_strength < 0.0:
      raise ValueError(
          'l2_shrinkage_regularization_strength %f needs to be positive'
          ' or zero' % l2_shrinkage_regularization_strength)

    self._set_hyper('learning_rate', learning_rate)
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('learning_rate_power', learning_rate_power)
    self._set_hyper('l1_regularization_strength', l1_regularization_strength)
    self._set_hyper('l2_regularization_strength', l2_regularization_strength)
    self._initial_accumulator_value = initial_accumulator_value
    self._l2_shrinkage_regularization_strength = (
        l2_shrinkage_regularization_strength)

  def _create_slots(self, var_list):
    # Create the "accum" and "linear" slots.
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = init_ops.constant_initializer(
          self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, 'accumulator', init)
      self.add_slot(var, 'linear')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Ftrl, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(dict(
        learning_rate_power=array_ops.identity(
            self._get_hyper('learning_rate_power', var_dtype)),
        l1_regularization_strength=array_ops.identity(
            self._get_hyper('l1_regularization_strength', var_dtype)),
        l2_regularization_strength=array_ops.identity(
            self._get_hyper('l2_regularization_strength', var_dtype)),
        l2_shrinkage_regularization_strength=math_ops.cast(
            self._l2_shrinkage_regularization_strength, var_dtype)
        ))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum = self.get_slot(var, 'accumulator')
    linear = self.get_slot(var, 'linear')

    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.resource_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)
    else:
      return training_ops.resource_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['l2_shrinkage_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum = self.get_slot(var, 'accumulator')
    linear = self.get_slot(var, 'linear')

    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.resource_sparse_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)
    else:
      return training_ops.resource_sparse_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['l2_shrinkage_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)

  def get_config(self):
    config = super(Ftrl, self).get_config()
    config.update({
        'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
        'decay':
            self._serialize_hyperparameter('decay'),
        'initial_accumulator_value':
            self._initial_accumulator_value,
        'learning_rate_power':
            self._serialize_hyperparameter('learning_rate_power'),
        'l1_regularization_strength':
            self._serialize_hyperparameter('l1_regularization_strength'),
        'l2_regularization_strength':
            self._serialize_hyperparameter('l2_regularization_strength'),
        'l2_shrinkage_regularization_strength':
            self._l2_shrinkage_regularization_strength,
    })
    return config
