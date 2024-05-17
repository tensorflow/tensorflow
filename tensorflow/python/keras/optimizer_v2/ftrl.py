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
"""Ftrl-proximal optimizer implementation."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops


class Ftrl(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the FTRL algorithm.

  "Follow The Regularized Leader" (FTRL) is an optimization algorithm developed
  at Google for click-through rate prediction in the early 2010s. It is most
  suitable for shallow models with large and sparse feature spaces.
  The algorithm is described by
  [McMahan et al., 2013](https://research.google.com/pubs/archive/41159.pdf).
  The Keras version has support for both online L2 regularization
  (the L2 regularization described in the paper
  above) and shrinkage-type L2 regularization
  (which is the addition of an L2 penalty to the loss function).

  Initialization:

  ```python
  n = 0
  sigma = 0
  z = 0
  ```

  Update rule for one variable `w`:

  ```python
  prev_n = n
  n = n + g ** 2
  sigma = (sqrt(n) - sqrt(prev_n)) / lr
  z = z + g - sigma * w
  if abs(z) < lambda_1:
    w = 0
  else:
    w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
  ```

  Notation:

  - `lr` is the learning rate
  - `g` is the gradient for the variable
  - `lambda_1` is the L1 regularization strength
  - `lambda_2` is the L2 regularization strength

  Check the documentation for the `l2_shrinkage_regularization_strength`
  parameter for more details when shrinkage is enabled, in which case gradient
  is replaced with a gradient with shrinkage.

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.
    learning_rate_power: A float value, must be less or equal to zero.
      Controls how the learning rate decreases during training. Use zero for
      a fixed learning rate.
    initial_accumulator_value: The starting value for accumulators.
      Only zero or positive values are allowed.
    l1_regularization_strength: A float value, must be greater than or
      equal to zero. Defaults to 0.0.
    l2_regularization_strength: A float value, must be greater than or
      equal to zero. Defaults to 0.0.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Ftrl"`.
    l2_shrinkage_regularization_strength: A float value, must be greater than
      or equal to zero. This differs from L2 above in that the L2 above is a
      stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
      When input is sparse shrinkage will only happen on the active weights.
    beta: A float value, representing the beta value from the paper.
      Defaults to 0.0.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Reference:
    - [McMahan et al., 2013](
      https://research.google.com/pubs/archive/41159.pdf)
  """

  def __init__(self,
               learning_rate=0.001,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               name='Ftrl',
               l2_shrinkage_regularization_strength=0.0,
               beta=0.0,
               **kwargs):
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
    self._set_hyper('beta', beta)
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
    apply_state[(var_device, var_dtype)].update(
        dict(
            learning_rate_power=array_ops.identity(
                self._get_hyper('learning_rate_power', var_dtype)),
            l1_regularization_strength=array_ops.identity(
                self._get_hyper('l1_regularization_strength', var_dtype)),
            l2_regularization_strength=array_ops.identity(
                self._get_hyper('l2_regularization_strength', var_dtype)),
            beta=array_ops.identity(self._get_hyper('beta', var_dtype)),
            l2_shrinkage_regularization_strength=math_ops.cast(
                self._l2_shrinkage_regularization_strength, var_dtype)))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # Adjust L2 regularization strength to include beta to avoid the underlying
    # TensorFlow ops needing to include it.
    adjusted_l2_regularization_strength = (
        coefficients['l2_regularization_strength'] + coefficients['beta'] /
        (2. * coefficients['lr_t']))

    accum = self.get_slot(var, 'accumulator')
    linear = self.get_slot(var, 'linear')

    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.ResourceApplyFtrl(
          var=var.handle,
          accum=accum.handle,
          linear=linear.handle,
          grad=grad,
          lr=coefficients['lr_t'],
          l1=coefficients['l1_regularization_strength'],
          l2=adjusted_l2_regularization_strength,
          lr_power=coefficients['learning_rate_power'],
          use_locking=self._use_locking)
    else:
      return gen_training_ops.ResourceApplyFtrlV2(
          var=var.handle,
          accum=accum.handle,
          linear=linear.handle,
          grad=grad,
          lr=coefficients['lr_t'],
          l1=coefficients['l1_regularization_strength'],
          l2=adjusted_l2_regularization_strength,
          l2_shrinkage=coefficients['l2_shrinkage_regularization_strength'],
          lr_power=coefficients['learning_rate_power'],
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # Adjust L2 regularization strength to include beta to avoid the underlying
    # TensorFlow ops needing to include it.
    adjusted_l2_regularization_strength = (
        coefficients['l2_regularization_strength'] + coefficients['beta'] /
        (2. * coefficients['lr_t']))

    accum = self.get_slot(var, 'accumulator')
    linear = self.get_slot(var, 'linear')

    if self._l2_shrinkage_regularization_strength <= 0.0:
      return gen_training_ops.ResourceSparseApplyFtrl(
          var=var.handle,
          accum=accum.handle,
          linear=linear.handle,
          grad=grad,
          indices=indices,
          lr=coefficients['lr_t'],
          l1=coefficients['l1_regularization_strength'],
          l2=adjusted_l2_regularization_strength,
          lr_power=coefficients['learning_rate_power'],
          use_locking=self._use_locking)
    else:
      return gen_training_ops.ResourceSparseApplyFtrlV2(
          var=var.handle,
          accum=accum.handle,
          linear=linear.handle,
          grad=grad,
          indices=indices,
          lr=coefficients['lr_t'],
          l1=coefficients['l1_regularization_strength'],
          l2=adjusted_l2_regularization_strength,
          l2_shrinkage=coefficients['l2_shrinkage_regularization_strength'],
          lr_power=coefficients['learning_rate_power'],
          use_locking=self._use_locking)

  def get_config(self):
    config = super(Ftrl, self).get_config()
    config.update({
        'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
        'decay':
            self._initial_decay,
        'initial_accumulator_value':
            self._initial_accumulator_value,
        'learning_rate_power':
            self._serialize_hyperparameter('learning_rate_power'),
        'l1_regularization_strength':
            self._serialize_hyperparameter('l1_regularization_strength'),
        'l2_regularization_strength':
            self._serialize_hyperparameter('l2_regularization_strength'),
        'beta':
            self._serialize_hyperparameter('beta'),
        'l2_shrinkage_regularization_strength':
            self._l2_shrinkage_regularization_strength,
    })
    return config
