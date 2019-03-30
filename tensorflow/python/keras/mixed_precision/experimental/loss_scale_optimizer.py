# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the loss scaling optimizer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.mixed_precision.experimental.LossScaleOptimizer')
class LossScaleOptimizer(optimizer_v2.OptimizerV2):
  """An optimizer that applies loss scaling.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:

  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used. By
  multiplying the loss, each intermediate gradient will have the same multiplier
  applied.

  This optimizer wraps another optimizer and applies loss scaling to it. Loss
  scaling is applied whenever gradients are computed, either through
  `minimize()` or `get_gradients()`.
  """

  def __init__(self, opt, loss_scale):
    """Initializes this loss scale optimizer.

    Args:
      opt: The Optimizer instance to wrap.
      loss_scale: A float loss scale to scale loss and gradients by
    """
    if not isinstance(opt, optimizer_v2.OptimizerV2):
      raise ValueError('"opt" must be an instance of OptimizerV2, but got: %s'
                       % opt)
    if hasattr(opt, 'clipnorm'):
      raise ValueError('LossScaleOptimizer does not support wrapping '
                       'optimizers with a clipnorm. Optimizer %s has clipnorm '
                       '%s' % (opt, opt.clipnorm))

    if hasattr(opt, 'clipvalue'):
      raise ValueError('LossScaleOptimizer does not support wrapping '
                       'optimizers with a clipvalue. Optimizer %s has '
                       'clipvalue %s' % (opt, opt.clipvalue))

    self._optimizer = opt
    self._loss_scale = float(loss_scale)

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    loss = self._scale_loss(loss)
    grads_and_vars = self._optimizer._compute_gradients(loss, var_list,  # pylint: disable=protected-access
                                                        grad_loss)
    grads = [g for g, _ in grads_and_vars]
    variables = [v for _, v in grads_and_vars]
    scaled_grads = self._scale_grads(grads)
    return list(zip(scaled_grads, variables))

  def get_gradients(self, loss, params):
    loss = self._scale_loss(loss)
    grads = self._optimizer.get_gradients(loss, params)
    return self._scale_grads(grads)

  def _scale_loss(self, loss):
    # The loss is callable for `_compute_gradients`, but not `get_gradients`.
    if callable(loss):
      return lambda: loss() * self._loss_scale
    else:
      return loss * self._loss_scale

  def _scale_grads(self, grads):
    loss_scale_reciprocal = 1 / self._loss_scale
    return [None if g is None else g * loss_scale_reciprocal for g in grads]

  def apply_gradients(self, grads_and_vars, name=None):
    return self._optimizer.apply_gradients(grads_and_vars, name)

  @property
  def learning_rate(self):
    return self._optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, lr):
    self._optimizer.learning_rate = lr

  # TODO(reedwm): Support dynamic loss scaling.

  # TODO(reedwm): Maybe merge this class's functionality into OptimizerV2.

  # TODO(reedwm): Maybe throw an error if mixed precision is used without this
  # optimizer being used.

  # TODO(reedwm): Define __getattr__ to delegate all methods/attributes to
  # self._optimizer. This is tricky because the super class overrides
  # __getattribute__.

  # TODO(reedwm): Implement get_config and from_config. This will first require
  # implementing deserialization support for OptimizerV2.
  def get_config(self):
    raise NotImplementedError('get_config() is not yet implemented for '
                              'LossScaleOptimizers')

  @classmethod
  def from_config(cls, config, custom_objects=None):
    raise NotImplementedError('from_config() is not yet implemented for '
                              'LossScaleOptimizers')
