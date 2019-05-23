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

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util.tf_export import keras_export


class _UnwrapPreventer(object):
  """Wrapper that DistributionStrategy will not unwrap.

  Typically, DistributionStrategy will unwrap values when going from a cross-
  replica context to a replica context via `call_for_each_replica`. This class
  is a wrapper that DistributionStrategy will not unwrap, so it can be used to
  prevent it from unwrapping a value.

  TODO(reedwm): Find/implement a better way of preventing values from being
  unwrapped by DistributionStrategy
  """

  def __init__(self, value):
    self.value = value


def scale_loss(loss, loss_scale):
  """Scales the loss by the loss scale."""
  if callable(loss):
    return lambda: loss() * loss_scale
  else:
    return loss * loss_scale


def unscale_grads(grads, loss_scale):
  """Unscales the gradients by the loss scale."""
  loss_scale_reciprocal = 1. / loss_scale
  return [g * loss_scale_reciprocal if g is not None else None for g in grads]


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

  The loss scale can either be a fixed constant, chosen by the user, or be
  dynamically determined. Dynamically determining the loss scale is convenient
  as a loss scale does not have to be explicitly chosen. However it reduces
  performance.

  This optimizer wraps another optimizer and applies loss scaling to it via a
  `LossScale`. Loss scaling is applied whenever gradients are
  computed, either through `minimize()` or `get_gradients()`.
  """

  def __init__(self, opt, loss_scale):
    """Initializes this loss scale optimizer.

    Args:
      opt: The Optimizer instance to wrap.
      loss_scale: The loss scale to scale the loss and gradients. This can
        either be an int/float to use a fixed loss scale, the string "dynamic"
        to use dynamic loss scaling, or an instance of a LossScale. The string
        "dynamic" equivalent to passing `DynamicLossScale()`, and passing an
        int/float is equivalent to passing a FixedLossScale with the given loss
        scale.
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
    self._loss_scale = loss_scale_module.get(loss_scale)
    for weight in loss_scale_module.get_loss_scale_weights(self._loss_scale):
      # We cannot call `track_variable` in the LossScale class itself, because a
      # file outside of Keras cannot depend on a Keras file. Calling it here
      # instead is OK, because a variable only needs to be tracked if used with
      # a Keras class, and the only way to use LossScale with a Keras class is
      # through the LossScaleOptimizer.
      backend.track_variable(weight)
    self._track_trackable(self._optimizer, 'base_optimizer')
    self._track_trackable(self._loss_scale, 'loss_scale')

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    loss = scale_loss(loss, self._loss_scale())
    grads_and_vars = self._optimizer._compute_gradients(loss, var_list,  # pylint: disable=protected-access
                                                        grad_loss)
    grads = [g for g, _ in grads_and_vars]
    variables = [v for _, v in grads_and_vars]
    unscaled_grads = unscale_grads(grads, self._loss_scale())
    return list(zip(unscaled_grads, variables))

  def get_gradients(self, loss, params):
    loss = scale_loss(loss, self._loss_scale())
    grads = self._optimizer.get_gradients(loss, params)
    return unscale_grads(grads, self._loss_scale())

  def apply_gradients(self, grads_and_vars, name=None):
    if distribution_strategy_context.in_cross_replica_context():
      raise ValueError('apply_gradients() must be called in a replica context.')
    grads_and_vars = tuple(grads_and_vars)
    return distribution_strategy_context.get_replica_context().merge_call(
        self._apply_gradients_cross_replica, args=(grads_and_vars, name))

  def _apply_gradients_cross_replica(self, distribution, grads_and_vars, name):
    grads = [g for g, _ in grads_and_vars]
    loss_scale_update_op, should_apply_grads = self._loss_scale.update(grads)

    def apply_fn():
      # We do not want DistributionStrategy to unwrap any MirroredVariables in
      # grads_and_vars, because even in a replica context, the wrapped optimizer
      # expects mirrored variables. So we wrap grads_and_vars with an
      # _UnwrapPreventer, preventing DistributionStrategy from unwrapping the
      # MirroredVariables.
      wrapped_grads_and_vars = _UnwrapPreventer(grads_and_vars)
      return distribution.extended.call_for_each_replica(
          self._apply_gradients, args=(wrapped_grads_and_vars, name))

    # Note: We must call this cond() in a cross-replica context.
    # DistributionStrategy does not support having a cond in a replica context
    # with a branch that calls `merge_call`, and self._optimizer.apply_gradients
    # calls `merge_call`.
    maybe_apply_op = smart_cond.smart_cond(should_apply_grads,
                                           apply_fn,
                                           control_flow_ops.no_op)
    return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)

  def _apply_gradients(self, wrapped_grads_and_vars, name):
    grads_and_vars = wrapped_grads_and_vars.value
    return self._optimizer.apply_gradients(grads_and_vars, name)

  @property
  def learning_rate(self):
    return self._optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, lr):
    self._optimizer.learning_rate = lr

  def get_slot_names(self):
    """A list of names for this optimizer's slots."""
    return self._optimizer.get_slot_names()

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
