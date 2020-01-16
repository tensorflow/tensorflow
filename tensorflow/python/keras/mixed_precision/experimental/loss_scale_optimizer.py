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
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision.experimental import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
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
  computed, either through `minimize()` or `get_gradients()`. The loss scale is
  updated via `LossScale.update()` whenever gradients are applied, either
  through `minimize()` or `apply_gradients()`. For example:

  ```python
  opt = tf.keras.optimizers.SGD(0.1)
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
  # 'minimize' applies loss scaling to the loss and updates the loss sale.
  opt.minimize(loss_fn)
  ```

  If a `tf.GradientTape` is used to compute gradients instead of
  `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, the loss
  and gradients must be scaled manually. This can be done by calling
  `LossScaleOptimizer.get_scaled_loss` before passing the loss to
  `tf.GradientTape`, and `LossScaleOptimizer.get_unscaled_gradients` after
  computing the gradients with `tf.GradientTape`. For example:

  ```python
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(...)
  vars = ...
  with tf.GradientTape() as tape:
    loss = ...
    scaled_loss = opt.get_scaled_loss(loss)
  scaled_grads = tape.gradient(scaled_loss, vars)
  grads = opt.get_unscaled_gradients(scaled_grads)
  opt.apply_gradients(zip(grads, vars))  # Loss scale will be updated here
  ```
  """

  def __init__(self, optimizer, loss_scale):
    """Initializes this loss scale optimizer.

    Args:
      optimizer: The Optimizer instance to wrap.
      loss_scale: The loss scale to scale the loss and gradients. This can
        either be an int/float to use a fixed loss scale, the string "dynamic"
        to use dynamic loss scaling, or an instance of a LossScale. The string
        "dynamic" equivalent to passing `DynamicLossScale()`, and passing an
        int/float is equivalent to passing a FixedLossScale with the given loss
        scale.
    """
    if not isinstance(optimizer, optimizer_v2.OptimizerV2):
      raise ValueError('"optimizer" must be an instance of OptimizerV2, but '
                       'got: %s' % optimizer)
    if hasattr(optimizer, 'clipnorm'):
      raise ValueError('LossScaleOptimizer does not support wrapping '
                       'optimizers with a clipnorm. Optimizer %s has clipnorm '
                       '%s' % (optimizer, optimizer.clipnorm))

    if hasattr(optimizer, 'clipvalue'):
      raise ValueError('LossScaleOptimizer does not support wrapping '
                       'optimizers with a clipvalue. Optimizer %s has '
                       'clipvalue %s' % (optimizer, optimizer.clipvalue))

    self._optimizer = optimizer
    self._loss_scale = keras_loss_scale_module.get(loss_scale)
    if self._loss_scale is None:
      raise ValueError('loss_scale cannot be None.')
    for weight in loss_scale_module.get_loss_scale_weights(self._loss_scale):
      # We cannot call `track_variable` in the LossScale class itself, because a
      # file outside of Keras cannot depend on a Keras file. Calling it here
      # instead is OK, because a variable only needs to be tracked if used with
      # a Keras class, and the only way to use LossScale with a Keras class is
      # through the LossScaleOptimizer.
      backend.track_variable(weight)
    self._track_trackable(self._optimizer, 'base_optimizer')
    self._track_trackable(self._loss_scale, 'loss_scale')

    # Needed because the superclass's __getattribute__ checks this.
    self._hyper = {}

  @property
  def loss_scale(self):
    """The `LossScale` instance associated with this optimizer."""
    return self._loss_scale

  def get_scaled_loss(self, loss):
    """Scales the loss by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for
    an example.

    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
        a tensor or a callable returning a tensor.

    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale()`.
    """
    loss_scale = self._loss_scale()
    if callable(loss):
      def new_loss():
        loss_val = loss()
        return loss_val * math_ops.cast(loss_scale, loss_val.dtype)
      return new_loss
    else:
      return loss * math_ops.cast(loss_scale, loss.dtype)

  def get_unscaled_gradients(self, grads):
    """Unscales the gradients by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to unscale the gradients
    after computing them with `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_scaled_loss` should also be called. See
    the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for an
    example.

    Args:
      grads: A list of tensors, each which will be divided by the loss scale.
        Can have None values, which are ignored.

    Returns:
      A new list the same size as `grads`, where every non-None value in `grads`
      is divided by `LossScaleOptimizer.loss_scale()`.
    """
    loss_scale = self._loss_scale()
    loss_scale_reciprocal = 1. / loss_scale
    return [g * math_ops.cast(loss_scale_reciprocal, g.dtype) if g is not None
            else None for g in grads]

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    loss = self.get_scaled_loss(loss)
    grads_and_vars = self._optimizer._compute_gradients(loss, var_list,  # pylint: disable=protected-access
                                                        grad_loss)
    grads = [g for g, _ in grads_and_vars]
    variables = [v for _, v in grads_and_vars]
    unscaled_grads = self.get_unscaled_gradients(grads)
    return list(zip(unscaled_grads, variables))

  def get_gradients(self, loss, params):
    loss = self.get_scaled_loss(loss)
    grads = self._optimizer.get_gradients(loss, params)
    return self.get_unscaled_gradients(grads)

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
      # expects mirrored variables. So we wrap the variables with an
      # _UnwrapPreventer, preventing DistributionStrategy from unwrapping the
      # MirroredVariables.
      wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])
      return distribution.extended.call_for_each_replica(
          self._apply_gradients, args=(grads, wrapped_vars, name))

    # Note: We must call this cond() in a cross-replica context.
    # DistributionStrategy does not support having a cond in a replica context
    # with a branch that calls `merge_call`, and self._optimizer.apply_gradients
    # calls `merge_call`.
    maybe_apply_op = smart_cond.smart_cond(should_apply_grads,
                                           apply_fn,
                                           control_flow_ops.no_op)
    return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)

  def _apply_gradients(self, grads, wrapped_vars, name):
    return self._optimizer.apply_gradients(list(zip(grads, wrapped_vars.value)),
                                           name)

  def get_config(self):
    serialized_optimizer = optimizers.serialize(self._optimizer)
    serialized_loss_scale = keras_loss_scale_module.serialize(self._loss_scale)
    return {
        'optimizer': serialized_optimizer,
        'loss_scale': serialized_loss_scale,
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()  # Make a copy, since we mutate config
    config['optimizer'] = optimizers.deserialize(
        config['optimizer'], custom_objects=custom_objects)
    config['loss_scale'] = keras_loss_scale_module.deserialize(
        config['loss_scale'], custom_objects=custom_objects)
    return cls(**config)

  # Delegations: We delegate most OptimizerV2 methods to the wrapped optimizer
  # below.

  @property
  def iterations(self):
    return self._optimizer.iterations

  @iterations.setter
  def iterations(self, variable):
    self._optimizer.iterations = variable

  def get_slot_names(self):
    return self._optimizer.get_slot_names()

  def variables(self):
    return self._optimizer.variables()

  @property
  def weights(self):
    return self._optimizer.weights

  def get_weights(self):
    return self._optimizer.get_weights()

  def set_weights(self, weights):
    return self._optimizer.set_weights(weights)

  # For the most part, we only expose methods in the base OptimizerV2, not
  # individual subclasses like Adam. However, although "learning_rate" and "lr"
  # properties are not part of the base OptimizerV2 class, they are part of most
  # subclasses, so we expose them here for convenience.

  @property
  def learning_rate(self):
    return self._optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, lr):
    self._optimizer.learning_rate = lr

  @property
  def lr(self):
    return self._optimizer.lr

  @lr.setter
  def lr(self, lr):
    self._optimizer.lr = lr

  def get_slot(self, var, slot_name):
    # We cannot implement get_slot for the following reason: When saving a
    # checkpoint, two optimizers cannot share slot variables. Since both the
    # LossScaleOptimizer and the wrapped optimizer (self and self._optimizer
    # respectively) are checkpointed, we cannot expose the wrapped optimizer's
    # slots in the LossScaleOptimizer. Otherwise, a checkpoint would believe
    # both optimizers share slot variables.
    raise AttributeError(
        'You cannot call get_slot on a LossScaleOptimizer. This limitation '
        'will be removed in the future.')

  def add_slot(self, var, slot_name, initializer='zeros'):
    # We disallow adding a slot for consistency with `get_slot`.
    raise AttributeError(
        'You cannot call add_slot on a LossScaleOptimizer. This limitation '
        'will be removed in the future.')

  # We do not override some OptimizerV2 methods. For each, we describe why we do
  # not delegate them to self._optimizer:
  # * get_updates: get_updates() calls get_gradients(). Since we override
  #   get_gradients(), we cannot delegate get_updates() to self._optimizer,
  #   otherwise the overridden get_gradients() method would not be called.
  #   Luckily, get_updates() does not access any OptimizerV2 fields, so
  #   inheriting the OptimizerV2 version works fine.
  # * minimize: We don't delegate for a similar as get_updates(): it calls
  #   both self._compute_gradients() and self.apply_gradients(), and both need
  #   to have the LossScaleOptimizer version called.

  # TODO(reedwm): Maybe merge this class's functionality into OptimizerV2.

  # TODO(reedwm): Maybe throw an error if mixed precision is used without this
  # optimizer being used.
