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

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision.experimental import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.training.tracking import base as trackable
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


class _DelegatingTrackableMixin(object):
  """A mixin that delegates all Trackable methods to another trackable object.

  This class must be used with multiple inheritance. A class that subclasses
  Trackable can also subclass this class, which causes all Trackable methods to
  be delegated to the trackable object passed in the constructor.

  A subclass can use this mixin to appear as if it were the trackable passed to
  the constructor, from a Checkpoint's perspective. LossScaleOptimizer uses this
  mixin, so that the checkpoint format for a LossScaleOptimizer is identical to
  the checkpoint format for a normal optimizer. This allows a model to be saved
  with a normal Optimizer and restored with a LossScaleOptimizer, or vice versa.
  The only difference in checkpoint format is that the loss scale is also saved
  with a LossScaleOptimizer.
  """

  def __init__(self, trackable_obj):
    self._trackable = trackable_obj

  # pylint: disable=protected-access
  @property
  def _setattr_tracking(self):
    return self._trackable._setattr_tracking

  @_setattr_tracking.setter
  def _setattr_tracking(self, value):
    self._trackable._setattr_tracking = value

  @property
  def _update_uid(self):
    return self._trackable._update_uid

  @_update_uid.setter
  def _update_uid(self, value):
    self._trackable._update_uid = value

  @property
  def _unconditional_checkpoint_dependencies(self):
    return self._trackable._unconditional_checkpoint_dependencies

  @property
  def _unconditional_dependency_names(self):
    return self._trackable._unconditional_dependency_names

  @property
  def _name_based_restores(self):
    return self._trackable._name_based_restores

  def _maybe_initialize_trackable(self):
    return self._trackable._maybe_initialize_trackable()

  @property
  def _object_identifier(self):
    return self._trackable._object_identifier

  @property
  def _tracking_metadata(self):
    return self._trackable._tracking_metadata

  def _no_dependency(self, value):
    return self._trackable._no_dependency(value)

  def _name_based_attribute_restore(self, checkpoint):
    return self._trackable._name_based_attribute_restore(checkpoint)

  @property
  def _checkpoint_dependencies(self):
    return self._trackable._checkpoint_dependencies

  @property
  def _deferred_dependencies(self):
    return self._trackable._deferred_dependencies

  def _lookup_dependency(self, name):
    self._trackable._lookup_dependency(name)

  def _add_variable_with_custom_getter(self,
                                       name,
                                       shape=None,
                                       dtype=dtypes.float32,
                                       initializer=None,
                                       getter=None,
                                       overwrite=False,
                                       **kwargs_for_getter):
    return self._trackable._add_variable_with_custom_getter(
        name, shape, dtype, initializer, getter, overwrite, **kwargs_for_getter)

  def _preload_simple_restoration(self, name, shape):
    return self._trackable._preload_simple_restoration(name, shape)

  def _track_trackable(self, trackable, name, overwrite=False):  # pylint: disable=redefined-outer-name
    return self._trackable._track_trackable(trackable, name, overwrite)

  def _handle_deferred_dependencies(self, name, trackable):  # pylint: disable=redefined-outer-name
    return self._trackable._handle_deferred_dependencies(name, trackable)

  def _restore_from_checkpoint_position(self, checkpoint_position):
    return self._trackable._restore_from_checkpoint_position(
        checkpoint_position)

  def _single_restoration_from_checkpoint_position(self, checkpoint_position,
                                                   visit_queue):
    return self._trackable._single_restoration_from_checkpoint_position(
        checkpoint_position, visit_queue)

  def _gather_saveables_for_checkpoint(self):
    return self._trackable._gather_saveables_for_checkpoint()

  def _list_extra_dependencies_for_serialization(self, serialization_cache):
    return self._trackable._list_extra_dependencies_for_serialization(
        serialization_cache)

  def _list_functions_for_serialization(self, serialization_cache):
    return self._trackable._list_functions_for_serialization(
        serialization_cache)
  # pylint: enable=protected-access


@keras_export('keras.mixed_precision.experimental.LossScaleOptimizer')
class LossScaleOptimizer(_DelegatingTrackableMixin, optimizer_v2.OptimizerV2):
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

  >>> opt = tf.keras.optimizers.SGD(0.25)
  >>> opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt,
  ...                                                                "dynamic")
  >>> var = tf.Variable(1.)
  >>> loss_fn = lambda: var ** 2
  >>> # 'minimize' applies loss scaling to the loss and updates the loss sale.
  >>> opt.minimize(loss_fn, var_list=var)
  >>> var.numpy()
  0.5

  If a `tf.GradientTape` is used to compute gradients instead of
  `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, the loss
  and gradients must be scaled manually. This can be done by calling
  `LossScaleOptimizer.get_scaled_loss` before passing the loss to
  `tf.GradientTape`, and `LossScaleOptimizer.get_unscaled_gradients` after
  computing the gradients with `tf.GradientTape`. For example:

  >>> with tf.GradientTape() as tape:
  ...   loss = loss_fn()
  ...   scaled_loss = opt.get_scaled_loss(loss)
  >>> scaled_grad = tape.gradient(scaled_loss, var)
  >>> (grad,) = opt.get_unscaled_gradients([scaled_grad])
  >>> opt.apply_gradients([(grad, var)])  # Loss scale is updated here
  >>> var.numpy()
  0.25
  """

  _HAS_AGGREGATE_GRAD = True

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
    if optimizer.clipnorm is not None:
      raise ValueError('LossScaleOptimizer does not support wrapping '
                       'optimizers with a clipnorm. Optimizer %s has clipnorm '
                       '%s' % (optimizer, optimizer.clipnorm))

    if optimizer.clipvalue is not None:
      raise ValueError('LossScaleOptimizer does not support wrapping '
                       'optimizers with a clipvalue. Optimizer %s has '
                       'clipvalue %s' % (optimizer, optimizer.clipvalue))
    self._raise_if_strategy_unsupported()

    self.clipnorm = None
    self.clipvalue = None

    self._optimizer = optimizer
    self._loss_scale = keras_loss_scale_module.get(loss_scale)
    if self._loss_scale is None:
      raise ValueError('loss_scale cannot be None.')

    # We don't call super().__init__, since we do not want to call OptimizerV2's
    # constructor.
    _DelegatingTrackableMixin.__init__(self, self._optimizer)

    for weight in loss_scale_module.get_loss_scale_weights(self._loss_scale):
      # We cannot call `track_variable` in the LossScale class itself, because a
      # file outside of Keras cannot depend on a Keras file. Calling it here
      # instead is OK, because a variable only needs to be tracked if used with
      # a Keras class, and the only way to use LossScale with a Keras class is
      # through the LossScaleOptimizer.
      backend.track_variable(weight)
    self._track_trackable(self._loss_scale, 'loss_scale')

    # Needed because the superclass's __getattribute__ checks this.
    self._hyper = {}

    # To support restoring TensorFlow 2.2 checkpoints.
    self._track_trackable(FakeOptimizerForRestoration(self._optimizer),
                          'base_optimizer')

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
    return [
        _multiply_gradient(g, loss_scale_reciprocal) if g is not None else None
        for g in grads
    ]

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

  def _create_all_weights(self, var_list):
    self._optimizer._create_all_weights(var_list)    # pylint: disable=protected-access

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    if distribution_strategy_context.in_cross_replica_context():
      raise ValueError('apply_gradients() must be called in a replica context.')
    # We check for the strategy here despite already checking in the constructor
    # as frequently the optimizer is created outside the strategy's scope.
    self._raise_if_strategy_unsupported()

    grads_and_vars = tuple(grads_and_vars)
    return distribution_strategy_context.get_replica_context().merge_call(
        self._apply_gradients_cross_replica,
        args=(grads_and_vars, name, experimental_aggregate_gradients))

  def _apply_gradients_cross_replica(self, distribution, grads_and_vars, name,
                                     experimental_aggregate_gradients):
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
          self._apply_gradients,
          args=(grads, wrapped_vars, name, experimental_aggregate_gradients))

    # Note: We must call this cond() in a cross-replica context.
    # DistributionStrategy does not support having a cond in a replica context
    # with a branch that calls `merge_call`, and self._optimizer.apply_gradients
    # calls `merge_call`.
    maybe_apply_op = smart_cond.smart_cond(should_apply_grads,
                                           apply_fn,
                                           control_flow_ops.no_op)
    return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)

  def _apply_gradients(self, grads, wrapped_vars, name,
                       experimental_aggregate_gradients):
    # TODO(reedwm): This will raise a fairly cryptic error message if
    # self._optimizer.apply_gradients does not take
    # experimental_aggregate_gradients.
    return self._optimizer.apply_gradients(
        list(zip(grads, wrapped_vars.value)), name,
        experimental_aggregate_gradients=experimental_aggregate_gradients)

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

  def _raise_if_strategy_unsupported(self):
    if not strategy_supports_loss_scaling():
      strategy = distribution_strategy_context.get_strategy()
      if isinstance(strategy,
                    (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
        raise ValueError(
            'Loss scaling is not supported with TPUStrategy. Loss scaling is '
            'unnecessary with TPUs, since they support bfloat16 instead of '
            'float16 and bfloat16 does not require loss scaling. You should '
            'remove the use of the LossScaleOptimizer when TPUs are used.')
      else:
        raise ValueError('Loss scaling is not supported with the '
                         'tf.distribute.Strategy: %s. Try using a different '
                         'Strategy, e.g. a MirroredStrategy' %
                         strategy.__class__.__name__)

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

  def _aggregate_gradients(self, grads_and_vars):
    return self._optimizer._aggregate_gradients(grads_and_vars)  # pylint: disable=protected-access

  def _restore_slot_variable(self, slot_name, variable, slot_variable):
    return self._optimizer._restore_slot_variable(slot_name, variable,  # pylint: disable=protected-access
                                                  slot_variable)

  def _create_or_restore_slot_variable(self, slot_variable_position, slot_name,
                                       variable):
    return self._optimizer._create_or_restore_slot_variable(  # pylint: disable=protected-access
        slot_variable_position, slot_name, variable)

  def get_slot(self, var, slot_name):
    return self._optimizer.get_slot(var, slot_name)

  def add_slot(self, var, slot_name, initializer='zeros'):
    return self._optimizer.add_slot(var, slot_name, initializer)

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

  # Trackable delegations: Delegate all Trackable methods to the wrapped
  # optimizer. This is so the checkpoint format for a LossScaleOptimizer is
  # identical to the checkpoint format for a normal optimizer, except the loss
  # scale is stored in the checkpoint.


class FakeOptimizerForRestoration(trackable.Trackable):
  """A fake optimizer used to support restoring TensorFlow 2.2 checkpoints.

  The checkpoint format for LossScaleOptimizers changed after TF 2.2. This class
  exists to support restoring TF 2.2 checkpoints in newer version of TensorFlow.

  In TF 2.2, LossScaleOptimizer would track the wrapped optimizer by calling the
  following in LossScaleOptimizer.__init__

  ```
  self._track_trackable(self._optimizer, 'base_optimizer')
  ```

  This means a dependency from the LossScaleOptimizer to the wrapped optimizer
  would be stored in the checkpoint. However now, the checkpoint format with a
  LossScaleOptimizer is the same as the format without a LossScaleOptimizer,
  except the loss scale is also stored. This means there is no dependency from
  the LossScaleOptimizer to the wrapped optimizer. Instead, the
  LossScaleOptimizer acts as if it is the wrapped optimizer, from a checkpoint's
  perspective, by overriding all Trackable methods and delegating them to the
  wrapped optimizer.

  To allow restoring TF 2.2. checkpoints, LossScaleOptimizer adds a dependency
  on this class instead of the inner optimizer. When restored, this class will
  instead restore the slot variables of the inner optimizer. Since this class
  has no variables, it does not affect the checkpoint when saved.
  """

  def __init__(self, optimizer):
    self._optimizer = optimizer

  def get_slot_names(self):
    return self._optimizer.get_slot_names()

  def _create_or_restore_slot_variable(self, slot_variable_position, slot_name,
                                       variable):
    return self._optimizer._create_or_restore_slot_variable(  # pylint: disable=protected-access
        slot_variable_position, slot_name, variable)


# pylint: disable=protected-access
mixed_precision._register_wrapper_optimizer_cls(optimizer_v2.OptimizerV2,
                                                LossScaleOptimizer)


def _multiply_gradient(gradient, scale):
  """Multiply a (possibly sparse) gradient by the given scale factor."""
  scale = math_ops.cast(scale, gradient.dtype)
  if isinstance(gradient, ops.IndexedSlices):
    return ops.IndexedSlices(
        gradient.values * scale,
        gradient.indices,
        dense_shape=gradient.dense_shape)
  else:
    return gradient * scale


def strategy_supports_loss_scaling():
  """Returns True if the current Strategy supports loss scaling."""
  if not distribution_strategy_context.has_strategy():
    return True
  strategy = distribution_strategy_context.get_strategy()
  # Strategies are supported if either there is only one replica or if variables
  # are replicated per device. Otherwise, the current model.fit() implementation
  # and most custom training loops incorrectly unscale the gradients. Currently,
  # gradients are unscaled once per compute replica, but they should be unscaled
  # once per variable replica. When there is one variable replica for each
  # compute replica, this works fine, but otherwise issues will occur.
  # TODO(reedwm): Support all strategies.
  return isinstance(strategy, (
      collective_all_reduce_strategy.CollectiveAllReduceStrategy,
      collective_all_reduce_strategy.CollectiveAllReduceStrategyV1,
      one_device_strategy.OneDeviceStrategy,
      one_device_strategy.OneDeviceStrategyV1,
      mirrored_strategy.MirroredStrategy,
      mirrored_strategy.MirroredStrategyV1,
  ))
