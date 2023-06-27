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
"""Contains LossScale classes."""
import abc

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated_endpoints('mixed_precision.experimental.LossScale',
                                  'train.experimental.LossScale')
@tf_export(
    v1=[
        'mixed_precision.LossScale',
        'mixed_precision.experimental.LossScale',
        'train.experimental.LossScale'
    ])
class LossScale(trackable.Trackable, metaclass=abc.ABCMeta):
  """Base class for all TF1 loss scales.

  This is an abstract base class, so you cannot instantiate it directly.
  Instead, use one of its concrete subclasses:
    * `tf.compat.v1.mixed_precision.DynamicLossScale`
    * `tf.compat.v1.mixed_precision.FixedLossScale`

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
  underflow in intermediate gradients when float16 tensors are used for mixed
  precision training. By multiplying the loss, each intermediate gradient will
  have the same multiplier applied.

  Instances of this class represent a loss scale. Calling instances of this
  class returns the loss scale as a scalar float32 tensor, while method
  `update()` updates the loss scale depending on the values of the gradients.
  Optimizers use instances of this class to scale loss and gradients.

  In most functions that accept a LossScale, you can also pass an int (such as
  8) to create a `FixedLossScale` or the string `"dynamic"` to create a dynamic
  loss scale.
  """

  def __init__(self):
    """Initializes the loss scale class."""
    self._weights = {}

  @abc.abstractmethod
  def __call__(self):
    """Returns the current loss scale as a scalar `float32` tensor."""
    pass

  @abc.abstractmethod
  def update(self, grads):
    """Updates the value of the loss scale.

    The loss scale will be potentially updated, based on the value of `grads`.
    The tensor returned by calling this class is only updated when this function
    is evaluated.

    In eager mode, this directly updates the loss scale, so that calling
    `__call__` will return the newly updated loss scale. In graph mode,
    this returns an op that, when evaluated, updates the loss scale.

    This function also returns a `should_apply_gradients` bool. If False,
    gradients should not be applied to the variables that step, as nonfinite
    gradients were found, and the loss scale has been be updated to reduce the
    chance of finding nonfinite gradients in the next step. Some loss scale
    classes will always return True, as they cannot adjust themselves in
    response to nonfinite gradients.

    When a DistributionStrategy is used, this function may only be called in a
    cross-replica context.

    Args:
      grads: A nested structure of unscaled gradients, each which is the
        gradient of the loss with respect to a weight. The gradients should have
        already been divided by the loss scale being before passed to this
        function. 'None' gradients are accepted, and are ignored.

    Returns:
      update_op: In eager mode, None. In graph mode, an op to update the loss
        scale.
      should_apply_gradients: Either a bool or a scalar boolean tensor. If
        False, the caller should skip applying `grads` to the variables this
        step.
    """
    pass

  def _add_weight(self, name, initial_value, dtype=None):
    """Adds a weight to this loss scale.

    Args:
      name: Variable name.
      initial_value: The variable's initial value.
      dtype: The type of the variable.

    Returns:
      A variable.

    Raises:
      RuntimeError: If a weight with `name` has already been added.
    """
    variable = variable_v1.VariableV1(
        initial_value=initial_value,
        name=name,
        dtype=dtype,
        trainable=False,
        use_resource=True,
        synchronization=variables.VariableSynchronization.AUTO,
        # Set aggregation to NONE, as loss scaling variables should never be
        # aggregated.
        aggregation=variables.VariableAggregation.NONE)
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key  # pylint: disable=protected-access

    key = (name, graph_key)
    if self._weights.get(key, None) is not None:
      raise RuntimeError('Duplicate variables detected. {}'.format(key))
    self._weights[key] = variable
    self._handle_deferred_dependencies(name=name, trackable=variable)
    return variable

  def _trackable_children(self,
                          save_type=trackable.SaveType.CHECKPOINT,
                          **kwargs):
    """From Trackable. Gather graph-specific weights to save."""
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key  # pylint: disable=protected-access
    weights = {}
    for (name, g), v in sorted(self._weights.items(), key=lambda i: i[0][0]):
      if g == graph_key:
        weights[name] = v
    weights.update(
        super(LossScale, self)._trackable_children(save_type, **kwargs))
    return weights

  def _lookup_dependency(self, name):
    """From Trackable. Find a weight in the current graph."""
    unconditional = super(LossScale, self)._lookup_dependency(name)
    if unconditional is not None:
      return unconditional
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key  # pylint: disable=protected-access
    return self._weights.get((name, graph_key), None)

  @abc.abstractmethod
  def get_config(self):
    """Returns the config of this loss scale."""
    pass

  @classmethod
  def from_config(cls, config):
    """Creates the LossScale from its config."""
    return cls(**config)


@deprecation.deprecated_endpoints('mixed_precision.experimental.FixedLossScale',
                                  'train.experimental.FixedLossScale')
@tf_export(
    v1=[
        'mixed_precision.FixedLossScale',
        'mixed_precision.experimental.FixedLossScale',
        'train.experimental.FixedLossScale'
    ])
class FixedLossScale(LossScale):
  """Loss scale with a fixed value.

  The loss scale is not updated for the lifetime of instances of this class.
  A given instance of this class always returns the same number when called.
  """

  @deprecation.deprecated(
      None, 'Use tf.keras.mixed_precision.LossScaleOptimizer instead. '
            'LossScaleOptimizer now has all the functionality of '
            'FixedLossScale')
  def __init__(self, loss_scale_value):
    """Creates the fixed loss scale.

    Args:
      loss_scale_value: A Python float. Its ideal value varies depending on
        models to run. Choosing a too small loss_scale might affect model
        quality; a too big loss_scale might cause inf or nan. There is no single
        right loss_scale to apply. There is no harm choosing a relatively big
        number as long as no nan or inf is encountered in training.

    Raises:
      ValueError: If loss_scale_value is less than 1.
    """
    super(FixedLossScale, self).__init__()
    if not isinstance(loss_scale_value, (int, float)):
      raise ValueError('loss_scale_value must be a Python int or float.')
    if loss_scale_value < 1:
      raise ValueError('loss_scale_value must be at least 1.')
    # It's important we do not create tensors in the constructor, as such
    # tensors might be on a different device or tf.function vs when the tensor
    # is used. This would hurt performance. Therefore, we do not create a tensor
    # from loss_scale_value, but instead leave it as a Python float.
    # TODO(reedwm): Also do not create tensors in the DynamicLossScale
    # constructor.
    self._loss_scale_value = float(loss_scale_value)

  def __call__(self):
    return ops.convert_to_tensor(self._loss_scale_value)

  def update(self, grads):
    del grads
    return control_flow_ops.no_op(), True

  def __repr__(self):
    return 'FixedLossScale(%s)' % self._loss_scale_value

  def get_config(self):
    return {'loss_scale_value': self._loss_scale_value}


def _is_all_finite(grads):
  """Returns a scalar boolean tensor indicating if all gradients are finite."""
  def raw_values(g):
    return g.values if isinstance(g, indexed_slices.IndexedSlices) else g

  is_finite_per_grad = [
      math_ops.reduce_all(math_ops.is_finite(raw_values(g)))
      for g in grads
      if g is not None
  ]
  return math_ops.reduce_all(is_finite_per_grad)


def _op_in_graph_mode(tensor):
  """Returns the tensor's op in graph mode, or the tensor in eager mode.

  This is useful because sometimes an op is needed in graph mode instead of a
  tensor. In eager mode, there are no ops.

  Args:
    tensor: A tensor.

  Returns:
    The tensor's op in graph mode. The tensor in eager mode.
  """
  if context.executing_eagerly():
    return tensor
  return tensor.op


def _assign_if_finite(var, value):
  """Assigns a value to a variable if the value is finite."""
  return cond.cond(
      math_ops.is_finite(value), lambda: _op_in_graph_mode(var.assign(value)),
      control_flow_ops.no_op)


@deprecation.deprecated_endpoints(
    'mixed_precision.experimental.DynamicLossScale',
    'train.experimental.DynamicLossScale')
@tf_export(
    v1=[
        'mixed_precision.DynamicLossScale',
        'mixed_precision.experimental.DynamicLossScale',
        'train.experimental.DynamicLossScale'
    ])
class DynamicLossScale(LossScale):
  """Loss scale that dynamically adjusts itself.

  Dynamic loss scaling works by adjusting the loss scale as training progresses.
  The goal is to keep the loss scale as high as possible without overflowing the
  gradients. As long as the gradients do not overflow, raising the loss scale
  never hurts.

  The algorithm starts by setting the loss scale to an initial value. Every N
  steps that the gradients are finite, the loss scale is increased by some
  factor. However, if a NaN or Inf gradient is found, the gradients for that
  step are not applied, and the loss scale is decreased by the factor. This
  process tends to keep the loss scale as high as possible without gradients
  overflowing.
  """

  @deprecation.deprecated(
      None, 'Use tf.keras.mixed_precision.LossScaleOptimizer instead. '
            'LossScaleOptimizer now has all the functionality of '
            'DynamicLossScale')
  def __init__(self,
               initial_loss_scale=2 ** 15,  # See docstring for why this is big.
               increment_period=2000,
               multiplier=2.):
    """Creates the dynamic loss scale.

    Args:
      initial_loss_scale: A Python float.  The loss scale to use at the
        beginning. It's better to start this at a very high number, because a
        loss scale that is too high gets lowered far more quickly than a loss
        scale that is too low gets raised. The default is 2 ** 15, which is
        approximately half the maximum float16 value.
      increment_period: Increases loss scale every `increment_period`
        consecutive steps that finite gradients are encountered. If a nonfinite
        gradient is encountered, the count is reset back to zero.
      multiplier: The multiplier to use when increasing or decreasing the loss
        scale.
    """
    super(DynamicLossScale, self).__init__()
    self._initial_loss_scale = float(initial_loss_scale)
    self._increment_period = int(increment_period)
    self._multiplier = float(multiplier)

    self._current_loss_scale = self._add_weight(
        name='current_loss_scale',
        dtype=dtypes.float32,
        initial_value=self._initial_loss_scale)
    # The number of consecutive steps with finite gradients since the last
    # nonfinite gradient or change in loss scale.
    self._num_good_steps = self._add_weight(
        name='good_steps', dtype=dtypes.int64, initial_value=0)

  @property
  def initial_loss_scale(self):
    return self._initial_loss_scale

  @property
  def increment_period(self):
    return self._increment_period

  @property
  def multiplier(self):
    return self._multiplier

  def __call__(self):
    return ops.convert_to_tensor(self._current_loss_scale)

  def update(self, grads):
    """Updates loss scale based on if gradients are finite in current step."""
    grads = nest.flatten(grads)
    if distribute_lib.has_strategy():
      distribution = distribute_lib.get_cross_replica_context()

      def get_is_finite(grads):
        is_finite = _is_all_finite(grads)
        # We cast to float, because we cannot reduce booleans with
        # DistributionStrategy.
        return math_ops.cast(is_finite, dtypes.float32)

      is_finite_float = distribution.extended.call_for_each_replica(
          get_is_finite, args=(grads,))
      reduced_is_finite_float = distribution.reduce(reduce_util.ReduceOp.SUM,
                                                    is_finite_float, axis=None)
      is_finite = math_ops.equal(reduced_is_finite_float,
                                 distribution.num_replicas_in_sync)
    else:
      is_finite = _is_all_finite(grads)

    def update_if_finite_grads():
      """Update assuming the gradients are finite."""

      def incr_loss_scale():
        new_loss_scale = self._current_loss_scale * self._multiplier
        return control_flow_ops.group(
            _assign_if_finite(self._current_loss_scale, new_loss_scale),
            self._num_good_steps.assign(0))

      return cond.cond(
          self._num_good_steps + 1 >= self._increment_period,
          incr_loss_scale, lambda: _op_in_graph_mode(
              self._num_good_steps.assign_add(1)))

    def update_if_not_finite_grads():
      """Update assuming the gradients are nonfinite."""

      new_loss_scale = math_ops.maximum(
          self._current_loss_scale / self._multiplier, 1)
      return control_flow_ops.group(
          self._num_good_steps.assign(0),
          self._current_loss_scale.assign(new_loss_scale))

    update_op = cond.cond(is_finite, update_if_finite_grads,
                          update_if_not_finite_grads)
    should_apply_gradients = is_finite
    return update_op, should_apply_gradients

  def __repr__(self):
    if context.executing_eagerly():
      return ('DynamicLossScale(current_loss_scale=%s, num_good_steps=%s, '
              'initial_loss_scale=%s, increment_period=%s, multiplier=%s)' %
              (self._current_loss_scale.numpy(), self._num_good_steps.numpy(),
               self.initial_loss_scale, self.increment_period, self.multiplier))
    else:
      return ('DynamicLossScale(initial_loss_scale=%s, increment_period=%s, '
              'multiplier=%s)' %
              (self.initial_loss_scale, self.increment_period, self.multiplier))

  def get_config(self):
    return {
        'initial_loss_scale': self.initial_loss_scale,
        'increment_period': self.increment_period,
        'multiplier': self.multiplier,
    }


def get(identifier):
  """Get a loss scale object."""
  if isinstance(identifier, (int, float)):
    return FixedLossScale(identifier)
  if identifier == 'dynamic':
    return DynamicLossScale()
  if isinstance(identifier, LossScale):
    return identifier
  elif identifier is None:
    return None
  else:
    raise ValueError('Could not interpret loss scale identifier: %s' %
                     identifier)
