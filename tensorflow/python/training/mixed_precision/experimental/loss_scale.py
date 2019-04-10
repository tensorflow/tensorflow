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
"""Contains LossScaler classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.ops import math_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.util.tf_export import tf_export


@six.add_metaclass(abc.ABCMeta)
@tf_export(v1=['train.mixed_precision.experimental.LossScale'])
class LossScale(trackable.Trackable):
  """Base class to compute the loss scale.

  Loss scaling is a process that:

  1) Applies a multiplier on the loss before computing gradients, and
  2) Applies the reciprocal of the multiplier on the gradients before they are
     applied on variables.

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow when float16 tensors are used. By multiplying the loss, each
  gradient will have the same multiplier applied.

  Instances of this class compute the loss scale. Method `get_loss_scale()`
  returns the current loss scale, while method `update()` updates the loss scale
  depending on the values of the gradients. Optimizers use instances of this
  class to scale loss and gradients.
  """

  def __init__(self):
    """Initializes the loss scale class.

    Note subclasses should create variables in build() instead of in the
    constructor. This is because callers might choose to place variables on
    a certain device by calling build() under a tf.device() scope.
    """
    self.built = False
    self._weights = {}

  def build(self):
    """Builds the weights of the loss scale class.

    If weights are needed, subclasses should build weights by calling
    `self.add_weight(...)`, then call the super's build to set self.built =
    True.
    """
    self.built = True

  def __call__(self):
    """Returns the current loss scale as a scalar `float32` tensor."""
    if not self.built:
      self.build()
    return self._get_loss_scale()

  @abc.abstractmethod
  def _get_loss_scale(self):
    """Returns the loss scale without calling build().

    Subclasses must implement this. Subclasses should not override the public
    `__call__` method, which calls this method.
    """
    pass

  def update(self, grads):
    """Updates the value of the loss scale.

    The loss scale tensor will be potentially updated, based on the value of
    `grads`. The tensor returned by `get_loss_scale` is only
    updated when this function is evaluated.

    In eager mode, this directly updates the loss scale, so that calling
    `get_loss_scale` will return the newly updated loss scale. In graph mode,
    this returns an op that, when evaluated, updates the loss scale.

    This function also returns a `should_apply_gradients` bool. If False,
    gradients should not be applied to the variables that step, as nonfinite
    gradients were found, and the loss scale controller can update the loss
    scale to reduce the chance of finding nonfinite gradients in the next step.
    Some loss scale controllers will always return True, as they cannot adjust
    the loss scale in response to nonfinite gradients.

    When a DistributionStrategy is used, this function may only be called in a
    cross-replica context.

    Args:
      grads: A list of unscaled gradients, each which is the gradient of the
        loss with respect to a weight. The gradients should have already been
        divided by the loss scale being before passed to this function.

    Returns:
      update_op: In eager mode, None. In graph mode, an op to update the loss
        scale.
      should_apply_gradients: Either a bool or a scalar boolean tensor. If
        False, the caller should skip applying `grads` to the variables this
        step.
    """
    if not self.built:
      self.build()
    return self._update(grads)

  @abc.abstractmethod
  def _update(self, grads):
    """Updates the value of the loss scale without calling build().

    Subclasses must implement this. Subclasses should not override the public
    `update_loss_scale` method, which calls this method.

    Args:
      grads: A list of unscaled gradients. See `update_loss_scale`.

    Returns:
      In eager mode, None. In graph mode, an op to update the loss scale.
    """
    pass


  def add_weight(self,
                 name,
                 shape=(),
                 dtype=None,
                 initializer='zeros'):
    """Adds a weight to this loss scale manager..

    This should be called by subclasses in `build()` to build the weights of the
    loss scale class.

    Args:
      name: Variable name.
      shape: Variable shape.
      dtype: The type of the variable.
      initializer: The initializer to use.

    Returns:
      A variable.
    """
    if isinstance(initializer, six.string_types) or callable(initializer):
      initializer = initializers.get(initializer)
    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        getter=base_layer_utils.make_variable,
        overwrite=True,
        initializer=initializer,
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
      graph_key = graph._graph_key # pylint: disable=protected-access

    key = (graph_key, name)
    if self._weights.get(key, None) is not None:
      raise RuntimeError('Duplicate variables detected. {}'.format(key))
    self._weights[key] = variable
    self._handle_deferred_dependencies(name=name, trackable=variable)
    return variable

  @property
  def _checkpoint_dependencies(self):
    """From Trackable. Gather graph-specific weights to save."""
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key # pylint: disable=protected-access
    weights = [trackable.TrackableReference(name=name, ref=v)
               for (g, name), v in sorted(
                   self._weights.items(), key=lambda i: i[0][1])
               if g == graph_key]
    return super(LossScale, self)._checkpoint_dependencies + weights

  def _lookup_dependency(self, name):
    """From Trackable. Find a weight in the current graph."""
    unconditional = super(LossScale, self)._lookup_dependency(name)
    if unconditional is not None:
      return unconditional
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key # pylint: disable=protected-access
    return self._weights.get((graph_key, name), None)

@tf_export(v1=['train.mixed_precision.experimental.FixedLossScale'])
class FixedLossScale(LossScale):
  """Loss scale class with a fixed value.

  The loss scale is not updated for the lifetime of the class.
  """

  def __init__(self, loss_scale_value):
    """Creates the fixed loss scale.

    Args:
      loss_scale: A Python float. Its ideal value varies depending on models to
        run. Choosing a too small loss_scale might affect model quality; a too
        big loss_scale might cause inf or nan. There is no single right
        loss_scale to apply. There is no harm choosing a relatively big number
        as long as no nan or inf is encountered in training.

    Raises:
      ValueError: If loss_scale is less than 1.
    """
    super(FixedLossScale, self).__init__()
    if not isinstance(loss_scale_value, six.integer_types + (float,)):
      raise ValueError('loss_scale must be a Python int or float.')
    if loss_scale_value < 1:
      raise ValueError('loss scale must be at least 1.')
    self._tensor_loss_scale = ops.convert_to_tensor(loss_scale_value,
                                                    dtype=dtypes.float32)

  def _get_loss_scale(self):
    return self._tensor_loss_scale

  def _update(self, grads):
    del grads
    return control_flow_ops.no_op(), True


def _is_all_finite(grads):
  """Returns a scalar boolean tensor indicating if all gradients are finite."""
  is_finite_per_grad = [math_ops.reduce_all(math_ops.is_finite(g))
                        for g in grads]
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
  return control_flow_ops.cond(
      math_ops.is_finite(value),
      lambda: _op_in_graph_mode(var.assign(value)),
      control_flow_ops.no_op)


@tf_export(v1=['train.mixed_precision.experimental.DynamicLossScale'])
class DynamicLossScale(LossScale):
  """Loss scale class that dynamically adjusts the loss scale.

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

  def __init__(self,
               initial_loss_scale=2 ** 15,
               increment_period=2000,
               multiplier=2.):
    """Constructor of exponential-update loss scale class.

    Args:
      initial_loss_scale: A Python float.  The loss scale to use at the
        beginning. It's better to start this at a very high number, because a
        loss scale that is too high gets lowered far more quickly than a loss
        scale that is to low gets raised. The default is 2 ** 15, which is
        approximately half the maximum float16 value.
      incr_every_n_steps: Increases loss scale every `incr_every_n_steps`
        consecutive steps that finite gradients are encountered. If a nonfinite
        gradient is encountered, the count is reset back to zero.
      loss_scale_multiplier: The multiplier to use when increasing or decreasing
        the loss scale.
    """
    super(DynamicLossScale, self).__init__()
    self._initial_loss_scale = float(initial_loss_scale)
    self._increment_period = int(increment_period)
    self._multiplier = float(multiplier)


  @property
  def initial_loss_scale(self):
    return self._initial_loss_scale

  @property
  def increment_period(self):
    return self._increment_period

  @property
  def multiplier(self):
    return self._multiplier

  def build(self):
    self._current_loss_scale = self.add_weight(
        name='loss_scale',
        dtype=dtypes.float32,
        initializer=self._initial_loss_scale)
    self._num_good_steps = self.add_weight(
        name='good_steps', dtype=dtypes.int64, initializer='zeros')
    self.built = True

  def _get_loss_scale(self):
    return self._current_loss_scale

  def _update(self, grads):
    """Updates loss scale based on if gradients are finite in current step."""
    if distribution_strategy_context.has_strategy():
      distribution = distribution_strategy_context.get_cross_replica_context()
      def get_is_finite(grads):
        is_finite = _is_all_finite(grads)
        # We cast to float, because we cannot reduce booleans with
        # DistributionStrategy.
        return math_ops.cast(is_finite, dtypes.float32)
      is_finite_float = distribution.extended.call_for_each_replica(
          get_is_finite, args=(grads,))
      reduced_is_finite_float = distribution.reduce(reduce_util.ReduceOp.SUM,
                                                    is_finite_float)
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

      return control_flow_ops.cond(
          self._num_good_steps + 1 >= self._increment_period,
          incr_loss_scale,
          lambda: _op_in_graph_mode(self._num_good_steps.assign_add(1)))

    def update_if_not_finite_grads():
      """Update assuming the gradients are nonfinite."""

      new_loss_scale = math_ops.maximum(
          self._current_loss_scale / self._multiplier, 1)
      return control_flow_ops.group(
          self._num_good_steps.assign(0),
          self._current_loss_scale.assign(new_loss_scale)
      )


    update_op = control_flow_ops.cond(is_finite, update_if_finite_grads,
                                      update_if_not_finite_grads)
    should_apply_gradients = is_finite
    return update_op, should_apply_gradients


def get(identifier):
  """Get a loss scale object."""
  if isinstance(identifier, six.integer_types + (float,)):
    return FixedLossScale(identifier)
  if identifier == 'dynamic':
    return DynamicLossScale()
  if isinstance(identifier, LossScale):
    return identifier
  elif identifier is None:
    return None
  else:
    raise ValueError('Could not interpret loss scale identifier: %s'
                     % identifier)
