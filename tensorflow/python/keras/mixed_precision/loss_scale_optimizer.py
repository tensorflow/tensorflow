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

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import base_delegate
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.util import nest
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

  __slots__ = ['value']

  def __init__(self, value):
    self.value = value


def _is_all_finite(grads):
  """Returns a scalar boolean tensor indicating if all gradients are finite."""
  def raw_values(g):
    return g.values if isinstance(g, indexed_slices.IndexedSlices) else g

  is_finite_per_grad = [
      math_ops.reduce_all(math_ops.is_finite(raw_values(g)))
      for g in grads if g is not None
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
  return control_flow_ops.cond(
      math_ops.is_finite(value), lambda: _op_in_graph_mode(var.assign(value)),
      control_flow_ops.no_op)


class _DynamicLossScaleState(trackable.Trackable):
  """The state of a dynamic loss scale."""

  def __init__(self,
               initial_loss_scale,
               growth_steps,
               multiplier):
    """Creates the dynamic loss scale."""
    super(_DynamicLossScaleState, self).__init__()
    self._initial_loss_scale = float(initial_loss_scale)
    self._growth_steps = int(growth_steps)
    self._multiplier = float(multiplier)

    self._weights = {}
    self._current_loss_scale = self._add_weight(
        name='current_loss_scale',
        dtype=dtypes.float32,
        initial_value=self._initial_loss_scale)
    # The number of consecutive steps with finite gradients since the last
    # nonfinite gradient or change in loss scale. The name is 'good_steps' for
    # backwards compatibility with older checkpoints.
    self._counter = self._add_weight(
        name='good_steps', dtype=dtypes.int64, initial_value=0)

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
    variable = variable_scope.variable(
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
    self._weights[key] = variable
    self._handle_deferred_dependencies(name=name, trackable=variable)
    backend.track_variable(variable)
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
        super(_DynamicLossScaleState,
              self)._trackable_children(save_type, **kwargs))
    return weights

  def _lookup_dependency(self, name):
    """From Trackable. Find a weight in the current graph."""
    unconditional = super(_DynamicLossScaleState, self)._lookup_dependency(name)
    if unconditional is not None:
      return unconditional
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key  # pylint: disable=protected-access
    return self._weights.get((name, graph_key), None)

  @property
  def initial_loss_scale(self):
    return self._initial_loss_scale

  @property
  def growth_steps(self):
    return self._growth_steps

  @property
  def multiplier(self):
    return self._multiplier

  @property
  def current_loss_scale(self):
    """Returns the current loss scale as a float32 `tf.Variable`."""
    return self._current_loss_scale

  @property
  def counter(self):
    """Returns the counter as a float32 `tf.Variable`."""
    return self._counter

  def __call__(self):
    """Returns the current loss scale as a scalar `float32` tensor."""
    return ops.convert_to_tensor_v2_with_dispatch(self._current_loss_scale)

  def update(self, grads):
    """Updates the value of the loss scale.

    Args:
      grads: A nested structure of unscaled gradients, each which is an
        all-reduced gradient of the loss with respect to a weight.

    Returns:
      update_op: In eager mode, None. In graph mode, an op to update the loss
        scale.
      should_apply_gradients: Either a bool or a scalar boolean tensor. If
        False, the caller should skip applying `grads` to the variables this
        step.
    """
    grads = nest.flatten(grads)
    if distribution_strategy_context.has_strategy(
    ) and distribution_strategy_context.in_cross_replica_context():
      distribution = distribution_strategy_context.get_strategy()
      is_finite_per_replica = distribution.extended.call_for_each_replica(
          _is_all_finite, args=(grads,))
      # Each replica computed the same `is_finite` value, since `grads` is
      # all-reduced across replicas. Arbitrarily take `is_finite` from the first
      # replica.
      is_finite = (
          distribution.experimental_local_results(is_finite_per_replica)[0])
    else:
      is_finite = _is_all_finite(grads)

    def update_if_finite_grads():
      """Update assuming the gradients are finite."""

      def incr_loss_scale():
        new_loss_scale = self.current_loss_scale * self.multiplier
        return control_flow_ops.group(
            _assign_if_finite(self.current_loss_scale, new_loss_scale),
            self.counter.assign(0))

      return control_flow_ops.cond(
          self.counter + 1 >= self.growth_steps,
          incr_loss_scale,
          lambda: _op_in_graph_mode(self.counter.assign_add(1)))

    def update_if_not_finite_grads():
      """Update assuming the gradients are nonfinite."""

      new_loss_scale = math_ops.maximum(
          self.current_loss_scale / self.multiplier, 1)
      return control_flow_ops.group(
          self.counter.assign(0),
          self.current_loss_scale.assign(new_loss_scale))

    update_op = control_flow_ops.cond(is_finite, update_if_finite_grads,
                                      update_if_not_finite_grads)
    should_apply_gradients = is_finite
    return update_op, should_apply_gradients


# See LossScaleOptimizer docstring for why this is so big
_DEFAULT_INITIAL_SCALE = 2 ** 15
_DEFAULT_GROWTH_STEPS = 2000


# pylint: disable=g-classes-have-attributes
@keras_export('keras.mixed_precision.LossScaleOptimizer')
class LossScaleOptimizer(base_delegate.DelegatingTrackableMixin,
                         optimizer_v2.OptimizerV2):
  """An optimizer that applies loss scaling to prevent numeric underflow.

  Loss scaling is a technique to prevent numeric underflow in intermediate
  gradients when float16 is used. To prevent underflow, the loss is multiplied
  (or "scaled") by a certain factor called the "loss scale", which causes
  intermediate gradients to be scaled by the loss scale as well. The final
  gradients are divided (or "unscaled") by the loss scale to bring them back to
  their original value.

  `LossScaleOptimizer` wraps another optimizer and applies loss scaling to it.
  By default, the loss scale is dynamically updated over time so you do not have
  to choose the loss scale. The `minimize` method automatically scales the loss,
  unscales the gradients, and updates the loss scale so all you have to do is
  wrap your optimizer with a `LossScaleOptimizer` if you use `minimize`. For
  example:

  >>> opt = tf.keras.optimizers.SGD(0.25)
  >>> opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
  >>> var = tf.Variable(1.)
  >>> loss_fn = lambda: var ** 2
  >>> # 'minimize' applies loss scaling and updates the loss sale.
  >>> opt.minimize(loss_fn, var_list=var)
  >>> var.numpy()
  0.5

  If a `tf.GradientTape` is used to compute gradients instead of `minimize`, you
  must scale the loss and gradients manually. This can be done with the
  `LossScaleOptimizer.get_scaled_loss` and
  `LossScaleOptimizer.get_unscaled_gradients` methods. For example:

  >>> with tf.GradientTape() as tape:
  ...   loss = loss_fn()
  ...   scaled_loss = opt.get_scaled_loss(loss)
  >>> scaled_grad = tape.gradient(scaled_loss, var)
  >>> (grad,) = opt.get_unscaled_gradients([scaled_grad])
  >>> opt.apply_gradients([(grad, var)])  # Loss scale is updated here
  >>> var.numpy()
  0.25

  Warning: If you forget to call `get_scaled_loss` or `get_unscaled_gradients`
  (or both) when using a `tf.GradientTape`, the model will likely converge to a
  worse quality. Please make sure you call each function exactly once.

  When mixed precision with float16 is used, there is typically no risk of
  underflow affecting model quality if loss scaling is properly used. See
  [the mixed precision guide](
  https://www.tensorflow.org/guide/keras/mixed_precision) for more information
  on how to use mixed precision.

  Args:
    inner_optimizer: The `tf.keras.optimizers.Optimizer` instance to wrap.
    dynamic: Bool indicating whether dynamic loss scaling is used. Defaults to
      True. If True, the loss scale will be dynamically updated over time using
      an algorithm that keeps the loss scale at approximately its optimal value.
      If False, a single fixed loss scale is used and `initial_scale` must be
      specified, which is used as the loss scale. Recommended to keep as True,
      as choosing a fixed loss scale can be tricky. Currently, there is a small
      performance overhead to dynamic loss scaling compared to fixed loss
      scaling.
    initial_scale: The initial loss scale. If `dynamic` is True, this defaults
      to `2 ** 15`. If `dynamic` is False, this must be specified and acts as
      the sole loss scale, as the loss scale does not change over time. When
      dynamic loss scaling is used, is better for this to be a very high number,
      because a loss scale that is too high gets lowered far more quickly than a
      loss scale that is too low gets raised.
    dynamic_growth_steps: With dynamic loss scaling, every
      `dynamic_growth_steps` steps with finite gradients, the loss scale is
      doubled. Defaults to 2000. If a nonfinite gradient is encountered, the
      count is reset back to zero, gradients are skipped that step, and the loss
      scale is halved. The count can be queried with
      `LossScaleOptimizer.dynamic_counter`. This argument can only be specified
      if `dynamic` is True.

  `LossScaleOptimizer` will occasionally skip applying gradients to the
  variables, in which case the trainable variables will not change that step.
  This is done because the dynamic loss scale will sometimes be raised too
  high, causing overflow in the gradients. Typically, the first 2 to 15 steps of
  the model are skipped as the initial loss scale is very high, but afterwards
  steps will only be skipped on average 0.05% of the time (the fraction of steps
  skipped is `1 / dynamic_growth_steps`).

  `LossScaleOptimizer` delegates all public `Optimizer` methods to the inner
  optimizer. Additionally, in methods `minimize` and `get_gradients`, it scales
  the loss and unscales the gradients. In methods `minimize` and
  `apply_gradients`, it additionally updates the loss scale and skips applying
  gradients if any gradient has a nonfinite value.

  ### Hyperparameters

  Hyperparameters can be accessed and set on the LossScaleOptimizer, which will
  be delegated to the wrapped optimizer.

  >>> opt = tf.keras.optimizers.Adam(beta_1=0.8, epsilon=1e-5)
  >>> opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
  >>> opt.beta_1  # Equivalent to `opt.inner_optimizer.beta_1`
  0.8
  >>> opt.beta_1 = 0.7  # Equivalent to `opt.inner_optimizer.beta_1 = 0.7`
  >>> opt.beta_1
  0.7
  >>> opt.inner_optimizer.beta_1
  0.7

  However, accessing or setting non-hyperparameters is not delegated to the
  LossScaleOptimizer. In an Adam optimizer, `beta_1` is a hyperparameter but
  `epsilon` is not, as the Adam optimizer only calls `Optimizer._set_hyper` on
  `beta_1`.

  >>> opt.inner_optimizer.epsilon
  1e-5
  >>> opt.epsilon
  Traceback (most recent call last):
  ...
  AttributeError: 'LossScaleOptimizer' object has no attribute 'epsilon'
  >>> opt.epsilon = 1e-4  # This does NOT set epsilon on `opt.inner_optimizer`
  >>> opt.inner_optimizer.epsilon
  >>> 1e-5

  In the above example, despite epsilon being set on the LossScaleOptimizer, the
  old epsilon value will still be used when training as epsilon was not set on
  the inner optimizer.
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self, inner_optimizer, dynamic=True, initial_scale=None,
               dynamic_growth_steps=None):
    if not isinstance(inner_optimizer, optimizer_v2.OptimizerV2):
      raise TypeError('"inner_optimizer" must be an instance of OptimizerV2, '
                      'but got: %s' % inner_optimizer)
    if not isinstance(dynamic, bool):
      # Catch errors if a user incorrectly passes a string or float to the
      # second argument argument, as this is commonly done for
      # LossScaleOptimizerV1.
      raise TypeError('"dynamic" argument to LossScaleOptimizer.__init__ must '
                      'be a bool, but got: %r' % (dynamic,))
    if isinstance(inner_optimizer, LossScaleOptimizer):
      raise TypeError('LossScaleOptimizer cannot wrap another '
                      'LossScaleOptimizer, but got: %s' % (inner_optimizer,))
    self._raise_if_strategy_unsupported()
    if getattr(inner_optimizer, '_is_wrapped_by_loss_scale_optimizer', False):
      # TODO(reedwm): Maybe support this. The difficulty is that LSO has the
      # same checkpoint format as the inner optimizer, so multiple LSOs wrapping
      # the same optimizer causes the checkpointing logic to become confused.
      raise ValueError('"inner_optimizer" is already wrapped by a '
                       'LossScaleOptimizer. An optimizer can only be wrapped '
                       'by a single LossScaleOptimizer')
    self._optimizer = inner_optimizer
    self._optimizer._is_wrapped_by_loss_scale_optimizer = True

    # We don't call super().__init__, since we do not want to call OptimizerV2's
    # constructor.
    base_delegate.DelegatingTrackableMixin.__init__(self, self._optimizer)

    if dynamic:
      if initial_scale is None:
        initial_scale = _DEFAULT_INITIAL_SCALE
      if dynamic_growth_steps is None:
        dynamic_growth_steps = _DEFAULT_GROWTH_STEPS
      self._loss_scale = _DynamicLossScaleState(
          initial_scale, dynamic_growth_steps, multiplier=2)
      self._track_trackable(self._loss_scale, 'loss_scale')
    else:
      if initial_scale is None:
        raise ValueError('"initial_scale" must be specified if "dynamic" is '
                         'False')
      self._loss_scale = float(initial_scale)
      if dynamic_growth_steps is not None:
        raise ValueError('"dynamic_growth_steps" must be None if "dynamic" '
                         'is False, but got: %s' % (dynamic_growth_steps,))

    # To support restoring TensorFlow 2.2 checkpoints.
    self._track_trackable(FakeOptimizerForRestoration(self._optimizer),
                          'base_optimizer')

  @property
  def dynamic(self):
    """Bool indicating whether dynamic loss scaling is used."""
    return isinstance(self._loss_scale, _DynamicLossScaleState)

  @property
  def loss_scale(self):
    """The current loss scale as a float32 scalar tensor."""
    if isinstance(self._loss_scale, _DynamicLossScaleState):
      return ops.convert_to_tensor_v2_with_dispatch(
          self._loss_scale.current_loss_scale)
    else:
      return ops.convert_to_tensor_v2_with_dispatch(self._loss_scale)

  @property
  def dynamic_counter(self):
    """The number of steps since the loss scale was last increased or decreased.

    This is None if `LossScaleOptimizer.dynamic` is False.

    The counter is incremented every step. Once it reaches
    `LossScaleOptimizer.dynamic_growth_steps`, the loss scale will be doubled
    and the counter will be reset back to zero. If nonfinite gradients are
    encountered, the loss scale will be halved and the counter will be reset
    back to zero.
    """
    if isinstance(self._loss_scale, _DynamicLossScaleState):
      return self._loss_scale.counter
    else:
      return None

  @property
  def initial_scale(self):
    """The initial loss scale.

    If `LossScaleOptimizer.dynamic` is False, this is the same number as
    `LossScaleOptimizer.loss_scale`, as the loss scale never changes.
    """
    if isinstance(self._loss_scale, _DynamicLossScaleState):
      return self._loss_scale.initial_loss_scale
    else:
      return self._loss_scale

  @property
  def dynamic_growth_steps(self):
    """The number of steps it takes to increase the loss scale.

    This is None if `LossScaleOptimizer.dynamic` is False.

    Every `dynamic_growth_steps` consecutive steps with finite gradients, the
    loss scale is increased.
    """
    if isinstance(self._loss_scale, _DynamicLossScaleState):
      return self._loss_scale.growth_steps
    else:
      return None

  @property
  def inner_optimizer(self):
    """The optimizer that this LossScaleOptimizer is wrapping."""
    return self._optimizer

  def get_scaled_loss(self, loss):
    """Scales the loss by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.LossScaleOptimizer` doc for
    an example.

    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
        a tensor or a callable returning a tensor.

    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale`.
    """
    if callable(loss):
      def new_loss():
        loss_val = loss()
        return loss_val * math_ops.cast(self.loss_scale, loss_val.dtype)
      return new_loss
    else:
      return loss * math_ops.cast(self.loss_scale, loss.dtype)

  def get_unscaled_gradients(self, grads):
    """Unscales the gradients by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to unscale the gradients
    after computing them with `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_scaled_loss` should also be called. See
    the `tf.keras.mixed_precision.LossScaleOptimizer` doc for an
    example.

    Args:
      grads: A list of tensors, each which will be divided by the loss scale.
        Can have None values, which are ignored.

    Returns:
      A new list the same size as `grads`, where every non-None value in `grads`
      is divided by `LossScaleOptimizer.loss_scale`.
    """
    loss_scale_reciprocal = 1. / self.loss_scale
    return [
        _multiply_gradient(g, loss_scale_reciprocal) if g is not None else None
        for g in grads
    ]

  def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
    tape = backprop.GradientTape() if tape is None else tape
    with tape:
      loss = self.get_scaled_loss(loss)
    grads_and_vars = self._optimizer._compute_gradients(  # pylint: disable=protected-access
        loss,
        var_list,
        grad_loss,
        tape=tape)
    grads = [g for g, _ in grads_and_vars]
    weights = [v for _, v in grads_and_vars]
    unscaled_grads = self.get_unscaled_gradients(grads)
    return list(zip(unscaled_grads, weights))

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

    grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
    if experimental_aggregate_gradients:
      # We must aggregate the gradients here instead of in
      # self.optimizer.apply_gradients, so that any NaN or Inf gradients are
      # propogated to each replica. If any replica has a NaN or Inf gradient,
      # they must all have a NaN or Inf gradient so that they all skip the step.
      # pylint: disable=protected-access
      grads_and_vars = self._optimizer._transform_unaggregated_gradients(
          grads_and_vars)
      grads_and_vars = self._optimizer._aggregate_gradients(grads_and_vars)
      # pylint: enable=protected-access

    grads_and_vars = tuple(grads_and_vars)
    grads = [g for g, _ in grads_and_vars]
    # We do not want DistributionStrategy to unwrap any MirroredVariables in
    # grads_and_vars, because even in a replica context, the wrapped
    # optimizer expects mirrored variables. So we wrap the variables with an
    # _UnwrapPreventer, preventing DistributionStrategy from unwrapping the
    # MirroredVariables.
    wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])

    def do_not_apply_fn():
      # Normally self._optimizer.iterations is incremented in
      # self._optimizer.apply_gradients(). Since that is not called in this
      # branch, we increment it here instead.
      return self._optimizer.iterations.assign_add(1, read_value=False)

    def _if_should_apply_grads(grads):
      if isinstance(self._loss_scale, _DynamicLossScaleState):
        return self._loss_scale.update(grads)
      else:
        return (control_flow_ops.no_op(), True)

    if optimizer_utils.strategy_supports_no_merge_call():
      loss_scale_update_op, should_apply_grads = _if_should_apply_grads(grads)
      def apply_fn():
        return self._apply_gradients(grads, wrapped_vars, name)

      maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn,
                                             do_not_apply_fn)
      return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)

    else:

      def _apply_gradients_cross_replica(distribution, grads, wrapped_vars,
                                         name):
        loss_scale_update_op, should_apply_grads = _if_should_apply_grads(grads)

        def apply_fn():
          return distribution.extended.call_for_each_replica(
              self._apply_gradients,
              args=(grads, wrapped_vars, name))

        # Note: We must call this cond() in a cross-replica context.
        # DistributionStrategy does not support having a cond in a replica
        # context with a branch that calls `merge_call`, and
        # self._optimizer.apply_gradients calls `merge_call`.
        maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn,
                                               do_not_apply_fn)
        return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)
      return distribution_strategy_context.get_replica_context().merge_call(
          _apply_gradients_cross_replica,
          args=(grads, wrapped_vars, name))

  def _apply_gradients(self, grads, wrapped_vars, name):
    # Pass experimental_aggregate_gradients=False since LossScaleOptimizer
    # already aggregated the gradients.
    # TODO(reedwm): This will raise a fairly cryptic error message if
    # self._optimizer.apply_gradients does not take
    # experimental_aggregate_gradients.
    return self._optimizer.apply_gradients(
        list(zip(grads, wrapped_vars.value)), name,
        experimental_aggregate_gradients=False)

  def get_config(self):
    serialized_optimizer = optimizers.serialize(self._optimizer)
    return {
        'inner_optimizer': serialized_optimizer,
        'dynamic': self.dynamic,
        'initial_scale': self.initial_scale,
        'dynamic_growth_steps': self.dynamic_growth_steps,
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()  # Make a copy, since we mutate config
    if 'loss_scale' in config:
      # If loss_scale is in config, we assume we are deserializing a
      # LossScaleOptimizer from TF 2.3 or below. We convert the config so it
      # can be deserialized in the current LossScaleOptimizer.
      loss_scale = keras_loss_scale_module.deserialize(
          config.pop('loss_scale'))
      if isinstance(loss_scale, loss_scale_module.FixedLossScale):
        config['dynamic'] = False
        config['initial_scale'] = loss_scale._loss_scale_value  # pylint: disable=protected-access
      elif isinstance(loss_scale, loss_scale_module.DynamicLossScale):
        config['dynamic'] = True
        config['initial_scale'] = loss_scale.initial_loss_scale
        config['dynamic_growth_steps'] = loss_scale.increment_period
        if loss_scale.multiplier != 2:
          raise ValueError('Cannot deserialize LossScaleOptimizer with a '
                           'DynamicLossScale whose multiplier is not 2. Got '
                           'DynamicLossScale: %s' % (loss_scale,))
      else:
        raise ValueError(
            'Serialized LossScaleOptimizers with a LossScale that is neither a '
            'FixedLossScale nor a DynamicLossScale can no longer be '
            'deserialized')
      config['inner_optimizer'] = config.pop('optimizer')
    config['inner_optimizer'] = optimizers.deserialize(
        config['inner_optimizer'], custom_objects=custom_objects)
    return cls(**config)

  def _raise_if_strategy_unsupported(self):
    if not strategy_supports_loss_scaling():
      strategy = distribution_strategy_context.get_strategy()
      if isinstance(strategy,
                    (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1,
                     tpu_strategy.TPUStrategyV2)):
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

  @property
  def clipnorm(self):
    return self._optimizer.clipnorm

  @clipnorm.setter
  def clipnorm(self, val):
    self._optimizer.clipnorm = val

  @property
  def global_clipnorm(self):
    return self._optimizer.global_clipnorm

  @global_clipnorm.setter
  def global_clipnorm(self, val):
    self._optimizer.global_clipnorm = val

  @property
  def clipvalue(self):
    return self._optimizer.clipvalue

  @clipvalue.setter
  def clipvalue(self, val):
    self._optimizer.clipvalue = val

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

  def __getattribute__(self, name):
    try:
      return object.__getattribute__(self, name)
    except AttributeError as e:
      if name == '_optimizer' or name == '_hyper':
        # Avoid infinite recursion
        raise e

      # Delegate hyperparameter accesses to inner optimizer.
      if name == 'lr':
        name = 'learning_rate'
      if name in self._optimizer._hyper:
        return self._optimizer._get_hyper(name)
      raise e

  def __dir__(self):
    result = set(super(LossScaleOptimizer, self).__dir__())
    if '_optimizer' in result:
      result |= self._optimizer._hyper.keys()
      if 'learning_rate' in self._optimizer._hyper.keys():
        result.add('lr')
    return list(result)

  def __setattr__(self, name, value):
    if name == 'lr':
      name = 'learning_rate'
    # Delegate setting hyperparameter to inner optimizer if the attribute does
    # not exist on the LossScaleOptimizer
    try:
      # We cannot check for the 'iterations' attribute as it cannot be set after
      # it is accessed.
      if name != 'iterations':
        object.__getattribute__(self, name)
      has_attribute = True
    except AttributeError:
      has_attribute = False
    if (name != '_optimizer' and name in self._optimizer._hyper
        and not has_attribute):
      self._optimizer._set_hyper(name, value)
    else:
      super(LossScaleOptimizer, self).__setattr__(name, value)

  # Explicitly delegate learning_rate. Normally hyperparameters are delegated in
  # __getattribute__, but if a hyperparameter is not in self._optimizer._hyper
  # (e.g. because self._optimizer itself wraps another optimizer), then it won't
  # be delegated. Since learning_rate is a very commonly accessed
  # hyperparameter, we delegate it here.
  @property
  def learning_rate(self):
    return self._optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, value):
    self._optimizer.learning_rate = value

  @property
  def lr(self):
    return self._optimizer.learning_rate

  @lr.setter
  def lr(self, value):
    self._optimizer.lr = value

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

  # TODO(reedwm): Maybe throw an error if mixed precision is used without this
  # optimizer being used.


@keras_export('keras.mixed_precision.experimental.LossScaleOptimizer')
class LossScaleOptimizerV1(LossScaleOptimizer):
  """An deprecated optimizer that applies loss scaling.

  Warning: This class is deprecated and will be removed in a future version of
  TensorFlow. Please use the non-experimental class
  `tf.keras.mixed_precision.LossScaleOptimizer` instead.

  This class is identical to the non-experimental
  `keras.mixed_precision.LossScaleOptimizer` except its constructor takes
  different arguments. For this class (the experimental version), the
  constructor takes a `loss_scale` argument.  For the non-experimental class,
  the constructor encodes the loss scaling information in multiple arguments.
  Note that unlike this class, the non-experimental class does not accept a
  `tf.compat.v1.mixed_precision.LossScale`, which is deprecated.

  If you currently use this class, you should switch to the non-experimental
  `tf.keras.mixed_precision.LossScaleOptimizer` instead. We show several
  examples of converting the use of the experimental class to the equivalent
  non-experimental class.

  >>> # In all of the examples below, `opt1` and `opt2` are identical
  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale='dynamic')
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD())
  >>> assert opt1.get_config() == opt2.get_config()

  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale=123)
  >>> # dynamic=False indicates to use fixed loss scaling. initial_scale=123
  >>> # refers to the initial loss scale, which is the single fixed loss scale
  >>> # when dynamic=False.
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), dynamic=False, initial_scale=123)
  >>> assert opt1.get_config() == opt2.get_config()

  >>> loss_scale = tf.compat.v1.mixed_precision.experimental.DynamicLossScale(
  ...     initial_loss_scale=2048, increment_period=500)
  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale=loss_scale)
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), initial_scale=2048,
  ...     dynamic_growth_steps=500)
  >>> assert opt1.get_config() == opt2.get_config()

  Make sure to also switch from this class to the non-experimental class in
  isinstance checks, if you have any. If you do not do this, your model may run
  into hard-to-debug issues, as the experimental `LossScaleOptimizer` subclasses
  the non-experimental `LossScaleOptimizer`, but not vice versa. It is safe to
  switch isinstance checks to the non-experimental `LossScaleOptimizer` even
  before using the non-experimental `LossScaleOptimizer`.

  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale='dynamic')
  >>> # The experimental class subclasses the non-experimental class
  >>> isinstance(opt1, tf.keras.mixed_precision.LossScaleOptimizer)
  True
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD())
  >>> # The non-experimental class does NOT subclass the experimental class.
  >>> isinstance(opt2, tf.keras.mixed_precision.experimental.LossScaleOptimizer)
  False

  Args:
    optimizer: The Optimizer instance to wrap.
    loss_scale: The loss scale to scale the loss and gradients. This can
      either be an int/float to use a fixed loss scale, the string "dynamic"
      to use dynamic loss scaling, or an instance of a LossScale. The string
      "dynamic" equivalent to passing `DynamicLossScale()`, and passing an
      int/float is equivalent to passing a FixedLossScale with the given loss
      scale. If a DynamicLossScale is passed, DynamicLossScale.multiplier must
      be 2 (the default).
  """

  def __init__(self, optimizer, loss_scale):
    warn_msg_prefix = (
        'tf.keras.mixed_precision.experimental.LossScaleOptimizer is '
        'deprecated. Please use tf.keras.mixed_precision.LossScaleOptimizer '
        'instead. ')

    if isinstance(loss_scale, dict):
      loss_scale = keras_loss_scale_module.deserialize(loss_scale)

    if isinstance(loss_scale, (int, float)):
      tf_logging.warning(
          warn_msg_prefix + 'For example:\n'
          '  opt = tf.keras.mixed_precision.LossScaleOptimizer('
          'opt, dynamic=False, initial_scale={})'.format(loss_scale))
      super(LossScaleOptimizerV1, self).__init__(optimizer, dynamic=False,
                                                 initial_scale=loss_scale)
    elif isinstance(loss_scale, loss_scale_module.FixedLossScale):
      ls_val = loss_scale._loss_scale_value  # pylint: disable=protected-access
      tf_logging.warning(
          warn_msg_prefix + 'For example:\n'
          '  opt = tf.keras.mixed_precision.LossScaleOptimizer('
          'opt, dynamic=False, initial_scale={})'.format(ls_val))
      super(LossScaleOptimizerV1, self).__init__(optimizer, dynamic=False,
                                                 initial_scale=ls_val)
    elif loss_scale == 'dynamic':
      tf_logging.warning(
          warn_msg_prefix + 'For example:\n'
          '  opt = tf.keras.mixed_precision.LossScaleOptimizer('
          'opt)')
      super(LossScaleOptimizerV1, self).__init__(optimizer)
    elif isinstance(loss_scale, loss_scale_module.DynamicLossScale):
      kwargs = {}
      extra_arguments = ''
      if loss_scale.initial_loss_scale != _DEFAULT_INITIAL_SCALE:
        kwargs['initial_scale'] = loss_scale.initial_loss_scale
        extra_arguments += (', initial_scale=%s' %
                            loss_scale.initial_loss_scale)
      if loss_scale.increment_period != _DEFAULT_GROWTH_STEPS:
        kwargs['dynamic_growth_steps'] = loss_scale.increment_period
        extra_arguments += (', dynamic_growth_steps=%s' %
                            loss_scale.increment_period)
      if loss_scale.multiplier != 2:
        raise ValueError('When passing a DynamicLossScale to "loss_scale", '
                         'DynamicLossScale.multiplier must be 2. Got: %s'
                         % (loss_scale,))
      tf_logging.warning(
          warn_msg_prefix +
          'Note that the non-experimental LossScaleOptimizer does not take a '
          'DynamicLossScale but instead takes the dynamic configuration '
          'directly in the constructor. For example:\n'
          '  opt = tf.keras.mixed_precision.LossScaleOptimizer('
          'opt{})\n'.format(extra_arguments))
      super(LossScaleOptimizerV1, self).__init__(optimizer, **kwargs)
    elif isinstance(loss_scale, loss_scale_module.LossScale):
      raise TypeError('Passing a LossScale that is not a FixedLossScale or a '
                      'DynamicLossScale is no longer supported. Got: {}'
                      .format(loss_scale))
    else:
      raise ValueError('Invalid value passed to loss_scale. loss_scale '
                       'must be the string "dynamic" (recommended), an int, '
                       'a float, a FixedLossScale, or a DynamicLossScale. Got '
                       'value: {}'.format(loss_scale))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()  # Make a copy, since we mutate config

    # If loss_scale is in config, we assume we are deserializing a
    # LossScaleOptimizer from TF 2.3 or below. Otherwise, we assume we are
    # deserializing a LossScaleOptimizer from TF 2.4 or above.
    if 'loss_scale' in config:
      config['loss_scale'] = keras_loss_scale_module.deserialize(
          config['loss_scale'])
      if (isinstance(config['loss_scale'], loss_scale_module.DynamicLossScale)
          and config['loss_scale'].multiplier != 2):
        raise ValueError('Cannot deserialize LossScaleOptimizer with a '
                         'DynamicLossScale whose multiplier is not 2. Got '
                         'DynamicLossScale: %s' % (config['loss_scale'],))
      config['optimizer'] = optimizers.deserialize(
          config['optimizer'], custom_objects=custom_objects)
      return cls(**config)

    # We convert the config, as generated by LossScaleOptimizer.get_config, to a
    # version that can be passed to LossScaleOptimizerV1.__init__
    if config['dynamic']:
      config['loss_scale'] = loss_scale_module.DynamicLossScale(
          config['initial_scale'], config['dynamic_growth_steps'], multiplier=2)
    else:
      config['loss_scale'] = loss_scale_module.FixedLossScale(
          config['initial_scale'])

    del config['dynamic']
    del config['initial_scale']
    del config['dynamic_growth_steps']
    config['optimizer'] = optimizers.deserialize(
        config.pop('inner_optimizer'), custom_objects=custom_objects)
    return cls(**config)


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


mixed_precision.register_loss_scale_wrapper(optimizer_v2.OptimizerV2,
                                            LossScaleOptimizerV1)


def _multiply_gradient(gradient, scale):
  """Multiply a (possibly sparse) gradient by the given scale factor."""
  scale = math_ops.cast(scale, gradient.dtype)
  if isinstance(gradient, indexed_slices.IndexedSlices):
    return indexed_slices.IndexedSlices(
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
