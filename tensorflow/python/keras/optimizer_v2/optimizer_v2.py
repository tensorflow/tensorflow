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
"""Version 2 of class Optimizer."""
# pylint: disable=g-bad-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib
import functools

import six

from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export


def _deduplicate_indexed_slices(values, indices):
  """Sums `values` associated with any non-unique `indices`.

  Args:
    values: A `Tensor` with rank >= 1.
    indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).

  Returns:
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
  """
  unique_indices, new_index_positions = array_ops.unique(indices)
  summed_values = math_ops.unsorted_segment_sum(
      values, new_index_positions,
      array_ops.shape(unique_indices)[0])
  return (summed_values, unique_indices)


@six.add_metaclass(abc.ABCMeta)
@keras_export("keras.optimizers.Optimizer")
class OptimizerV2(trackable.Trackable):
  """Base class for Keras optimizers.

  You should not use this class directly, but instead instantiate one of its
  subclasses such as `tf.keras.optimizers.SGD`, `tf.keras.optimizers.Adam`, etc.

  ### Usage

  ```python
  # Create an optimizer with the desired parameters.
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  # `loss` is a callable that takes no argument and returns the value
  # to minimize.
  loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
  # In graph mode, returns op that minimizes the loss by updating the listed
  # variables.
  opt_op = opt.minimize(loss, var_list=[var1, var2])
  opt_op.run()
  # In eager mode, simply call minimize to update the list of variables.
  opt.minimize(loss, var_list=[var1, var2])
  ```

  ### Usage in custom training loops

  In Keras models, sometimes variables are created when the model is first
  called, instead of construction time. Examples include 1) sequential models
  without input shape pre-defined, or 2) subclassed models. Pass var_list as
  callable in these cases.

  Example:

  ```python
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
  model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
  loss_fn = lambda: tf.keras.losses.mse(model(input), output)
  var_list_fn = lambda: model.trainable_weights
  for input, output in data:
    opt.minimize(loss_fn, var_list_fn)
  ```

  ### Processing gradients before applying them

  Calling `minimize()` takes care of both computing the gradients and
  applying them to the variables.  If you want to process the gradients
  before applying them you can instead use the optimizer in three steps:

  1.  Compute the gradients with `tf.GradientTape`.
  2.  Process the gradients as you wish.
  3.  Apply the processed gradients with `apply_gradients()`.

  Example:

  ```python
  # Create an optimizer.
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)

  # Compute the gradients for a list of variables.
  with tf.GradientTape() as tape:
    loss = <call_loss_function>
  vars = <list_of_variables>
  grads = tape.gradient(loss, vars)

  # Process the gradients, for example cap them, etc.
  # capped_grads = [MyCapper(g) for g in grads]
  processed_grads = [process_gradient(g) for g in grads]

  # Ask the optimizer to apply the processed gradients.
  opt.apply_gradients(zip(processed_grads, var_list))
  ```

  ### Use with `tf.distribute.Strategy`

  This optimizer class is `tf.distribute.Strategy` aware, which means it
  automatically sums gradients across all replicas. To average gradients,
  you divide your loss by the global batch size, which is done
  automatically if you use `tf.keras` built-in training or evaluation loops.
  See the `reduction` argument of your loss which should be set to
  `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` for averaging or
  `tf.keras.losses.Reduction.SUM` for not.

  To aggregate gradients yourself, call `apply_gradients` with
  `experimental_aggregate_gradients` set to False. This is useful if you need to
  process aggregated gradients.

  If you are not using these and you want to average gradients, you should use
  `tf.math.reduce_sum` to add up your per-example losses and then divide by the
  global batch size. Note that when using `tf.distribute.Strategy`, the first
  component of a tensor's shape is the *replica-local* batch size, which is off
  by a factor equal to the number of replicas being used to compute a single
  step. As a result, using `tf.math.reduce_mean` will give the wrong answer,
  resulting in gradients that can be many times too big.

  ### Variable Constraints

  All Keras optimizers respect variable constraints. If constraint function is
  passed to any variable, the constraint will be applied to the variable after
  the gradient has been applied to the variable.
  Important: If gradient is sparse tensor, variable constraint is not supported.

  ### Thread Compatibility

  The entire optimizer is currently thread compatible, not thread-safe. The user
  needs to perform synchronization if necessary.

  ### Slots

  Many optimizer subclasses, such as `Adam` and `Adagrad` allocate and manage
  additional variables associated with the variables to train.  These are called
  <i>Slots</i>.  Slots have names and you can ask the optimizer for the names of
  the slots that it uses.  Once you have a slot name you can ask the optimizer
  for the variable it created to hold the slot value.

  This can be useful if you want to log debug a training algorithm, report stats
  about the slots, etc.

  ### Hyperparameters

  These are arguments passed to the optimizer subclass constructor
  (the `__init__` method), and then passed to `self._set_hyper()`.
  They can be either regular Python values (like 1.0), tensors, or
  callables. If they are callable, the callable will be called during
  `apply_gradients()` to get the value for the hyper parameter.

  Hyperparameters can be overwritten through user code:

  Example:

  ```python
  # Create an optimizer with the desired parameters.
  opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  # `loss` is a callable that takes no argument and returns the value
  # to minimize.
  loss = lambda: 3 * var1 + 2 * var2
  # In eager mode, simply call minimize to update the list of variables.
  opt.minimize(loss, var_list=[var1, var2])
  # update learning rate
  opt.learning_rate = 0.05
  opt.minimize(loss, var_list=[var1, var2])
  ```

  ### Callable learning rate

  Optimizer accepts a callable learning rate in two ways. The first way is
  through built-in or customized
  `tf.keras.optimizers.schedules.LearningRateSchedule`. The schedule will be
  called on each iteration with `schedule(iteration)`, a `tf.Variable`
  owned by the optimizer.

  Example:

  >>> var = tf.Variable(np.random.random(size=(1,)))
  >>> learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
  ... initial_learning_rate=.01, decay_steps=20, decay_rate=.1)
  >>> opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  >>> loss = lambda: 3 * var
  >>> opt.minimize(loss, var_list=[var])
  <tf.Variable...

  The second way is through a callable function that
  does not accept any arguments.

  Example:

  >>> var = tf.Variable(np.random.random(size=(1,)))
  >>> def lr_callable():
  ...   return .1
  >>> opt = tf.keras.optimizers.SGD(learning_rate=lr_callable)
  >>> loss = lambda: 3 * var
  >>> opt.minimize(loss, var_list=[var])
  <tf.Variable...

  ### Creating a custom optimizer

  If you intend to create your own optimization algorithm, simply inherit from
  this class and override the following methods:

    - `_resource_apply_dense` (update variable given gradient tensor is dense)
    - `_resource_apply_sparse` (update variable given gradient tensor is sparse)
    - `_create_slots`
      (if your optimizer algorithm requires additional variables)
    - `get_config`
      (serialization of the optimizer, include all hyper parameters)
  """

  # Subclasses should set this to True unless they override `apply_gradients`
  # with a version that does not have the `experimental_aggregate_gradients`
  # argument.  Older versions of Keras did not have this argument so custom
  # optimizers may have overridden `apply_gradients` without the
  # `experimental_aggregate_gradients` argument. Keras only passes
  # `experimental_aggregate_gradients` if this attribute is True.
  # Note: This attribute will likely be removed in an upcoming release.
  _HAS_AGGREGATE_GRAD = False

  def __init__(self, name, **kwargs):
    """Create a new Optimizer.

    This must be called by the constructors of subclasses.
    Note that Optimizer instances should not bind to a single graph,
    and so shouldn't keep Tensors as member variables. Generally
    you should be able to use the _set_hyper()/state.get_hyper()
    facility instead.

    This class in stateful and thread-compatible.

    Args:
      name: A non-empty string.  The name to use for accumulators created
        for the optimizer.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    Raises:
      ValueError: If name is malformed.
    """
    allowed_kwargs = {"clipnorm", "clipvalue", "lr", "decay"}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError("Unexpected keyword argument "
                        "passed to optimizer: " + str(k))
      # checks that all keyword arguments are non-negative.
      if kwargs[k] is not None and kwargs[k] < 0:
        raise ValueError("Expected {} >= 0, received: {}".format(k, kwargs[k]))

    self._use_locking = True
    self._init_set_name(name)
    self._hyper = {}
    # dict: {variable name : {slot name : variable}}
    self._slots = {}
    self._slot_names = []
    self._weights = []
    self._iterations = None

    # For implementing Trackable. Stores information about how to restore
    # slot variables which have not yet been created
    # (trackable._CheckpointPosition objects).
    #  {slot_name :
    #      {_var_key(variable_to_train): [checkpoint_position, ... ], ... },
    #   ... }
    self._deferred_slot_restorations = {}

    decay = kwargs.pop("decay", 0.0)
    if decay < 0.:
      raise ValueError("decay cannot be less than 0: {}".format(decay))
    self._initial_decay = decay

    # Set the gradient clipping properties
    self.clipnorm = kwargs.pop("clipnorm", None)
    self.clipvalue = kwargs.pop("clipvalue", None)
    if ((self.clipnorm is not None or self.clipvalue is not None)
        and distribute_ctx.has_strategy()):
      raise ValueError("Gradient clipping in the optimizer "
                       "(by setting clipnorm or clipvalue) is currently "
                       "unsupported when using a distribution strategy.")

    self._hypers_created = False

    # Store the distribution strategy object if the optimizer is created inside
    # strategy scope, so it could be used to create variables later.
    if distribute_ctx.has_strategy():
      self._distribution_strategy = distribute_ctx.get_strategy()
    else:
      self._distribution_strategy = None

  def minimize(self, loss, var_list, grad_loss=None, name=None):
    """Minimize `loss` by updating `var_list`.

    This method simply computes gradient using `tf.GradientTape` and calls
    `apply_gradients()`. If you want to process the gradient before applying
    then call `tf.GradientTape` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A callable taking no arguments which returns the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` since the variables are created at the first time `loss` is
        called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      name: Optional name for the returned operation.

    Returns:
      An `Operation` that updates the variables in `var_list`. The `iterations`
      will be automatically increased by 1.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.

    """
    grads_and_vars = self._compute_gradients(
        loss, var_list=var_list, grad_loss=grad_loss)

    return self.apply_gradients(grads_and_vars, name=name)

  def _clip_gradients(self, grads):
    """Clip gradients according to the clipnorm and clipvalue attributes."""
    if self.clipnorm is not None:
      if distribute_ctx.has_strategy():
        raise ValueError("Gradient clipping in the optimizer "
                         "(by setting clipnorm or clipvalue) is currently "
                         "unsupported when using a distribution strategy.")
      grads = [None if g is None else clip_ops.clip_by_norm(g, self.clipnorm)
               for g in grads]
    if self.clipvalue is not None:
      if distribute_ctx.has_strategy():
        raise ValueError("Gradient clipping in the optimizer "
                         "(by setting clipnorm or clipvalue) is currently "
                         "unsupported when using a distribution strategy.")
      v = self.clipvalue
      grads = [
          None if g is None else clip_ops.clip_by_value(g, -v, v) for g in grads
      ]
    return grads

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A callable taking no arguments which returns the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` and the variables are created at the first time when `loss`
        is called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid, or var_list is None.
    """
    # TODO(josh11b): Test that we handle weight decay in a reasonable way.
    with backprop.GradientTape() as tape:
      if not callable(var_list):
        tape.watch(var_list)
      loss_value = loss()
    if callable(var_list):
      var_list = var_list()
    var_list = nest.flatten(var_list)
    with backend.name_scope(self._name + "/gradients"):
      grads = tape.gradient(loss_value, var_list, grad_loss)
      grads = self._clip_gradients(grads)

    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes([
        v for g, v in grads_and_vars
        if g is not None and v.dtype != dtypes.resource
    ])

    return grads_and_vars

  def get_gradients(self, loss, params):
    """Returns gradients of `loss` with respect to `params`.

    Arguments:
      loss: Loss tensor.
      params: List of variables.

    Returns:
      List of gradient tensors.

    Raises:
      ValueError: In case any gradient cannot be computed (e.g. if gradient
        function not implemented).
    """
    params = nest.flatten(params)
    with backend.get_graph().as_default(), backend.name_scope(self._name +
                                                              "/gradients"):
      grads = gradients.gradients(loss, params)
      for grad, param in zip(grads, params):
        if grad is None:
          raise ValueError("Variable {} has `None` for gradient. "
                           "Please make sure that all of your ops have a "
                           "gradient defined (i.e. are differentiable). "
                           "Common ops without gradient: "
                           "K.argmax, K.round, K.eval.".format(param))
      grads = self._clip_gradients(grads)
    return grads

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    The method sums gradients from all replicas in the presence of
    `tf.distribute.Strategy` by default. You can aggregate gradients yourself by
    passing `experimental_aggregate_gradients=False`.

    Example:

    ```python
    grads = tape.gradient(loss, vars)
    grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    # Processing aggregated gradients.
    optimizer.apply_gradients(zip(grads, vars),
        experimental_aggregate_gradients=False)

    ```

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      name: Optional name for the returned operation. Default to the name passed
        to the `Optimizer` constructor.
      experimental_aggregate_gradients: Whether to sum gradients from different
        replicas in the presense of `tf.distribute.Strategy`. If False, it's
        user responsibility to aggregate the gradients. Default to True.

    Returns:
      An `Operation` that applies the specified gradients. The `iterations`
      will be automatically increased by 1.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    """
    grads_and_vars = _filter_grads(grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]

    with backend.name_scope(self._name):
      # Create iteration if necessary.
      with ops.init_scope():
        self._create_all_weights(var_list)

      if not grads_and_vars:
        # Distribution strategy does not support reducing an empty list of
        # gradients
        return control_flow_ops.no_op()

      if distribute_ctx.in_cross_replica_context():
        raise RuntimeError(
            "`apply_gradients() cannot be called in cross-replica context. "
            "Use `tf.distribute.Strategy.run` to enter replica "
            "context.")

      strategy = distribute_ctx.get_strategy()
      if (not experimental_aggregate_gradients and strategy and isinstance(
          strategy.extended,
          parameter_server_strategy.ParameterServerStrategyExtended)):
        raise NotImplementedError(
            "`experimental_aggregate_gradients=False is not supported for "
            "ParameterServerStrategy and CentralStorageStrategy")

      apply_state = self._prepare(var_list)
      if experimental_aggregate_gradients:
        reduced_grads = self._aggregate_gradients(grads_and_vars)
        var_list = [v for _, v in grads_and_vars]
        grads_and_vars = list(zip(reduced_grads, var_list))
      return distribute_ctx.get_replica_context().merge_call(
          functools.partial(self._distributed_apply, apply_state=apply_state),
          args=(grads_and_vars,),
          kwargs={
              "name": name,
          })

  def _aggregate_gradients(self, grads_and_vars):
    """Returns all-reduced gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      A list of all-reduced gradients.
    """
    grads_and_vars = list(grads_and_vars)
    filtered_grads_and_vars = _filter_grads(grads_and_vars)
    def all_reduce_fn(distribution, grads_and_vars):
      return distribution.extended.batch_reduce_to(
          ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    # We switch to a cross-replica context since there is a bug which causes
    # IndexedSlices to be converted to dense tensors when all-reduced in a
    # replica context.
    # TODO(b/150507409): Do not switch to a cross-replica context once the bug
    # is fixed.
    if filtered_grads_and_vars:
      reduced = distribute_ctx.get_replica_context().merge_call(
          all_reduce_fn, args=(filtered_grads_and_vars,))
    else:
      reduced = []
    # Copy 'reduced' but add None gradients back in
    reduced_with_nones = []
    reduced_pos = 0
    for g, _ in grads_and_vars:
      if g is None:
        reduced_with_nones.append(None)
      else:
        reduced_with_nones.append(reduced[reduced_pos])
        reduced_pos += 1
    assert reduced_pos == len(reduced), "Failed to add all gradients"
    return reduced_with_nones

  def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    """`apply_gradients` using a `DistributionStrategy`."""

    def apply_grad_to_update_var(var, grad):
      """Apply gradient to variable."""
      if isinstance(var, ops.Tensor):
        raise NotImplementedError("Trying to update a Tensor ", var)

      apply_kwargs = {}
      if isinstance(grad, ops.IndexedSlices):
        if var.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")
        if "apply_state" in self._sparse_apply_args:
          apply_kwargs["apply_state"] = apply_state
        return self._resource_apply_sparse_duplicate_indices(
            grad.values, var, grad.indices, **apply_kwargs)

      if "apply_state" in self._dense_apply_args:
        apply_kwargs["apply_state"] = apply_state
      update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
      if var.constraint is not None:
        with ops.control_dependencies([update_op]):
          return var.assign(var.constraint(var))
      else:
        return update_op

    eagerly_outside_functions = ops.executing_eagerly_outside_functions()
    update_ops = []
    with ops.name_scope(name or self._name, skip_on_eager=True):
      for grad, var in grads_and_vars:
        # TODO(crccw): It's not allowed to assign PerReplica value to
        # MirroredVariable.  Remove this after we relax this restriction.
        def _assume_mirrored(grad):
          if isinstance(grad, ds_values.PerReplica):
            return ds_values.Mirrored(grad.values)
          return grad

        grad = nest.map_structure(_assume_mirrored, grad)
        # Colocate the update with variables to avoid unnecessary communication
        # delays. See b/136304694.
        with distribution.extended.colocate_vars_with(var):
          with ops.name_scope("update" if eagerly_outside_functions else
                              "update_" + var.op.name, skip_on_eager=True):
            update_ops.extend(distribution.extended.update(
                var, apply_grad_to_update_var, args=(grad,), group=False))

      any_symbolic = any(isinstance(i, ops.Operation) or
                         tf_utils.is_symbolic_tensor(i) for i in update_ops)
      if not context.executing_eagerly() or any_symbolic:
        # If the current context is graph mode or any of the update ops are
        # symbolic then the step update should be carried out under a graph
        # context. (eager updates execute immediately)
        with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
          with ops.control_dependencies(update_ops):
            return self._iterations.assign_add(1, read_value=False)

      return self._iterations.assign_add(1)

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    grads_and_vars = list(zip(grads, params))
    self._assert_valid_dtypes([
        v for g, v in grads_and_vars
        if g is not None and v.dtype != dtypes.resource
    ])
    return [self.apply_gradients(grads_and_vars)]

  def _set_hyper(self, name, value):
    """set hyper `name` to value. value can be callable, tensor, numeric."""
    if isinstance(value, trackable.Trackable):
      self._track_trackable(value, name, overwrite=True)
    if name not in self._hyper:
      self._hyper[name] = value
    else:
      prev_value = self._hyper[name]
      if (callable(prev_value)
          or isinstance(prev_value,
                        (ops.Tensor, int, float,
                         learning_rate_schedule.LearningRateSchedule))
          or isinstance(value, learning_rate_schedule.LearningRateSchedule)):
        self._hyper[name] = value
      else:
        backend.set_value(self._hyper[name], value)

  def _get_hyper(self, name, dtype=None):
    if not self._hypers_created:
      self._create_hypers()
    value = self._hyper[name]
    if isinstance(value, learning_rate_schedule.LearningRateSchedule):
      return value
    if callable(value):
      value = value()
    if dtype:
      return math_ops.cast(value, dtype)
    else:
      return value

  def _create_slots(self, var_list):
    pass

  def _create_all_weights(self, var_list):
    """Creates all weights, including iterations, hyperparameters and slot vars.

    This will add newly created variables to `optimizer.weights`.

    New variables are only created when this method is called the first time, or
    when called with different variables in the var_list.

    Args:
      var_list: list or tuple of `Variable` objects that will be minimized
        using this optimizer.
    """

    _ = self.iterations
    self._create_hypers()
    self._create_slots(var_list)

  def __getattribute__(self, name):
    """Overridden to support hyperparameter access."""
    try:
      return super(OptimizerV2, self).__getattribute__(name)
    except AttributeError as e:
      # Needed to avoid infinite recursion with __setattr__.
      if name == "_hyper":
        raise e
      # Backwards compatibility with Keras optimizers.
      if name == "lr":
        name = "learning_rate"
      if name in self._hyper:
        return self._get_hyper(name)
      raise e

  def __setattr__(self, name, value):
    """Override setattr to support dynamic hyperparameter setting."""
    # Backwards compatibility with Keras optimizers.
    if name == "lr":
      name = "learning_rate"
    if hasattr(self, "_hyper") and name in self._hyper:
      self._set_hyper(name, value)
    else:
      super(OptimizerV2, self).__setattr__(name, value)

  def get_slot_names(self):
    """A list of names for this optimizer's slots."""
    return self._slot_names

  def add_slot(self, var, slot_name, initializer="zeros"):
    """Add a new slot variable for `var`."""
    if slot_name not in self._slot_names:
      self._slot_names.append(slot_name)
    var_key = _var_key(var)
    slot_dict = self._slots.setdefault(var_key, {})
    weight = slot_dict.get(slot_name, None)
    if weight is None:
      if isinstance(initializer, six.string_types) or callable(initializer):
        initializer = initializers.get(initializer)
        initial_value = functools.partial(
            initializer, shape=var.shape, dtype=var.dtype)
      else:
        initial_value = initializer
      strategy = distribute_ctx.get_strategy()
      if not strategy.extended.variable_created_in_scope(var):
        raise ValueError(
            "Trying to create optimizer slot variable under the scope for "
            "tf.distribute.Strategy ({}), which is different from the scope "
            "used for the original variable ({}). Make sure the slot "
            "variables are created under the same strategy scope. This may "
            "happen if you're restoring from a checkpoint outside the scope"
            .format(strategy, var))

      with strategy.extended.colocate_vars_with(var):
        weight = tf_variables.Variable(
            name="%s/%s" % (var._shared_name, slot_name),  # pylint: disable=protected-access
            dtype=var.dtype,
            trainable=False,
            initial_value=initial_value)
      backend.track_variable(weight)
      slot_dict[slot_name] = weight
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=weight)
      self._weights.append(weight)
    return weight

  def get_slot(self, var, slot_name):
    var_key = _var_key(var)
    slot_dict = self._slots[var_key]
    return slot_dict[slot_name]

  def _prepare(self, var_list):
    keys = set()
    for var in var_list:
      if isinstance(var, ds_values.DistributedValues):
        var_devices = var._devices   # pylint: disable=protected-access
      else:
        var_devices = [var.device]
      var_dtype = var.dtype.base_dtype
      for var_device in var_devices:
        keys.add((var_device, var_dtype))

    apply_state = {}
    for var_device, var_dtype in keys:
      apply_state[(var_device, var_dtype)] = {}
      with ops.device(var_device):
        self._prepare_local(var_device, var_dtype, apply_state)

    return apply_state

  def _prepare_local(self, var_device, var_dtype, apply_state):
    if "learning_rate" in self._hyper:
      lr_t = array_ops.identity(self._decayed_lr(var_dtype))
      apply_state[(var_device, var_dtype)]["lr_t"] = lr_t

  def _fallback_apply_state(self, var_device, var_dtype):
    """Compatibility for subclasses that don't pass apply_state through."""
    apply_state = {(var_device, var_dtype): {}}
    self._prepare_local(var_device, var_dtype, apply_state)
    return apply_state[(var_device, var_dtype)]

  def _create_hypers(self):
    if self._hypers_created:
      return
    with self._distribution_strategy_scope():
      # Iterate hyper values deterministically.
      for name, value in sorted(self._hyper.items()):
        if isinstance(value,
                      (ops.Tensor, tf_variables.Variable)) or callable(value):
          continue
        else:
          self._hyper[name] = self.add_weight(
              name,
              shape=[],
              trainable=False,
              initializer=value,
              aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
    self._hypers_created = True

  @property
  def iterations(self):
    """Variable. The number of training steps this Optimizer has run."""
    if self._iterations is None:
      with self._distribution_strategy_scope():
        self._iterations = self.add_weight(
            "iter",
            shape=[],
            dtype=dtypes.int64,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self._weights.append(self._iterations)
    return self._iterations

  @iterations.setter
  def iterations(self, variable):
    if self._iterations is not None:
      raise RuntimeError("Cannot set `iterations` to a new Variable after "
                         "the Optimizer weights have been created")
    self._iterations = variable
    self._weights.append(self._iterations)

  def _decayed_lr(self, var_dtype):
    """Get decayed learning rate as a Tensor with dtype=var_dtype."""
    lr_t = self._get_hyper("learning_rate", var_dtype)
    if isinstance(lr_t, learning_rate_schedule.LearningRateSchedule):
      local_step = math_ops.cast(self.iterations, var_dtype)
      lr_t = math_ops.cast(lr_t(local_step), var_dtype)
    if self._initial_decay > 0.:
      local_step = math_ops.cast(self.iterations, var_dtype)
      decay_t = self._get_hyper("decay", var_dtype)
      lr_t = lr_t / (1. + decay_t * local_step)
    return lr_t

  @abc.abstractmethod
  def get_config(self):
    """Returns the config of the optimizer.

    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer.
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.

    Returns:
        Python dictionary.
    """
    config = {"name": self._name}
    if self.clipnorm is not None:
      config["clipnorm"] = self.clipnorm
    if self.clipvalue is not None:
      config["clipvalue"] = self.clipvalue
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates an optimizer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An optimizer instance.
    """
    if "lr" in config:
      config["learning_rate"] = config.pop("lr")
    if "learning_rate" in config:
      if isinstance(config["learning_rate"], dict):
        config["learning_rate"] = learning_rate_schedule.deserialize(
            config["learning_rate"], custom_objects=custom_objects)
    return cls(**config)

  def _serialize_hyperparameter(self, hyperparameter_name):
    """Serialize a hyperparameter that can be a float, callable, or Tensor."""
    value = self._hyper[hyperparameter_name]
    if isinstance(value, learning_rate_schedule.LearningRateSchedule):
      return learning_rate_schedule.serialize(value)
    if callable(value):
      return value()
    if tensor_util.is_tensor(value):
      return backend.get_value(value)
    return value

  def variables(self):
    """Returns variables of this Optimizer based on the order created."""
    return self._weights

  @property
  def weights(self):
    """Returns variables of this Optimizer based on the order created."""
    return self._weights

  def get_weights(self):
    """Returns the current weights of the optimizer.

    The weights of an optimizer are its state (ie, variables).
    This function returns the weight values associated with this
    optimizer as a list of Numpy arrays. The first value is always the
    iterations count of the optimizer, followed by the optimizer's state
    variables in the order they were created. The returned list can in turn
    be used to load state into similarly parameterized optimizers.

    For example, the RMSprop optimizer for this simple model returns a list of
    three values-- the iteration count, followed by the root-mean-square value
    of the kernel and bias of the single Dense layer:

    >>> opt = tf.keras.optimizers.RMSprop()
    >>> m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> m.compile(opt, loss='mse')
    >>> data = np.arange(100).reshape(5, 20)
    >>> labels = np.zeros(5)
    >>> print('Training'); results = m.fit(data, labels)
    Training ...
    >>> len(opt.get_weights())
    3

    Returns:
        Weights values as a list of numpy arrays.
    """
    params = self.weights
    return backend.batch_get_value(params)

  # TODO(tanzheny): Maybe share this logic with base_layer.
  def set_weights(self, weights):
    """Set the weights of the optimizer.

    The weights of an optimizer are its state (ie, variables).
    This function takes the weight values associated with this
    optimizer as a list of Numpy arrays. The first value is always the
    iterations count of the optimizer, followed by the optimizer's state
    variables in the order they are created. The passed values are used to set
    the new state of the optimizer.

    For example, the RMSprop optimizer for this simple model takes a list of
    three values-- the iteration count, followed by the root-mean-square value
    of the kernel and bias of the single Dense layer:

    >>> opt = tf.keras.optimizers.RMSprop()
    >>> m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> m.compile(opt, loss='mse')
    >>> data = np.arange(100).reshape(5, 20)
    >>> labels = np.zeros(5)
    >>> print('Training'); results = m.fit(data, labels)
    Training ...
    >>> new_weights = [np.array(10), np.ones([20, 10]), np.zeros([10])]
    >>> opt.set_weights(new_weights)
    >>> opt.iterations
    <tf.Variable 'RMSprop/iter:0' shape=() dtype=int64, numpy=10>

    Arguments:
        weights: weight values as a list of numpy arrays.
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError(
          "You called `set_weights(weights)` on optimizer " + self._name +
          " with a  weight list of length " + str(len(weights)) +
          ", but the optimizer was expecting " + str(len(params)) +
          " weights. Provided weights: " + str(weights)[:50] + "...")
    if not params:
      return
    weight_value_tuples = []
    param_values = backend.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError("Optimizer weight shape " + str(pv.shape) +
                         " not compatible with "
                         "provided weight shape " + str(w.shape))
      weight_value_tuples.append((p, w))
    backend.batch_set_value(weight_value_tuples)

  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer="zeros",
                 trainable=None,
                 synchronization=tf_variables.VariableSynchronization.AUTO,
                 aggregation=tf_variables.VariableAggregation.NONE):

    if dtype is None:
      dtype = dtypes.float32
    if isinstance(initializer, six.string_types) or callable(initializer):
      initializer = initializers.get(initializer)

    if synchronization == tf_variables.VariableSynchronization.ON_READ:
      if trainable:
        raise ValueError(
            "Synchronization value can be set to "
            "VariableSynchronization.ON_READ only for non-trainable variables. "
            "You have specified trainable=True and "
            "synchronization=VariableSynchronization.ON_READ.")
      else:
        # Set trainable to be false when variable is to be synced on read.
        trainable = False
    elif trainable is None:
      trainable = True

    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        getter=base_layer_utils.make_variable,
        overwrite=True,
        initializer=initializer,
        dtype=dtype,
        trainable=trainable,
        use_resource=True,
        synchronization=synchronization,
        aggregation=aggregation)
    backend.track_variable(variable)

    return variable

  def _init_set_name(self, name, zero_based=True):
    if not name:
      self._name = backend.unique_object_name(
          generic_utils.to_snake_case(self.__class__.__name__),
          zero_based=zero_based)
    else:
      self._name = name

  def _assert_valid_dtypes(self, tensors):
    """Asserts tensors are all valid types (see `_valid_dtypes`).

    Args:
      tensors: Tensors to check.

    Raises:
      ValueError: If any tensor is not a valid type.
    """
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError("Invalid type %r for %s, expected: %s." %
                         (dtype, t.name, [v for v in valid_dtypes]))

  def _valid_dtypes(self):
    """Valid types for loss, variables and gradients.

    Subclasses should override to allow other float types.

    Returns:
      Valid types for loss, variables and gradients.
    """
    return set([
        dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64,
        dtypes.complex64, dtypes.complex128
    ])

  def _call_if_callable(self, param):
    """Call the function if param is callable."""
    return param() if callable(param) else param

  def _resource_apply_dense(self, grad, handle, apply_state):
    """Add ops to apply dense gradients to the variable `handle`.

    Args:
      grad: a `Tensor` representing the gradient.
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      apply_state: A dict which is used across multiple apply calls.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError("Must be implemented in subclasses.")

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices,
                                               **kwargs):
    """Add ops to apply sparse gradients to `handle`, with repeated indices.

    Optimizers which override this method must deal with repeated indices. See
    the docstring of `_apply_sparse_duplicate_indices` for details. By default
    the correct behavior, to sum non-unique indices and their associated
    gradients, is enforced by first pre-processing `grad` and `indices` and
    passing them on to `_resource_apply_sparse`. Optimizers which deal correctly
    with duplicate indices may instead override this method to avoid the
    overhead of summing.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      indices: a `Tensor` of integral type representing the indices for which
        the gradient is nonzero. Indices may be repeated.
      **kwargs: May optionally contain `apply_state`

    Returns:
      An `Operation` which updates the value of the variable.
    """
    summed_grad, unique_indices = _deduplicate_indexed_slices(
        values=grad, indices=indices)
    return self._resource_apply_sparse(summed_grad, handle, unique_indices,
                                       **kwargs)

  def _resource_apply_sparse(self, grad, handle, indices, apply_state):
    """Add ops to apply sparse gradients to the variable `handle`.

    Similar to `_apply_sparse`, the `indices` argument to this method has been
    de-duplicated. Optimizers which deal correctly with non-unique indices may
    instead override `_resource_apply_sparse_duplicate_indices` to avoid this
    overhead.

    Args:
      grad: a `Tensor` representing the gradient for the affected indices.
      handle: a `Tensor` of dtype `resource` which points to the variable to be
        updated.
      indices: a `Tensor` of integral type representing the indices for which
        the gradient is nonzero. Indices are unique.
      apply_state: A dict which is used across multiple apply calls.

    Returns:
      An `Operation` which updates the value of the variable.
    """
    raise NotImplementedError("Must be implemented in subclasses.")

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_scatter_update(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_update(x.handle, i, v)]):
      return x.value()

  @property
  @tracking.cached_per_instance
  def _dense_apply_args(self):
    return tf_inspect.getfullargspec(self._resource_apply_dense).args

  @property
  @tracking.cached_per_instance
  def _sparse_apply_args(self):
    return tf_inspect.getfullargspec(self._resource_apply_sparse).args

  # ---------------
  # For implementing the trackable interface
  # ---------------

  def _restore_slot_variable(self, slot_name, variable, slot_variable):
    """Restore a newly created slot variable's value."""
    variable_key = _var_key(variable)
    deferred_restorations = self._deferred_slot_restorations.get(
        slot_name, {}).pop(variable_key, [])
    # Iterate over restores, highest restore UID first to minimize the number
    # of assignments.
    deferred_restorations.sort(key=lambda position: position.restore_uid,
                               reverse=True)
    for checkpoint_position in deferred_restorations:
      checkpoint_position.restore(slot_variable)

  def _create_or_restore_slot_variable(
      self, slot_variable_position, slot_name, variable):
    """Restore a slot variable's value, possibly creating it.

    Called when a variable which has an associated slot variable is created or
    restored. When executing eagerly, we create the slot variable with a
    restoring initializer.

    No new variables are created when graph building. Instead,
    _restore_slot_variable catches these after normal creation and adds restore
    ops to the graph. This method is nonetheless important when graph building
    for the case when a slot variable has already been created but `variable`
    has just been added to a dependency graph (causing us to realize that the
    slot variable needs to be restored).

    Args:
      slot_variable_position: A `trackable._CheckpointPosition` object
        indicating the slot variable `Trackable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    variable_key = _var_key(variable)
    slot_dict = self._slots.get(variable_key, {})
    slot_variable = slot_dict.get(slot_name, None)
    if (slot_variable is None and context.executing_eagerly() and
        slot_variable_position.is_simple_variable()
        # Defer slot variable creation if there is an active variable creator
        # scope. Generally we'd like to eagerly create/restore slot variables
        # when possible, but this may mean that scopes intended to catch
        # `variable` also catch its eagerly created slot variable
        # unintentionally (specifically make_template would add a dependency on
        # a slot variable if not for this case). Deferring is mostly harmless
        # (aside from double initialization), and makes variable creator scopes
        # behave the same way they do when graph building.
        and not ops.get_default_graph()._variable_creator_stack):  # pylint: disable=protected-access
      initializer = trackable.CheckpointInitialValue(
          checkpoint_position=slot_variable_position)
      slot_variable = self.add_slot(
          var=variable,
          initializer=initializer,
          slot_name=slot_name)
      # Slot variables are not owned by any one object (because we don't want to
      # save the slot variable if the optimizer is saved without the non-slot
      # variable, or if the non-slot variable is saved without the optimizer;
      # it's a dependency hypergraph with edges of the form (optimizer, non-slot
      # variable, variable)). So we don't _track_ slot variables anywhere, and
      # instead special-case this dependency and otherwise pretend it's a normal
      # graph.
    if slot_variable is not None:
      # If we've either made this slot variable, or if we've pulled out an
      # existing slot variable, we should restore it.
      slot_variable_position.restore(slot_variable)
    else:
      # We didn't make the slot variable. Defer restoring until it gets created
      # normally. We keep a list rather than the one with the highest restore
      # UID in case slot variables have their own dependencies, in which case
      # those could differ between restores.
      self._deferred_slot_restorations.setdefault(
          slot_name, {}).setdefault(variable_key, []).append(
              slot_variable_position)

  @contextlib.contextmanager
  def _distribution_strategy_scope(self):
    """Returns the `tf.distribute.Strategy` this optimizer was created under."""
    if self._distribution_strategy and not distribute_ctx.has_strategy():
      with self._distribution_strategy.scope():
        yield self._distribution_strategy.scope()
    else:
      yield


def _filter_grads(grads_and_vars):
  """Filter out iterable with grad equal to None."""
  grads_and_vars = tuple(grads_and_vars)
  if not grads_and_vars:
    return grads_and_vars
  filtered = []
  vars_with_empty_grads = []
  for grad, var in grads_and_vars:
    if grad is None:
      vars_with_empty_grads.append(var)
    else:
      filtered.append((grad, var))
  filtered = tuple(filtered)
  if not filtered:
    raise ValueError("No gradients provided for any variable: %s." %
                     ([v.name for _, v in grads_and_vars],))
  if vars_with_empty_grads:
    logging.warning(
        ("Gradients do not exist for variables %s when minimizing the loss."),
        ([v.name for v in vars_with_empty_grads]))
  return filtered


def _var_key(var):
  """Key for representing a primary variable, for looking up slots.

  In graph mode the name is derived from the var shared name.
  In eager mode the name is derived from the var unique id.
  If distribution strategy exists, get the primary variable first.

  Args:
    var: the variable.

  Returns:
    the unique name of the variable.
  """

  # pylint: disable=protected-access
  # Get the distributed variable if it exists.
  if hasattr(var, "_distributed_container"):
    var = var._distributed_container
  if var._in_graph_mode:
    return var._shared_name
  return var._unique_id


def _get_slot_key_from_var(var, slot_name):
  """Get the slot key for the variable: var_name/slot_name."""

  name = _var_key(var)
  return name + "/" + slot_name


class RestoredOptimizer(OptimizerV2):
  """A non-functional Optimizer implementation for checkpoint compatibility.

  Holds slot variables and hyperparameters when an optimizer is restored from a
  SavedModel. These variables may be referenced in functions along with ops
  created by the original optimizer, but currently we do not support using the
  optimizer object iself (e.g. through `apply_gradients`).
  """
  # TODO(allenl): Make the restored optimizer functional by tracing its apply
  # methods.

  def __init__(self):
    super(RestoredOptimizer, self).__init__("RestoredOptimizer")
    self._hypers_created = True

  def get_config(self):
    # TODO(allenl): Save and restore the Optimizer's config
    raise NotImplementedError(
        "Restoring functional Optimizers from SavedModels is not currently "
        "supported. Please file a feature request if this limitation bothers "
        "you.")

revived_types.register_revived_type(
    "optimizer",
    lambda obj: isinstance(obj, OptimizerV2),
    versions=[revived_types.VersionedTypeRegistration(
        object_factory=lambda proto: RestoredOptimizer(),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=RestoredOptimizer._set_hyper  # pylint: disable=protected-access
    )])
