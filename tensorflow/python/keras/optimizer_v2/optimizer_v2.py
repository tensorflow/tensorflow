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
import functools

import six

from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import reduce_util as ds_reduce_util
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
  """Updated base class for optimizers.

  This class defines the API to add Ops to train a model.  You never use this
  class directly, but instead instantiate one of its subclasses such as
  `tf.keras.optimizers.SGD`, `tf.keras.optimizers.Adam`.

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

  ### Custom training loop with Keras models

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

  ### Processing gradients before applying them.

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

  ### Use with `tf.distribute.Strategy`.

  This optimizer class is `tf.distribute.Strategy` aware, which means it
  automatically sums gradients across all replicas. To average gradients,
  you divide your loss by the global batch size, which is done
  automatically if you use `tf.keras` built-in training or evaluation loops.
  See the `reduction` argument of your loss which should be set to
  `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` for averaging or
  `tf.keras.losses.Reduction.SUM` for not.

  If you are not using these and you want to average gradients, you should use
  `tf.math.reduce_sum` to add up your per-example losses and then divide by the
  global batch size. Note that when using `tf.distribute.Strategy`, the first
  component of a tensor's shape is the *replica-local* batch size, which is off
  by a factor equal to the number of replicas being used to compute a single
  step. As a result, using `tf.math.reduce_mean` will give the wrong answer,
  resulting in gradients that can be many times too big.

  ### Variable Constraint

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

  ### Hyper parameters

  These are arguments passed to the optimizer subclass constructor
  (the `__init__` method), and then passed to `self._set_hyper()`.
  They can be either regular Python values (like 1.0), tensors, or
  callables. If they are callable, the callable will be called during
  `apply_gradients()` to get the value for the hyper parameter.

  Hyper parameters can be overwritten through user code:

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

  ### Write a customized optimizer.
  If you intend to create your own optimization algorithm, simply inherit from
  this class and override the following methods:

    - resource_apply_dense (update variable given gradient tensor is dense)
    - resource_apply_sparse (update variable given gradient tensor is sparse)
    - create_slots (if your optimizer algorithm requires additional variables)
    - get_config (serialization of the optimizer, include all hyper parameters)
  """

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
      RuntimeError: If _create_slots has been overridden instead of
          _create_vars.
    """
    allowed_kwargs = {"clipnorm", "clipvalue", "lr", "decay"}
    for k in kwargs:
      if k not in allowed_kwargs:
        raise TypeError("Unexpected keyword argument "
                        "passed to optimizer: " + str(k))
      # checks that all keyword arguments are non-negative.
      if kwargs[k] < 0:
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
    if "clipnorm" in kwargs:
      self.clipnorm = kwargs.pop("clipnorm")
    if "clipvalue" in kwargs:
      self.clipvalue = kwargs.pop("clipvalue")

    self._hypers_created = False

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
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.

    """
    grads_and_vars = self._compute_gradients(
        loss, var_list=var_list, grad_loss=grad_loss)

    return self.apply_gradients(grads_and_vars, name=name)

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

      if hasattr(self, "clipnorm"):
        grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
      if hasattr(self, "clipvalue"):
        grads = [
            clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
            for g in grads
        ]

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
      if hasattr(self, "clipnorm"):
        grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
      if hasattr(self, "clipvalue"):
        grads = [
            clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
            for g in grads
        ]
    return grads

  def apply_gradients(self, grads_and_vars, name=None):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

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
        _ = self.iterations
        self._create_hypers()
        self._create_slots(var_list)

      apply_state = self._prepare(var_list)
      return distribute_ctx.get_replica_context().merge_call(
          functools.partial(self._distributed_apply, apply_state=apply_state),
          args=(grads_and_vars,),
          kwargs={"name": name})

  def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    """`apply_gradients` using a `DistributionStrategy`."""
    reduced_grads = distribution.extended.batch_reduce_to(
        ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    var_list = [v for _, v in grads_and_vars]
    grads_and_vars = zip(reduced_grads, var_list)

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

    update_ops = []
    with backend.name_scope(name or self._name):
      for grad, var in grads_and_vars:
        scope_name = ("update" if ops.executing_eagerly_outside_functions() else
                      "update_" + var.op.name)
        # Colocate the update with variables to avoid unnecessary communication
        # delays. See b/136304694.
        with backend.name_scope(
            scope_name), distribution.extended.colocate_vars_with(var):
          update_ops.extend(
              distribution.extended.update(
                  var, apply_grad_to_update_var, args=(grad,), group=False))

      any_symbolic = any(isinstance(i, ops.Operation) or
                         tf_utils.is_symbolic_tensor(i) for i in update_ops)
      if not context.executing_eagerly() or any_symbolic:
        # If the current context is graph mode or any of the update ops are
        # symbolic then the step update should be carried out under a graph
        # context. (eager updates execute immediately)
        with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
          with ops.control_dependencies(update_ops):
            return self._iterations.assign_add(1).op

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
      var_devices = (getattr(var, "devices", None) or  # Distributed
                     [var.device])                     # Regular
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
    # Iterate hyper values deterministically.
    for name, value in sorted(self._hyper.items()):
      if isinstance(
          value, (ops.Tensor, tf_variables.Variable)) or callable(value):
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
    """Returns the config of the optimimizer.

    An optimizer config is a Python dictionary (serializable)
    containing the configuration of an optimizer.
    The same optimizer can be reinstantiated later
    (without any saved state) from this configuration.

    Returns:
        Python dictionary.
    """
    config = {"name": self._name}
    if hasattr(self, "clipnorm"):
      config["clipnorm"] = self.clipnorm
    if hasattr(self, "clipvalue"):
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
    params = self.weights
    return backend.batch_get_value(params)

  # TODO(tanzheny): Maybe share this logic with base_layer.
  def set_weights(self, weights):
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
    return set(
        [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64])

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
    raise NotImplementedError()

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
    raise NotImplementedError()

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
    var = var._distributed_container()
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
        "Restoring functional Optimzers from SavedModels is not currently "
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
