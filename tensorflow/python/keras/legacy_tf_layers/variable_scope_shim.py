# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
# pylint: disable=g-classes-have-attributes
"""Contains a shim to allow using TF1 get_variable code in TF2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator


def as_shape(shape):
  """Converts the given object to a TensorShape."""
  if isinstance(shape, tensor_shape.TensorShape):
    return shape
  else:
    return tensor_shape.TensorShape(shape)


def _is_callable_object(obj):
  return hasattr(obj, "__call__") and tf_inspect.ismethod(obj.__call__)


def _has_kwargs(fn):
  """Returns whether the passed callable has **kwargs in its signature.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `bool`: if `fn` has **kwargs in its signature.

  Raises:
     `TypeError`: If fn is not a Function, or function-like object.
  """
  if isinstance(fn, functools.partial):
    fn = fn.func
  elif _is_callable_object(fn):
    fn = fn.__call__
  elif not callable(fn):
    raise TypeError(
        "fn should be a function-like object, but is of type {}.".format(
            type(fn)))
  return tf_inspect.getfullargspec(fn).varkw is not None


def fn_args(fn):
  """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.

  Raises:
    ValueError: if partial function has positionally bound arguments
  """
  if isinstance(fn, functools.partial):
    args = fn_args(fn.func)
    args = [a for a in args[len(fn.args):] if a not in (fn.keywords or [])]
  else:
    if hasattr(fn, "__call__") and tf_inspect.ismethod(fn.__call__):
      fn = fn.__call__
    args = tf_inspect.getfullargspec(fn).args
    if _is_bound_method(fn) and args:
      # If it's a bound method, it may or may not have a self/cls first
      # argument; for example, self could be captured in *args.
      # If it does have a positional argument, it is self/cls.
      args.pop(0)
  return tuple(args)


def _is_bound_method(fn):
  _, fn = tf_decorator.unwrap(fn)
  return tf_inspect.ismethod(fn) and (fn.__self__ is not None)


def validate_synchronization_aggregation_trainable(
    synchronization, aggregation, trainable, name):
  """Given user-provided variable properties, sets defaults and validates."""
  if aggregation is None:
    aggregation = variables.VariableAggregation.NONE
  else:
    if not isinstance(aggregation,
                      (variables.VariableAggregation,
                       variables.VariableAggregationV2)):
      try:
        aggregation = variables.VariableAggregationV2(aggregation)
      except ValueError:
        raise ValueError(
            "Invalid variable aggregation mode: {} for variable: {}".format(
                aggregation, name))
  if synchronization is None:
    synchronization = variables.VariableSynchronization.AUTO
  else:
    try:
      synchronization = variables.VariableSynchronization(synchronization)
    except ValueError:
      raise ValueError(
          "Invalid variable synchronization mode: {} for variable: {}".format(
              synchronization, name))
  if trainable is None:
    trainable = synchronization != variables.VariableSynchronization.ON_READ
  return synchronization, aggregation, trainable


class _EagerVariableStore(object):
  """TF2-compatible VariableStore that avoids collections & tracks regularizers.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  All variables get created in `tf.init_scope.` to avoid a bad
  interaction between `tf.function` `FuncGraph` internals, Keras
  Functional Models, and TPUStrategy variable initialization.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """

  __slots__ = ["_vars", "_regularizers", "_store_eager_variables"]

  def __init__(self):
    """Create a variable store."""
    self._vars = {}  # A dictionary of the stored TensorFlow variables.
    self._regularizers = {}  # A dict mapping var names to their regularizers.
    self._store_eager_variables = True

  def get_variable(
      self,
      name,
      shape=None,
      dtype=dtypes.float32,
      initializer=None,
      regularizer=None,
      reuse=None,
      trainable=None,
      collections=None,
      caching_device=None,
      partitioner=None,
      validate_shape=True,
      use_resource=None,
      custom_getter=None,
      constraint=None,
      synchronization=vs.VariableSynchronization.AUTO,
      aggregation=vs.VariableAggregation.NONE):
    """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If a partitioner is provided, a `PartitionedVariable` is returned.
    Accessing this object as a `Tensor` returns the shards concatenated along
    the partition axis.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: Initializer for the variable.
      regularizer: A (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables. When eager execution is enabled  this argument is always
        forced to be False.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`
        defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys to add the `Variable` to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the `Variable` reside, to
        deduplicate copying through `Switch` and other conditional statements.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates
        instead an experimental ResourceVariable which has well-defined
        semantics. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be true.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. The signature
        of `custom_getter` should match that of this method,
        but the most future-proof version will allow for changes: `def
          custom_getter(getter, *args, **kwargs)`.  Direct access to
        all `get_variable` parameters is also allowed: `def
          custom_getter(getter, name, *args, **kwargs)`.  A simple identity
        custom getter that simply creates variables with modified names is:
          ```python
        def custom_getter(getter, name, *args, **kwargs): return getter(name +
          '_suffix', *args, **kwargs) ```
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Returns:
      The created or existing `Variable` (or `PartitionedVariable`, if a
      partitioner was used).

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
      RuntimeError: when eager execution is enabled and not called from an
        EagerVariableStore.
    """
    if custom_getter is not None and not callable(custom_getter):
      raise ValueError("Passed a custom_getter which is not callable: %s" %
                       custom_getter)

    with ops.init_scope():
      if context.executing_eagerly():
        # Variable creation and initialization takes place in `init_scope`s;
        # as such, if an `init_scope` lifts us into the eager context, then we
        # need to use `ResourceVariable`s.
        use_resource = True

    # Note that it's fine to reuse eager variables whose initialization was
    # lifted from a function-building graph into the eager context (that's why
    # the following clause is not wrapped in an `init_scope`); lifted variables
    # are tracked by the graph's `VariableStore`.
    if context.executing_eagerly():
      reuse = vs.AUTO_REUSE

    # If a *_ref type is passed in an error would be triggered further down the
    # stack. We prevent this using base_dtype to get a non-ref version of the
    # type, before doing anything else. When _ref types are removed in favor of
    # resources, this line can be removed.
    try:
      dtype = dtype.base_dtype
    except AttributeError:
      # .base_dtype not existing means that we will try and use the raw dtype
      # which was passed in - this might be a NumPy type which is valid.
      pass

    # This is the main logic of get_variable.  However, custom_getter
    # may override this logic.  So we save it as a callable and pass
    # it to custom_getter.
    # Note: the parameters of _true_getter, and their documentation, match
    # *exactly* item-for-item with the docstring of this method.
    def _true_getter(  # pylint: disable=missing-docstring
        name,
        shape=None,
        dtype=dtypes.float32,
        initializer=None,
        regularizer=None,
        reuse=None,
        trainable=None,
        collections=None,  # pylint: disable=unused-argument
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        use_resource=None,  # pylint: disable=unused-argument
        constraint=None,
        synchronization=vs.VariableSynchronization.AUTO,
        aggregation=vs.VariableAggregation.NONE):
      # Partitioned variable currently unsupported w/ the shim
      if partitioner is not None:
        raise ValueError(
            "`partitioner` arg for `get_variable` is unsupported in TF2."
            "File a bug if you need help. You passed %s" % partitioner)

      # Single variable case
      if "%s/part_0" % name in self._vars:
        raise ValueError(
            "No partitioner was provided, but a partitioned version of the "
            "variable was found: %s/part_0. Perhaps a variable of the same "
            "name was already created with partitioning?" % name)

      return self._get_single_variable(
          name=name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          caching_device=caching_device,
          validate_shape=validate_shape,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

    synchronization, aggregation, trainable = (
        validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name))

    if custom_getter is not None:
      # Handle backwards compatibility with getter arguments that were added
      # to the API after users started writing custom getters.
      custom_getter_kwargs = {
          "getter": _true_getter,
          "name": name,
          "shape": shape,
          "dtype": dtype,
          "initializer": initializer,
          "regularizer": regularizer,
          "reuse": reuse,
          "trainable": trainable,
          "collections": collections,
          "caching_device": caching_device,
          "partitioner": partitioner,
          "validate_shape": validate_shape,
          "use_resource": use_resource,
          "synchronization": synchronization,
          "aggregation": aggregation,
      }
      # `fn_args` and `has_kwargs` can handle functions, `functools.partial`,
      # `lambda`.
      if ("constraint" in fn_args(custom_getter) or
          _has_kwargs(custom_getter)):
        custom_getter_kwargs["constraint"] = constraint
      return custom_getter(**custom_getter_kwargs)
    else:
      return _true_getter(
          name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

  def _get_single_variable(
      self,
      name,
      shape=None,
      dtype=dtypes.float32,
      initializer=None,
      regularizer=None,
      partition_info=None,
      reuse=None,
      trainable=None,
      caching_device=None,
      validate_shape=True,
      constraint=None,
      synchronization=vs.VariableSynchronization.AUTO,
      aggregation=vs.VariableAggregation.NONE):
    """Get or create a single Variable (e.g.

    a shard or entire variable).

    See the documentation of get_variable above (ignore partitioning components)
    for details.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.
      initializer: see get_variable.
      regularizer: see get_variable.
      partition_info: _PartitionInfo object.
      reuse: see get_variable.
      trainable: see get_variable.
      caching_device: see get_variable.
      validate_shape: see get_variable.
      constraint: see get_variable.
      synchronization: see get_variable.
      aggregation: see get_variable.

    Returns:
      A Variable.  See documentation of get_variable above.

    Raises:
      ValueError: See documentation of get_variable above.
    """
    # Set to true if initializer is a constant.
    initializing_from_value = False
    if initializer is not None and not callable(initializer):
      initializing_from_value = True
    if shape is not None and initializing_from_value:
      raise ValueError("If initializer is a constant, do not specify shape.")

    dtype = dtypes.as_dtype(dtype)
    shape = as_shape(shape)

    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if reuse is False:  # pylint: disable=g-bool-id-comparison
        err_msg = ("Variable %s already exists, disallowed."
                   " Did you mean to set reuse=True or "
                   "reuse=tf.AUTO_REUSE in VarScope?" % name)
        # ResourceVariables don't have an op associated with so no traceback
        raise ValueError(err_msg)
      found_var = self._vars[name]
      if not shape.is_compatible_with(found_var.get_shape()):
        raise ValueError("Trying to share variable %s, but specified shape %s"
                         " and found shape %s." %
                         (name, shape, found_var.get_shape()))
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError("Trying to share variable %s, but specified dtype %s"
                         " and found dtype %s." %
                         (name, dtype_str, found_type_str))
      return found_var

    # The code below handles only the case of creating a new variable.
    if reuse is True:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Variable %s does not exist, or was not created with "
                       "tf.get_variable(). Did you mean to set "
                       "reuse=tf.AUTO_REUSE in VarScope?" % name)

    # Create the tensor to initialize the variable with default value.
    if initializer is None:
      initializer, initializing_from_value = self._get_default_initializer(
          name=name, shape=shape, dtype=dtype)
    # Enter an init scope when creating the initializer.
    with ops.init_scope():
      if initializing_from_value:
        init_val = initializer
        variable_dtype = None
      else:
        # Instantiate initializer if provided initializer is a type object.
        if tf_inspect.isclass(initializer):
          initializer = initializer()
        if shape.is_fully_defined():
          if "partition_info" in tf_inspect.getargspec(initializer).args:
            init_val = functools.partial(initializer,
                                         shape.as_list(),
                                         dtype=dtype,
                                         partition_info=partition_info)
          else:
            init_val = functools.partial(initializer,
                                         shape.as_list(), dtype=dtype)
          variable_dtype = dtype.base_dtype
        else:
          init_val = initializer
          variable_dtype = None

    # Create the variable (Always eagerly as a workaround for a strange
    # tpu / funcgraph / keras functional model interaction )
    with ops.init_scope():
      v = variables.Variable(
          initial_value=init_val,
          name=name,
          trainable=trainable,
          caching_device=caching_device,
          dtype=variable_dtype,
          validate_shape=validate_shape,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

    self._vars[name] = v
    logging.vlog(1, "Created variable %s with shape %s and init %s", v.name,
                 format(shape), initializer)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      self.add_regularizer(v, regularizer)

    return v

  def add_regularizer(self, var, regularizer):
    self._regularizers[var.name] = functools.partial(regularizer, var)

  # Initialize variable when no initializer provided
  def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
    """Provide a default initializer and a corresponding value.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.

    Returns:
      initializer and initializing_from_value. See get_variable above.

    Raises:
      ValueError: When giving unsupported dtype.
    """
    del shape
    # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
    if dtype.is_floating:
      initializer = init_ops.glorot_uniform_initializer()
      initializing_from_value = False
    # If dtype is DT_INT/DT_UINT, provide a default value `zero`
    # If dtype is DT_BOOL, provide a default value `FALSE`
    elif (dtype.is_integer or dtype.is_unsigned or dtype.is_bool or
          dtype == dtypes.string):
      initializer = init_ops.zeros_initializer()
      initializing_from_value = False
    # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
    else:
      raise ValueError("An initializer for variable %s of %s is required" %
                       (name, dtype.base_dtype))

    return initializer, initializing_from_value


class VariableAndLossTracker(module.Module):
  """Module that has a scope to capture vars/losses made by `get_variable`."""

  def __init__(self):
    self._var_store = _EagerVariableStore()  # pylint: disable=protected-access
    self._variables = {}

  def _variable_creator(self, next_creator, **kwargs):
    var = next_creator(**kwargs)
    self._variables[var.name] = var

    return var

  @tf_contextlib.contextmanager
  def scope(self):
    with vs.variable_creator_scope(
        self._variable_creator), vs.with_variable_store(self._var_store):
      yield

  def get_regularization_losses(self):
    # TODO(kaftan): Consider adding a regex scope like the collection access.
    # But, < 40-50 usages of get_regularization_loss(es) with `scope`
    # & possible to do manually?
    losses = {}
    for var_name, regularizer in self._var_store._regularizers.items():  # pylint: disable=protected-access
      losses[var_name] = regularizer()
    return losses


class VariableScopeWrapperLayer(base_layer.Layer):
  """Wrapper Layer to capture `compat.v1.get_variable` and `compat.v1.layers`.

  See go/tf2-migration-model-bookkeeping for background.

  This shim layer allows using large sets of TF1 model-forward-pass code as a
  Keras layer that works in TF2 with TF2 behaviors enabled. To use it,
  override this class and put your TF1 model's forward pass inside your
  implementation for `forward_pass`.

  Below are some examples, and then more details on the functionality of this
  shhim layer to wrap TF1 model forward passes.

  Example of capturing tf.compat.v1.layer-based modeling code as a Keras layer:

  ```python
  class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeWrapperLayer):

    def __init__(self, units, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.units = units

    def forward_pass(self, inputs, training=None):
      out = tf.compat.v1.layers.dense(
          inputs, self.units, name="dense_one",
          kernel_initializer=init_ops.ones_initializer(),
          kernel_regularizer="l2")
      with variable_scope.variable_scope("nested_scope"):
        out = tf.compat.v1.layers.dense(
            out, self.units, name="dense_two",
            kernel_initializer=init_ops.ones_initializer(),
            kernel_regularizer="l2")
      return out

  # Create a layer that can be used as a standard keras layer
  layer = WrappedDoubleDenseLayer(10)

  # call the layer on inputs
  layer(...)

  # Variables created/used within the scope will be tracked by the layer
  layer.weights
  layer.trainable_variables

  # Regularization losses will be captured in layer.losses after a call,
  # just like any other Keras layer
  reg_losses = layer.losses
  ```

  The solution is to wrap the model construction and execution in a keras-style
  scope:

  ```python
  class WrappedDoubleDenseLayer(variable_scope_shim.VariableScopeWrapperLayer):

    def __init__(self, units, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.units = units

    def forward_pass(self, inputs, training=None):
      out = inputs
      with tf.compat.v1.variable_scope("dense_one"):
        # The weights are created with a `regularizer`,
        # so the layer should track their regularization losses
        kernel = tf.compat.v1.get_variable(
            shape=[out.shape[-1], self.units],
            regularizer=regularizers.L2(),
            initializer=init_ops.ones_initializer(),
            name="kernel")
        bias = tf.compat.v1.get_variable(
            shape=[self.units,],
            initializer=init_ops.zeros_initializer(),
            name="bias")
        out = tf.compat.v1.math.matmul(out, kernel)
        out = tf.compat.v1.nn.bias_add(out, bias)
      with tf.compat.v1.variable_scope("nested_scope"):
        with tf.compat.v1.variable_scope("dense_two"):
          kernel = tf.compat.v1.get_variable(
              shape=[out.shape[-1], self.units],
              regularizer=regularizers.L2(),
              initializer=init_ops.ones_initializer(),
              name="kernel")
          bias = tf.compat.v1.get_variable(
              shape=[self.units,],
              initializer=init_ops.zeros_initializer(),
              name="bias")
          out = tf.compat.v1.math.matmul(out, kernel)
          out = tf.compat.v1.nn.bias_add(out, bias)
      return out

  # Create a layer that can be used as a standard keras layer
  layer = WrappedDoubleDenseLayer(10)

  # call the layer on inputs
  layer(...)

  # Variables created/used within the scope will be tracked by the layer
  layer.weights
  layer.trainable_variables

  # Regularization losses will be captured in layer.losses after a call,
  # just like any other Keras layer
  reg_losses = layer.losses
  ```

  Regularization losses:
    Any regularizers specified in the `get_variable` calls or `compat.v1.layer`
    creations will get captured by this wrapper layer. Regularization losses
    are accessible in `layer.losses` after a call just like in a standard
    Keras layer, and will be captured by any model that includes this layer.

  Variable scope / variable reuse:
    variable-scope based reuse in the `forward_pass` will be respected,
    and work like variable-scope based reuse in TF1.

  Variable Names/Pre-trained checkpoint loading:
    variable naming from get_variable and `compat.v1.layer` layers will match
    the TF1 names, so you should be able to re-use your old name-based
    checkpoints.

  Training Arg in `forward_pass`:
    Keras will pass a `training` arg to this layer similarly to how it
    passes `training` to other layers in TF2. See more details in the docs
    on `tf.keras.layers.Layer` to understand what will be passed and when.
    Note: tf.compat.v1.layers are usually not called with `training=None`,
    so the training arg to `forward_pass` might not feed through to them
    unless you pass it to their calls explicitly.

  Call signature of the forward pass:
    The semantics of the forward pass signature roughly match the standard
    Keras layer `call` signature, except that a `training` arg will *always*
    be passed, so your `forward_pass` must accept either.

  Limitations:
    * TF2 will not prune unused variable updates (or unused outputs). You may
      need to adjust your forward pass code to avoid computations or variable
      updates that you don't intend to use. (E.g. by adding a flag to the
      `forward_pass` call signature and branching on it).
    * Avoid Nesting variable creation in tf.function inside of `forward_pass`
      While the layer may safetely be used from inside a `tf.function`, using
      a function inside of `forward_pass` will break the variable scoping.
    * TBD: Nesting keras layers/models or other `VariableScopeWrapperLayer`s
      directly in `forward_pass` may not work correctly just yet.
      Support for this/instructions for how to do this is sill being worked on.

  Coming soon: A better guide, testing/verification guide.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Relies on keras layers tracking Modules
    self.tracker = VariableAndLossTracker()
    # May need to inspect func to see if it should pass a `training` arg or not

  def forward_pass(self, *args, **kwargs):
    raise NotImplementedError

  def call(self, *args, **kwargs):
    with self.tracker.scope():
      out = self.forward_pass(*args, **kwargs)
    if not self._eager_losses:
      # We have to record regularization losses in the call as if they
      # are activity losses.
      # So, don't double-count regularization losses if the layer is used
      # multiple times in a model
      for loss in self.tracker.get_regularization_losses().values():
        self.add_loss(loss)
    return out
