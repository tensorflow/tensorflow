# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""A class to store named variables and a scope operator to manage sharing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as collections_lib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import sys
import threading
import traceback

import six
from six import iteritems
from six.moves import xrange, zip  # pylint: disable=redefined-builtin

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "AUTO_REUSE", "VariableScope", "get_variable_scope", "get_variable",
    "get_local_variable", "variable_scope", "variable_op_scope",
    "no_regularizer", "VariableSynchronization", "VariableAggregation"
]

_api_usage_gauge = monitoring.BoolGauge(
    "/tensorflow/api/resource_variables",
    "Whether variable_scope.enable_resource_variables() is called.")


class _PartitionInfo(object):
  """Holds partition info used by initializer functions."""

  def __init__(self, full_shape, var_offset):
    """Constructor.

    Args:
      full_shape: Tuple or list of `int` indicating the full combined shape of
        the partitioned variables.
      var_offset: Tuple or list of `int` specifying offset of this partition
        with respect to the full variable for each dimension.

    Raises:
      TypeError: If `full_shape` or `var_offset` is not a sequence.
      ValueError: If `full_shape` or `var_offset` differ in length. If
        `var_offset` exceeds `full_shape` in any dimension.
    """
    if not isinstance(full_shape, collections_lib.Sequence) or isinstance(
        full_shape, six.string_types):
      raise TypeError(
          "`full_shape` must be a sequence (like tuple or list) instead of " +
          type(full_shape).__name__)

    if not isinstance(var_offset, collections_lib.Sequence) or isinstance(
        var_offset, six.string_types):
      raise TypeError(
          "`var_offset` must be a sequence (like tuple or list) instead of " +
          type(var_offset).__name__)

    if len(var_offset) != len(full_shape):
      raise ValueError(
          "Expected equal length, but `var_offset` is of length {} while "
          "full_shape is of length {}.".format(
              len(var_offset), len(full_shape)))

    for offset, shape in zip(var_offset, full_shape):
      if offset < 0 or offset >= shape:
        raise ValueError(
            "Expected 0 <= offset < shape but found offset={}, shape={} for "
            "var_offset={}, full_shape={}".format(offset, shape, var_offset,
                                                  full_shape))

    self._full_shape = full_shape
    self._var_offset = var_offset

  @property
  def full_shape(self):
    return self._full_shape

  @property
  def var_offset(self):
    return self._var_offset

  def single_offset(self, shape):
    """Returns the offset when the variable is partitioned in at most one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the offset in the dimension along which the variable is
       partitioned. Returns 0 if the variable is not being partitioned.

    Raises:
      ValueError: Depending on self.single_slice_dim().
    """

    single_slice_dim = self.single_slice_dim(shape)
    # If this variable is not being partitioned at all, single_slice_dim() could
    # return None.
    if single_slice_dim is None:
      return 0
    return self.var_offset[single_slice_dim]

  def single_slice_dim(self, shape):
    """Returns the slice dim when the variable is partitioned only in one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the dimension that the variable is partitioned in, or
      `None` if the variable doesn't seem to be partitioned at all.

    Raises:
      TypeError: If `shape` is not a sequence.
      ValueError: If `shape` is not the same length as `self.full_shape`. If
        the variable is partitioned in more than one dimension.
    """
    if not isinstance(shape, collections_lib.Sequence) or isinstance(
        shape, six.string_types):
      raise TypeError(
          "`shape` must be a sequence (like tuple or list) instead of " +
          type(shape).__name__)

    if len(shape) != len(self.full_shape):
      raise ValueError(
          "Expected equal length, but received shape={} of length {} while "
          "self.full_shape={} is of length {}.".format(shape, len(shape),
                                                       self.full_shape,
                                                       len(self.full_shape)))

    for i in xrange(len(shape)):
      if self.var_offset[i] + shape[i] > self.full_shape[i]:
        raise ValueError(
            "With self.var_offset={}, a partition of shape={} would exceed "
            "self.full_shape={} in dimension {}.".format(
                self.var_offset, shape, self.full_shape, i))

    slice_dim = None
    for i in xrange(len(shape)):
      if shape[i] == self.full_shape[i]:
        continue
      if slice_dim is not None:
        raise ValueError(
            "Cannot use single_slice_dim() with shape={} and "
            "self.full_shape={} since slice dim could be either dimension {} "
            "or {}.".format(shape, self.full_shape, i, slice_dim))
      slice_dim = i

    return slice_dim


class _ReuseMode(enum.Enum):
  """Mode for variable access within a variable scope."""

  # Indicates that variables are to be fetched if they already exist or
  # otherwise created.
  AUTO_REUSE = 1

  # TODO(alive): For TensorFlow 2.0, Deprecate True/False/None API in favor of
  #              enum values.
  # REUSE_FALSE = 2
  # REUSE_TRUE = 3


# TODO(apassos) remove these forwarding symbols.
VariableSynchronization = variables.VariableSynchronization  # pylint: disable=invalid-name
VariableAggregation = variables.VariableAggregation  # pylint: disable=invalid-name

AUTO_REUSE = _ReuseMode.AUTO_REUSE
tf_export(v1=["AUTO_REUSE"]).export_constant(__name__, "AUTO_REUSE")
AUTO_REUSE.__doc__ = """
When passed in as the value for the `reuse` flag, AUTO_REUSE indicates that
get_variable() should create the requested variable if it doesn't exist or, if
it does exist, simply return it.
"""

_DEFAULT_USE_RESOURCE = tf2.enabled()


@tf_export(v1=["enable_resource_variables"])
def enable_resource_variables():
  """Creates resource variables by default.

  Resource variables are improved versions of TensorFlow variables with a
  well-defined memory model. Accessing a resource variable reads its value, and
  all ops which access a specific read value of the variable are guaranteed to
  see the same value for that tensor. Writes which happen after a read (by
  having a control or data dependency on the read) are guaranteed not to affect
  the value of the read tensor, and similarly writes which happen before a read
  are guaranteed to affect the value. No guarantees are made about unordered
  read/write pairs.

  Calling tf.enable_resource_variables() lets you opt-in to this TensorFlow 2.0
  feature.
  """
  global _DEFAULT_USE_RESOURCE
  _DEFAULT_USE_RESOURCE = True
  _api_usage_gauge.get_cell().set(True)


@tf_export(v1=["resource_variables_enabled"])
def resource_variables_enabled():
  """Returns `True` if resource variables are enabled.

  Resource variables are improved versions of TensorFlow variables with a
  well-defined memory model. Accessing a resource variable reads its value, and
  all ops which access a specific read value of the variable are guaranteed to
  see the same value for that tensor. Writes which happen after a read (by
  having a control or data dependency on the read) are guaranteed not to affect
  the value of the read tensor, and similarly writes which happen before a read
  are guaranteed to affect the value. No guarantees are made about unordered
  read/write pairs.

  Calling tf.enable_resource_variables() lets you opt-in to this TensorFlow 2.0
  feature.
  """
  global _DEFAULT_USE_RESOURCE
  return _DEFAULT_USE_RESOURCE


@deprecation.deprecated(
    None, "non-resource variables are not supported in the long term")
@tf_export(v1=["disable_resource_variables"])
def disable_resource_variables():
  """Opts out of resource variables.

  If your code needs tf.disable_resource_variables() to be called to work
  properly please file a bug.
  """
  global _DEFAULT_USE_RESOURCE
  _DEFAULT_USE_RESOURCE = False
  _api_usage_gauge.get_cell().set(False)


class _VariableStore(object):
  """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """

  def __init__(self):
    """Create a variable store."""
    self._vars = {}  # A dictionary of the stored TensorFlow variables.
    self._partitioned_vars = {}  # A dict of the stored PartitionedVariables.
    self._store_eager_variables = False

  def get_variable(self,
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
                   synchronization=VariableSynchronization.AUTO,
                   aggregation=VariableAggregation.NONE):
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
      if not self._store_eager_variables and reuse:
        raise RuntimeError(
            "When eager execution is enabled variable reuse is only supported"
            " when an EagerVariableStore is active. See the documentation on"
            " EagerVariableStore for example usage.")
      if self._store_eager_variables:
        reuse = AUTO_REUSE

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
        collections=None,
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        constraint=None,
        synchronization=VariableSynchronization.AUTO,
        aggregation=VariableAggregation.NONE):
      is_scalar = (
          shape is not None and isinstance(shape, collections_lib.Sequence) and
          not shape)
      # Partitioned variable case
      if partitioner is not None and not is_scalar:
        if not callable(partitioner):
          raise ValueError("Partitioner must be callable, but received: %s" %
                           partitioner)
        with ops.name_scope(None):
          return self._get_partitioned_variable(
              name=name,
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

      # Special case for partitioned variable to allow reuse without having to
      # specify partitioner.
      if (reuse is True and partitioner is None
          and name in self._partitioned_vars):
        return self._get_partitioned_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            partitioner=None,
            validate_shape=validate_shape,
            use_resource=use_resource,
            constraint=constraint,
            synchronization=synchronization,
            aggregation=aggregation)

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
          collections=collections,
          caching_device=caching_device,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
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
      if ("constraint" in function_utils.fn_args(custom_getter) or
          function_utils.has_kwargs(custom_getter)):
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

  def _get_partitioned_variable(self,
                                name,
                                partitioner,
                                shape=None,
                                dtype=dtypes.float32,
                                initializer=None,
                                regularizer=None,
                                reuse=None,
                                trainable=None,
                                collections=None,
                                caching_device=None,
                                validate_shape=True,
                                use_resource=None,
                                constraint=None,
                                synchronization=VariableSynchronization.AUTO,
                                aggregation=VariableAggregation.NONE):
    """Gets or creates a sharded variable list with these parameters.

    The `partitioner` must be a callable that accepts a fully defined
    `TensorShape` and returns a sequence of integers (the `partitions`).
    These integers describe how to partition the given sharded `Variable`
    along the given dimension.  That is, `partitions[1] = 3` means split
    the `Variable` into 3 shards along dimension 1.  Currently, sharding along
    only one axis is supported.

    If the list of variables with the given name (prefix) is already stored,
    we return the stored variables. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If the initializer is a callable, then it will be called for each
    shard.  Otherwise the initializer should match the shape of the entire
    sharded Variable, and it will be sliced accordingly for each shard.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: the name of the new or existing sharded variable.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and `dtype` of the Variable to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      shape: shape of the new or existing sharded variable.
      dtype: type of the new or existing sharded variable (defaults to
        `DT_FLOAT`).
      initializer: initializer for the sharded variable.
      regularizer: a (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      collections: List of graph collections keys to add the Variable to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates an
        experimental ResourceVariable which has well-defined semantics. Defaults
        to False (will later change to True).
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
      A `PartitionedVariable` object.

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        when violating reuse during variable creation, or if an existing
        sharded variable exists for the given name but with different sharding.
    """
    initializing_from_value = initializer is not None and isinstance(
        initializer, ops.Tensor)
    if name in self._vars:
      raise ValueError(
          "A partitioner was provided, but an unpartitioned version of the "
          "variable was found: %s.  Perhaps a variable of the same name was "
          "already created without partitioning?" % name)

    shape = tensor_shape.as_shape(shape)
    if initializing_from_value:
      shape = shape.merge_with(initializer.get_shape())

    partitions = None
    if not reuse or partitioner:
      partitions = _call_partitioner(partitioner, shape, dtype)

    if name in self._partitioned_vars:
      if reuse is False:
        raise ValueError(
            "Partitioned variable with name %s already exists. Did you mean to "
            "set reuse=True or reuse=tf.AUTO_REUSE in VarScope?" % name)

      existing_var = self._partitioned_vars[name]
      if not shape.is_compatible_with(existing_var.get_shape()):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified shape %s "
            "and found shape %s." % (name, shape, existing_var.get_shape()))
      if not dtype.is_compatible_with(existing_var.dtype):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified dtype %s "
            "and found dtype %s." % (name, dtype.name, existing_var.dtype.name))

      # pylint: disable=protected-access
      if (partitions is not None and
          existing_var._get_partitions() != partitions):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified partitions "
            "%s and found partitions %s." %
            (name, partitions, existing_var._get_partitions()))
      # pylint: enable=protected-access

      return existing_var

    if reuse is True:
      raise ValueError("PartitionedVariable %s does not exist, or was not "
                       "created with tf.get_variable(). Did you mean to set "
                       "reuse=False or reuse=tf.AUTO_REUSE in VarScope?" % name)

    slice_dim, num_slices = _get_slice_dim_and_num_slices(partitions)

    if "%s/part_0" % name in self._vars:
      if "%s/part_%d" % (name, num_slices - 1) not in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but %s/part_%d was not." %
            (num_slices, name, name, num_slices - 1))
      if "%s/part_%d" % (name, num_slices) in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but so was the extra shard %s/part_%d." %
            (num_slices, name, name, num_slices))

    vs = []
    for i, (var_offset, var_shape) in enumerate(
        _iter_slices(shape.as_list(), num_slices, slice_dim)):
      partition_info = _PartitionInfo(
          full_shape=shape.as_list(), var_offset=var_offset)
      var_full_name = "%s/part_%d" % (name, i)
      with ops.name_scope(var_full_name + "/PartitionedInitializer"):
        # Create the tensor to initialize the variable with default value.
        if initializer is None:
          init, initializing_from_value = self._get_default_initializer(
              name=name, shape=shape, dtype=dtype)
          if initializing_from_value:
            init_shape = None
          else:
            init_shape = var_shape
        elif callable(initializer):
          init = initializer
          init_shape = var_shape
        elif isinstance(initializer, ops.Tensor):
          init = array_ops.slice(initializer, var_offset, var_shape)
          # Use the dtype of the given tensor.
          dtype = init.dtype.base_dtype
          init_shape = None
        else:
          init = ops.convert_to_tensor(initializer, dtype=dtype)
          init = array_ops.slice(init, var_offset, var_shape)
          init_shape = None

      with ops.name_scope(None):
        var = self._get_single_variable(
            name=var_full_name,
            shape=init_shape,
            dtype=dtype,
            initializer=init,
            partition_info=partition_info,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            validate_shape=validate_shape,
            use_resource=use_resource,
            constraint=constraint,
            synchronization=synchronization,
            aggregation=aggregation)

      # pylint: disable=protected-access
      var._set_save_slice_info(
          variables.Variable.SaveSliceInfo(name, shape.as_list(), var_offset,
                                           var_shape))
      vs.append(var)
      # pylint: enable=protected-access

    partitioned_var = variables.PartitionedVariable(
        name=name,
        shape=shape,
        dtype=dtype,
        variable_list=vs,
        partitions=partitions)
    if not context.executing_eagerly() or self._store_eager_variables:
      self._partitioned_vars[name] = partitioned_var
    return partitioned_var

  def _get_single_variable(self,
                           name,
                           shape=None,
                           dtype=dtypes.float32,
                           initializer=None,
                           regularizer=None,
                           partition_info=None,
                           reuse=None,
                           trainable=None,
                           collections=None,
                           caching_device=None,
                           validate_shape=True,
                           use_resource=None,
                           constraint=None,
                           synchronization=VariableSynchronization.AUTO,
                           aggregation=VariableAggregation.NONE):
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
      collections: see get_variable.
      caching_device: see get_variable.
      validate_shape: see get_variable.
      use_resource: see get_variable.
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
    shape = tensor_shape.as_shape(shape)

    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if reuse is False:
        var = self._vars[name]
        err_msg = ("Variable %s already exists, disallowed."
                   " Did you mean to set reuse=True or "
                   "reuse=tf.AUTO_REUSE in VarScope?" % name)
        # ResourceVariables don't have an op associated with so no traceback
        if isinstance(var, resource_variable_ops.ResourceVariable):
          raise ValueError(err_msg)
        tb = var.op.traceback[::-1]
        # Throw away internal tf entries and only take a few lines. In some
        # cases the traceback can be longer (e.g. if someone uses factory
        # functions to create variables) so we take more than needed in the
        # default case.
        tb = [x for x in tb if "tensorflow/python" not in x[0]][:5]
        raise ValueError("%s Originally defined at:\n\n%s" %
                         (err_msg, "".join(traceback.format_list(tb))))
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
    if reuse is True:
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
        if shape is not None and shape.is_fully_defined():
          init_val = lambda: initializer(  # pylint: disable=g-long-lambda
              shape.as_list(),
              dtype=dtype,
              partition_info=partition_info)
          variable_dtype = dtype.base_dtype
        elif len(tf_inspect.getargspec(initializer).args) == len(
            tf_inspect.getargspec(initializer).defaults or []):
          init_val = initializer
          variable_dtype = None
        else:
          raise ValueError("The initializer passed is not valid. It should "
                           "be a callable with no arguments and the "
                           "shape should not be provided or an instance of "
                           "`tf.keras.initializers.*' and `shape` should be "
                           "fully defined.")

    # Create the variable.
    if use_resource is None:
      # Set the default value if unspecified.
      use_resource = _DEFAULT_USE_RESOURCE
    v = variables.VariableV1(
        initial_value=init_val,
        name=name,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        dtype=variable_dtype,
        validate_shape=validate_shape,
        constraint=constraint,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation)
    if context.executing_eagerly() and self._store_eager_variables:
      if collections:
        ops.add_to_collections(collections, v)
      else:
        ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, v)
      if trainable:
        ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, v)

    if not context.executing_eagerly() or self._store_eager_variables:
      # In eager mode we do not want to keep default references to Variable
      # objects as this will prevent their memory from being released.
      self._vars[name] = v
    logging.vlog(1, "Created variable %s with shape %s and init %s", v.name,
                 format(shape), initializer)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      with ops.colocate_with(v):
        with ops.name_scope(name + "/Regularizer/"):
          with ops.init_scope():
            loss = regularizer(v)
        if loss is not None:
          if context.executing_eagerly():
            v_name = "v_%s" % type(v)
            loss_name = "loss_%s" % type(loss)
          else:
            v_name = v.name
            loss_name = loss.name
          logging.vlog(
              1, "Applied regularizer to %s and added the result %s "
              "to REGULARIZATION_LOSSES.", v_name, loss_name)
          ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)
    return v

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


# To stop regularization, use this regularizer
@tf_export(v1=["no_regularizer"])
def no_regularizer(_):
  """Use this function to prevent regularization of variables."""
  return None


# TODO(alive): support caching devices and partitioned variables in Eager mode.
@tf_export(v1=["VariableScope"])
class VariableScope(object):
  """Variable scope object to carry defaults to provide to `get_variable`.

  Many of the arguments we need for `get_variable` in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    regularizer: default regularizer passed to get_variable.
    reuse: Boolean, None, or tf.compat.v1.AUTO_REUSE, setting the reuse in
      get_variable. When eager execution is enabled this argument is always
      forced to be False.
    caching_device: string, callable, or None: the caching device passed to
      get_variable.
    partitioner: callable or `None`: the partitioner passed to `get_variable`.
    custom_getter: default custom getter passed to get_variable.
    name_scope: The name passed to `tf.name_scope`.
    dtype: default type passed to get_variable (defaults to DT_FLOAT).
    use_resource: if False, create a normal Variable; if True create an
      experimental ResourceVariable with well-defined semantics. Defaults to
      False (will later change to True). When eager execution is enabled this
      argument is always forced to be True.
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
  """

  def __init__(self,
               reuse,
               name="",
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               name_scope="",
               dtype=dtypes.float32,
               use_resource=None,
               constraint=None):
    """Creates a new VariableScope with the given properties."""
    self._name = name
    self._initializer = initializer
    self._regularizer = regularizer
    self._reuse = reuse
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._name_scope = name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    if context.executing_eagerly():
      if self._caching_device is not None:
        raise NotImplementedError("Caching devices is not yet supported "
                                  "when eager execution is enabled.")
      self._reuse = AUTO_REUSE
      self._use_resource = True

  @property
  def name(self):
    return self._name

  @property
  def original_name_scope(self):
    return self._name_scope

  @property
  def reuse(self):
    return self._reuse

  @property
  def initializer(self):
    return self._initializer

  @property
  def dtype(self):
    return self._dtype

  @property
  def use_resource(self):
    return self._use_resource

  @property
  def regularizer(self):
    return self._regularizer

  @property
  def caching_device(self):
    return self._caching_device

  @property
  def partitioner(self):
    return self._partitioner

  @property
  def custom_getter(self):
    return self._custom_getter

  @property
  def constraint(self):
    return self._constraint

  def reuse_variables(self):
    """Reuse variables in this scope."""
    self._reuse = True

  def set_initializer(self, initializer):
    """Set initializer for this scope."""
    self._initializer = initializer

  def set_dtype(self, dtype):
    """Set data type for this scope."""
    self._dtype = dtype

  def set_use_resource(self, use_resource):
    """Sets whether to use ResourceVariables for this scope."""
    if context.executing_eagerly() and not use_resource:
      raise ValueError("When eager execution is enabled, "
                       "use_resource cannot be set to false.")
    self._use_resource = use_resource

  def set_regularizer(self, regularizer):
    """Set regularizer for this scope."""
    self._regularizer = regularizer

  def set_caching_device(self, caching_device):
    """Set caching_device for this scope."""
    if context.executing_eagerly():
      raise NotImplementedError("Caching devices are not yet supported "
                                "when eager execution is enabled.")
    self._caching_device = caching_device

  def set_partitioner(self, partitioner):
    """Set partitioner for this scope."""
    self._partitioner = partitioner

  def set_custom_getter(self, custom_getter):
    """Set custom getter for this scope."""
    self._custom_getter = custom_getter

  def get_collection(self, name):
    """Get this scope's variables."""
    scope = self._name + "/" if self._name else ""
    return ops.get_collection(name, scope)

  def trainable_variables(self):
    """Get this scope's trainable variables."""
    return self.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)

  def global_variables(self):
    """Get this scope's global variables."""
    return self.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

  def local_variables(self):
    """Get this scope's local variables."""
    return self.get_collection(ops.GraphKeys.LOCAL_VARIABLES)

  def get_variable(self,
                   var_store,
                   name,
                   shape=None,
                   dtype=None,
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
                   synchronization=VariableSynchronization.AUTO,
                   aggregation=VariableAggregation.NONE):
    """Gets an existing variable with this name or create a new one."""
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if custom_getter is None:
      custom_getter = self._custom_getter
    if context.executing_eagerly():
      reuse = False
      use_resource = True
    else:
      if reuse is None:
        reuse = self._reuse
      if use_resource is None:
        use_resource = self._use_resource

    full_name = self.name + "/" + name if self.name else name
    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # Check that `initializer` dtype and `dtype` are consistent before
      # replacing them with defaults.
      if (dtype is not None and initializer is not None and
          not callable(initializer)):
        init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
        if init_dtype != dtype:
          raise ValueError("Initializer type '%s' and explicit dtype '%s' "
                           "don't match." % (init_dtype, dtype))
      if initializer is None:
        initializer = self._initializer
      if constraint is None:
        constraint = self._constraint
      if dtype is None:
        dtype = self._dtype
      return var_store.get_variable(
          full_name,
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
          custom_getter=custom_getter,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

  def _get_partitioned_variable(self,
                                var_store,
                                name,
                                shape=None,
                                dtype=None,
                                initializer=None,
                                regularizer=None,
                                trainable=None,
                                collections=None,
                                caching_device=None,
                                partitioner=None,
                                validate_shape=True,
                                use_resource=None,
                                constraint=None,
                                synchronization=VariableSynchronization.AUTO,
                                aggregation=VariableAggregation.NONE):
    """Gets an existing variable with this name or create a new one."""
    if initializer is None:
      initializer = self._initializer
    if regularizer is None:
      regularizer = self._regularizer
    if constraint is None:
      constraint = self._constraint
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if dtype is None:
      dtype = self._dtype
    if use_resource is None:
      use_resource = self._use_resource

    if self._custom_getter is not None:
      raise ValueError(
          "Private access to _get_partitioned_variable is not allowed when "
          "a custom getter is set.  Current custom getter: %s.  "
          "It is likely that you're using create_partitioned_variables.  "
          "If so, consider instead using get_variable with a non-empty "
          "partitioner parameter instead." % self._custom_getter)

    if partitioner is None:
      raise ValueError("No partitioner was specified")

    # This allows the variable scope name to be used as the variable name if
    # this function is invoked with an empty name arg, for backward
    # compatibility with create_partitioned_variables().
    full_name_list = []
    if self.name:
      full_name_list.append(self.name)
    if name:
      full_name_list.append(name)
    full_name = "/".join(full_name_list)

    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # pylint: disable=protected-access
      return var_store._get_partitioned_variable(
          full_name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=self.reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)
      # pylint: enable=protected-access


_VARSTORE_KEY = ("__variable_store",)
_VARSCOPESTORE_KEY = ("__varscope",)


class _VariableScopeStore(threading.local):
  """A thread local store for the current variable scope and scope counts."""

  def __init__(self):
    super(_VariableScopeStore, self).__init__()
    self.current_scope = VariableScope(False)
    self.variable_scopes_count = {}

  def open_variable_scope(self, scope_name):
    if scope_name in self.variable_scopes_count:
      self.variable_scopes_count[scope_name] += 1
    else:
      self.variable_scopes_count[scope_name] = 1

  def close_variable_subscopes(self, scope_name):
    for k in list(self.variable_scopes_count.keys()):
      if scope_name is None or k.startswith(scope_name + "/"):
        self.variable_scopes_count[k] = 0

  def variable_scope_count(self, scope_name):
    return self.variable_scopes_count.get(scope_name, 0)


def get_variable_scope_store():
  """Returns the variable scope store for current thread."""
  scope_store = ops.get_collection(_VARSCOPESTORE_KEY)

  if not scope_store:
    scope_store = _VariableScopeStore()
    ops.add_to_collection(_VARSCOPESTORE_KEY, scope_store)
  else:
    scope_store = scope_store[0]

  return scope_store


@tf_export(v1=["get_variable_scope"])
def get_variable_scope():
  """Returns the current variable scope."""
  return get_variable_scope_store().current_scope


def _get_default_variable_store():
  store = ops.get_collection(_VARSTORE_KEY)
  if store:
    return store[0]
  store = _VariableStore()
  ops.add_to_collection(_VARSTORE_KEY, store)
  return store


@tf_contextlib.contextmanager
def with_variable_store(store):
  store_collection = ops.get_collection_ref(_VARSTORE_KEY)
  old = list(store_collection)
  store_collection[:] = [store]
  try:
    yield
  finally:
    store_collection[:] = old


class EagerVariableStore(object):
  """Wrapper allowing functional layers to be used with eager execution.

  When eager execution is enabled Variables get deleted when they go out of
  scope, and are not stored in global collections by default. A lot of code
  (mostly the functional layers in tf.layers) assumes that variables are kept in
  a global list.

  EagerVariableStore can be used in conjunction with this code to make it
  eager-friendly. For example, to create a dense layer, use:

  ```
    container = tfe.EagerVariableStore()
    for input in dataset_iterator:
      with container.as_default():
        x = tf.compat.v1.layers.dense(input, name="l1")
    print(container.variables)  # Should print the variables used in the layer.
  ```
  """

  def __init__(self, store=None):
    if store is not None:
      if not store._store_eager_variables:  # pylint: disable=protected-access
        raise ValueError("Cannot construct EagerVariableStore from a "
                         "VariableStore object that does not hold eager "
                         "variables.")
      self._store = store
    else:
      self._store = _VariableStore()
    self._store._store_eager_variables = True  # pylint: disable=protected-access

  def as_default(self):
    return with_variable_store(self._store)

  def variables(self):
    return sorted(self._store._vars.values(), key=lambda x: x.name)  # pylint: disable=protected-access

  def trainable_variables(self):
    # pylint: disable=protected-access
    return sorted([x for x in self._store._vars.values() if x.trainable],
                  key=lambda x: x.name)
    # pylint: enable=protected-access

  def non_trainable_variables(self):
    # pylint: disable=protected-access
    return sorted([x for x in self._store._vars.values() if not x.trainable],
                  key=lambda x: x.name)
    # pylint: enable=protected-access

  def copy(self):
    """Copy this variable store and all of its contents.

    Variables contained in this store will be copied over to the new variable
    store, meaning that they can be modified without affecting the variables in
    this store.

    Returns:
      A new EagerVariableStore instance containing copied variables.
    """
    # pylint: disable=protected-access
    new_store = EagerVariableStore()
    for key, var in iteritems(self._store._vars):
      # Strip device out of variable name.
      try:
        index = var.name.index(":")
      except ValueError:
        stripped_var_name = var.name
      else:
        stripped_var_name = var.name[:index]

      # Create new variable with same value, name, and "trainable" flag.
      new_var = resource_variable_ops.ResourceVariable(
          var.read_value(), name=stripped_var_name, trainable=var.trainable)
      new_store._store._vars[key] = new_var
    return new_store
    # pylint: enable=protected-access


# The argument list for get_variable must match arguments to get_local_variable.
# So, if you are updating the arguments, also update arguments to
# get_local_variable below.
@tf_export(v1=["get_variable"])
def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None,
                 constraint=None,
                 synchronization=VariableSynchronization.AUTO,
                 aggregation=VariableAggregation.NONE):
  return get_variable_scope().get_variable(
      _get_default_variable_store(),
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      custom_getter=custom_getter,
      constraint=constraint,
      synchronization=synchronization,
      aggregation=aggregation)


get_variable_or_local_docstring = ("""%s

%sThis function prefixes the name with the current variable scope
and performs reuse checks. See the
[Variable Scope How To](https://tensorflow.org/guide/variables)
for an extensive description of how reusing works. Here is a basic example:

```python
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2
```

If initializer is `None` (the default), the default initializer passed in
the variable scope will be used. If that one is `None` too, a
`glorot_uniform_initializer` will be used. The initializer can also be
a Tensor, in which case the variable is initialized to this value and shape.

Similarly, if the regularizer is `None` (the default), the default regularizer
passed in the variable scope will be used (if that is `None` too,
then by default no regularization is performed).

If a partitioner is provided, a `PartitionedVariable` is returned.
Accessing this object as a `Tensor` returns the shards concatenated along
the partition axis.

Some useful partitioners are available.  See, e.g.,
`variable_axis_size_partitioner` and `min_max_variable_partitioner`.

Args:
  name: The name of the new or existing variable.
  shape: Shape of the new or existing variable.
  dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
  initializer: Initializer for the variable if one is created. Can either be
    an initializer object or a Tensor. If it's a Tensor, its shape must be known
    unless validate_shape is False.
  regularizer: A (Tensor -> Tensor or None) function; the result of
    applying it on a newly created variable will be added to the collection
    `tf.GraphKeys.REGULARIZATION_LOSSES` and can be used for regularization.
  %scollections: List of graph collections keys to add the Variable to.
    Defaults to `[%s]` (see `tf.Variable`).
  caching_device: Optional device string or function describing where the
    Variable should be cached for reading.  Defaults to the Variable's
    device.  If not `None`, caches on another device.  Typical use is to
    cache on the device where the Ops using the Variable reside, to
    deduplicate copying through `Switch` and other conditional statements.
  partitioner: Optional callable that accepts a fully defined `TensorShape`
    and `dtype` of the Variable to be created, and returns a list of
    partitions for each axis (currently only one axis can be partitioned).
  validate_shape: If False, allows the variable to be initialized with a
      value of unknown shape. If True, the default, the shape of initial_value
      must be known. For this to be used the initializer must be a Tensor and
      not an initializer object.
  use_resource: If False, creates a regular Variable. If true, creates an
    experimental ResourceVariable instead with well-defined semantics.
    Defaults to False (will later change to True). When eager execution is
    enabled this argument is always forced to be True.
  custom_getter: Callable that takes as a first argument the true getter, and
    allows overwriting the internal get_variable method.
    The signature of `custom_getter` should match that of this method,
    but the most future-proof version will allow for changes:
    `def custom_getter(getter, *args, **kwargs)`.  Direct access to
    all `get_variable` parameters is also allowed:
    `def custom_getter(getter, name, *args, **kwargs)`.  A simple identity
    custom getter that simply creates variables with modified names is:
    ```python
    def custom_getter(getter, name, *args, **kwargs):
      return getter(name + '_suffix', *args, **kwargs)
    ```
  constraint: An optional projection function to be applied to the variable
    after being updated by an `Optimizer` (e.g. used to implement norm
    constraints or value constraints for layer weights). The function must
    take as input the unprojected Tensor representing the value of the
    variable and return the Tensor for the projected value
    (which must have the same shape). Constraints are not safe to
    use when doing asynchronous distributed training.
  synchronization: Indicates when a distributed a variable will be
    aggregated. Accepted values are constants defined in the class
    `tf.VariableSynchronization`. By default the synchronization is set to
    `AUTO` and the current `DistributionStrategy` chooses
    when to synchronize.
  aggregation: Indicates how a distributed variable will be aggregated.
    Accepted values are constants defined in the class
    `tf.VariableAggregation`.

Returns:
  The created or existing `Variable` (or `PartitionedVariable`, if a
  partitioner was used).

Raises:
  ValueError: when creating a new variable and shape is not declared,
    when violating reuse during variable creation, or when `initializer` dtype
    and `dtype` don't match. Reuse is set inside `variable_scope`.
""")
get_variable.__doc__ = get_variable_or_local_docstring % (
    "Gets an existing variable with these parameters or create a new one.", "",
    "trainable: If `True` also add the variable to the graph collection\n"
    "    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n  ",
    "GraphKeys.GLOBAL_VARIABLES")


# The argument list for get_local_variable must match arguments to get_variable.
# So, if you are updating the arguments, also update arguments to get_variable.
@tf_export(v1=["get_local_variable"])
def get_local_variable(  # pylint: disable=missing-docstring
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=False,  # pylint: disable=unused-argument
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    synchronization=VariableSynchronization.AUTO,
    aggregation=VariableAggregation.NONE):
  if collections:
    collections += [ops.GraphKeys.LOCAL_VARIABLES]
  else:
    collections = [ops.GraphKeys.LOCAL_VARIABLES]
  return get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=False,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      synchronization=synchronization,
      aggregation=aggregation,
      custom_getter=custom_getter,
      constraint=constraint)


get_local_variable.__doc__ = get_variable_or_local_docstring % (
    "Gets an existing *local* variable or creates a new one.",
    "Behavior is the same as in `get_variable`, except that variables are\n"
    "added to the `LOCAL_VARIABLES` collection and `trainable` is set to\n"
    "`False`.\n", "", "GraphKeys.LOCAL_VARIABLES")


def _get_partitioned_variable(name,
                              shape=None,
                              dtype=None,
                              initializer=None,
                              regularizer=None,
                              trainable=True,
                              collections=None,
                              caching_device=None,
                              partitioner=None,
                              validate_shape=True,
                              use_resource=None,
                              constraint=None,
                              synchronization=VariableSynchronization.AUTO,
                              aggregation=VariableAggregation.NONE):
  """Gets or creates a sharded variable list with these parameters.

  The `partitioner` must be a callable that accepts a fully defined
  `TensorShape` and returns a sequence of integers (the `partitions`).
  These integers describe how to partition the given sharded `Variable`
  along the given dimension.  That is, `partitions[1] = 3` means split
  the `Variable` into 3 shards along dimension 1.  Currently, sharding along
  only one axis is supported.

  If the list of variables with the given name (prefix) is already stored,
  we return the stored variables. Otherwise, we create a new one.

  If initializer is `None` (the default), the default initializer passed in
  the constructor is used. If that one is `None` too, we use a new
  `glorot_uniform_initializer`. If initializer is a Tensor, we use
  it as a value and derive the shape from the initializer.

  If the initializer is a callable, then it will be called for each
  shard.  Otherwise the initializer should match the shape of the entire
  sharded Variable, and it will be sliced accordingly for each shard.

  Some useful partitioners are available.  See, e.g.,
  `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: Initializer for the variable if one is created.
    regularizer: A (Tensor -> Tensor or None) function; the result of applying
      it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: List of graph collections keys to add the Variable to. Defaults
      to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
    caching_device: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's device.
      If not `None`, caches on another device.  Typical use is to cache on the
      device where the Ops using the Variable reside, to deduplicate copying
      through `Switch` and other conditional statements.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and `dtype` of the Variable to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    validate_shape: If False, allows the variable to be initialized with a value
      of unknown shape. If True, the default, the shape of initial_value must be
      known.
    use_resource: If False, creates a regular Variable. If True, creates an
      experimental ResourceVariable instead which has well-defined semantics.
      Defaults to False (will later change to True).
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.

  Returns:
    A tuple `(shards, partitions)` where `shards` is the list of `Variable`
    shards and `partitions` is the output of the partitioner on the input
    shape.

  Raises:
    ValueError: when creating a new variable and shape is not declared,
      or when violating reuse during variable creation. Reuse is set inside
      `variable_scope`.
  """
  # pylint: disable=protected-access
  scope = get_variable_scope()
  if scope.custom_getter is not None:
    raise ValueError(
        "Private access to _get_partitioned_variable is not allowed when "
        "a custom getter is set.  Current custom getter: %s.  "
        "It is likely that you're using create_partitioned_variables.  "
        "If so, consider instead using get_variable with a non-empty "
        "partitioner parameter instead." % scope.custom_getter)
  return scope._get_partitioned_variable(
      _get_default_variable_store(),
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      constraint=constraint,
      synchronization=synchronization,
      aggregation=aggregation)
  # pylint: enable=protected-access


# Named like a function for compatibility with the previous
# @tf_contextlib.contextmanager definition.
class _pure_variable_scope(object):  # pylint: disable=invalid-name
  """A context for the variable_scope, see `variable_scope` for docs."""

  def __init__(self,
               name_or_scope,
               reuse=None,
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               old_name_scope=None,
               dtype=dtypes.float32,
               use_resource=None,
               constraint=None):
    """Creates a context for the variable_scope, see `variable_scope` for docs.

    Note: this does not create a name scope.

    Args:
      name_or_scope: `string` or `VariableScope`: the scope to open.
      reuse: `True` or None, or tf.compat.v1.AUTO_REUSE; if `None`, we inherit
        the parent scope's reuse flag.
      initializer: default initializer for variables within this scope.
      regularizer: default regularizer for variables within this scope.
      caching_device: default caching device for variables within this scope.
      partitioner: default partitioner for variables within this scope.
      custom_getter: default custom getter for variables within this scope.
      old_name_scope: the original name scope when re-entering a variable scope.
      dtype: type of the variables within this scope (defaults to `DT_FLOAT`).
      use_resource: If False, variables in this scope will be regular Variables.
        If True, experimental ResourceVariables will be creates instead, with
        well-defined semantics. Defaults to False (will later change to True).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
    """
    self._name_or_scope = name_or_scope
    self._reuse = reuse
    self._initializer = initializer
    self._regularizer = regularizer
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._old_name_scope = old_name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    self._var_store = _get_default_variable_store()
    self._var_scope_store = get_variable_scope_store()
    self._last_variable_scope_object = None
    if isinstance(self._name_or_scope, VariableScope):
      self._new_name = self._name_or_scope.name
      name_scope = self._name_or_scope._name_scope  # pylint: disable=protected-access
      # Handler for the case when we jump to a shared scope.  We create a new
      #   VariableScope (self._var_scope_object) that contains a copy of the
      #   provided shared scope, possibly with changed reuse and initializer, if
      #   the user requested this.
      variable_scope_object = VariableScope(
          self._name_or_scope.reuse if not self._reuse else self._reuse,
          name=self._new_name,
          initializer=self._name_or_scope.initializer,
          regularizer=self._name_or_scope.regularizer,
          caching_device=self._name_or_scope.caching_device,
          partitioner=self._name_or_scope.partitioner,
          dtype=self._name_or_scope.dtype,
          custom_getter=self._name_or_scope.custom_getter,
          name_scope=name_scope,
          use_resource=self._name_or_scope.use_resource,
          constraint=self._constraint)
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            _maybe_wrap_custom_getter(self._custom_getter,
                                      self._name_or_scope.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._cached_variable_scope_object = variable_scope_object

  def __enter__(self):
    """Begins the scope block.

    Returns:
      A VariableScope.
    Raises:
      ValueError: when trying to reuse within a create scope, or create within
        a reuse scope, or if reuse is not `None` or `True`.
      TypeError: when the types of some arguments are not appropriate.
    """
    self._old = self._var_scope_store.current_scope
    if isinstance(self._name_or_scope, VariableScope):
      self._var_scope_store.open_variable_scope(self._new_name)
      self._old_subscopes = copy.copy(
          self._var_scope_store.variable_scopes_count)
      variable_scope_object = self._cached_variable_scope_object
    else:
      # Handler for the case when we just prolong current variable scope.
      #   VariableScope with name extended by the provided one, and inherited
      #   reuse and initializer (except if the user provided values to set).
      self._new_name = (
          self._old.name + "/" +
          self._name_or_scope if self._old.name else self._name_or_scope)
      self._reuse = (self._reuse or
                     self._old.reuse)  # Re-using is inherited by sub-scopes.
      if self._old_name_scope is None:
        name_scope = self._name_or_scope
      else:
        name_scope = self._old_name_scope
      variable_scope_object = VariableScope(
          self._reuse,
          name=self._new_name,
          initializer=self._old.initializer,
          regularizer=self._old.regularizer,
          caching_device=self._old.caching_device,
          partitioner=self._old.partitioner,
          dtype=self._old.dtype,
          use_resource=self._old.use_resource,
          custom_getter=self._old.custom_getter,
          name_scope=name_scope,
          constraint=self._constraint)
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            _maybe_wrap_custom_getter(self._custom_getter,
                                      self._old.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._var_scope_store.open_variable_scope(self._new_name)
    self._var_scope_store.current_scope = variable_scope_object
    self._last_variable_scope_object = variable_scope_object
    return variable_scope_object

  def __exit__(self, type_arg, value_arg, traceback_arg):
    if (self._var_scope_store.current_scope is
        not self._last_variable_scope_object):
      raise RuntimeError("Improper nesting of variable_scope.")
    # If jumping out from a non-prolonged scope, restore counts.
    if isinstance(self._name_or_scope, VariableScope):
      self._var_scope_store.variable_scopes_count = self._old_subscopes
    else:
      self._var_scope_store.close_variable_subscopes(self._new_name)
    self._var_scope_store.current_scope = self._old


def _maybe_wrap_custom_getter(custom_getter, old_getter):
  """Wrap a call to a custom_getter to use the old_getter internally."""
  if old_getter is None:
    return custom_getter

  # The new custom_getter should call the old one
  def wrapped_custom_getter(getter, *args, **kwargs):
    # Call:
    #  custom_getter(
    #    lambda: old_getter(true_getter, ...), *args, **kwargs)
    # which means custom_getter will call old_getter, which
    # will call the true_getter, perform any intermediate
    # processing, and return the results to the current
    # getter, which will also perform additional processing.
    return custom_getter(functools.partial(old_getter, getter), *args, **kwargs)

  return wrapped_custom_getter


def _get_unique_variable_scope(prefix):
  """Get a name with the given prefix unique in the current variable scope."""
  var_scope_store = get_variable_scope_store()
  current_scope = get_variable_scope()
  name = current_scope.name + "/" + prefix if current_scope.name else prefix
  if var_scope_store.variable_scope_count(name) == 0:
    return prefix
  idx = 1
  while var_scope_store.variable_scope_count(name + ("_%d" % idx)) > 0:
    idx += 1
  return prefix + ("_%d" % idx)


# Named like a function for backwards compatibility with the
# @tf_contextlib.contextmanager version, which was switched to a class to avoid
# some object creation overhead.
@tf_export(v1=["variable_scope"])  # pylint: disable=invalid-name
class variable_scope(object):
  """A context manager for defining ops that creates variables (layers).

  This context manager validates that the (optional) `values` are from the same
  graph, ensures that graph is the default graph, and pushes a name scope and a
  variable scope.

  If `name_or_scope` is not None, it is used as is. If `name_or_scope` is None,
  then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will be made unique by appending `_N`
  to it.

  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](https://tensorflow.org/guide/variables), here
  we present only a few basic examples.

  Simple example of how to create a new variable:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      with tf.compat.v1.variable_scope("bar"):
          v = tf.compat.v1.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```

  Simple example of how to reenter a premade variable scope safely:

  ```python
  with tf.compat.v1.variable_scope("foo") as vs:
    pass

  # Re-enter the variable scope.
  with tf.compat.v1.variable_scope(vs,
                         auxiliary_name_scope=False) as vs1:
    # Restore the original name_scope.
    with tf.name_scope(vs1.original_name_scope):
        v = tf.compat.v1.get_variable("v", [1])
        assert v.name == "foo/v:0"
        c = tf.constant([1], name="c")
        assert c.name == "foo/c:0"
  ```

  Basic example of sharing a variable AUTO_REUSE:

  ```python
  def foo():
    with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable("v", [1])
    return v

  v1 = foo()  # Creates v.
  v2 = foo()  # Gets the same, existing v.
  assert v1 == v2
  ```

  Basic example of sharing a variable with reuse=True:

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.compat.v1.variable_scope("foo") as scope:
      v = tf.compat.v1.get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.compat.v1.get_variable("v", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.

  ```python
  with tf.compat.v1.variable_scope("foo"):
      v = tf.compat.v1.get_variable("v", [1])
      v1 = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```

  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.

  ```python
  with tf.compat.v1.variable_scope("foo", reuse=True):
      v = tf.compat.v1.get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope, then all
  its sub-scopes become reusing as well.

  A note about name scoping: Setting `reuse` does not impact the naming of other
  ops such as mult. See related discussion on
  [github#6189](https://github.com/tensorflow/tensorflow/issues/6189)

  Note that up to and including version 1.0, it was allowed (though explicitly
  discouraged) to pass False to the reuse argument, yielding undocumented
  behaviour slightly different from None. Starting at 1.1.0 passing None and
  False as reuse has exactly the same effect.

  A note about using variable scopes in multi-threaded environment: Variable
  scopes are thread local, so one thread will not see another thread's current
  scope. Also, when using `default_name`, unique scopes names are also generated
  only on a per thread basis. If the same name was used within a different
  thread, that doesn't prevent a new thread from creating the same scope.
  However, the underlying variable store is shared across threads (within the
  same graph). As such, if another thread tries to create a new variable with
  the same name as a variable created by a previous thread, it will fail unless
  reuse is True.

  Further, each thread starts with an empty variable scope. So if you wish to
  preserve name prefixes from a scope from the main thread, you should capture
  the main thread's scope and re-enter it in each thread. For e.g.

  ```
  main_thread_scope = variable_scope.get_variable_scope()

  # Thread's target function:
  def thread_target_fn(captured_scope):
    with variable_scope.variable_scope(captured_scope):
      # .... regular code for this thread


  thread = threading.Thread(target=thread_target_fn, args=(main_thread_scope,))
  ```
  """

  def __init__(self,
               name_or_scope,
               default_name=None,
               values=None,
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               reuse=None,
               dtype=None,
               use_resource=None,
               constraint=None,
               auxiliary_name_scope=True):
    """Initialize the context manager.

    Args:
      name_or_scope: `string` or `VariableScope`: the scope to open.
      default_name: The default name to use if the `name_or_scope` argument is
        `None`, this name will be uniquified. If name_or_scope is provided it
        won't be used and therefore it is not required and can be None.
      values: The list of `Tensor` arguments that are passed to the op function.
      initializer: default initializer for variables within this scope.
      regularizer: default regularizer for variables within this scope.
      caching_device: default caching device for variables within this scope.
      partitioner: default partitioner for variables within this scope.
      custom_getter: default custom getter for variables within this scope.
      reuse: `True`, None, or tf.compat.v1.AUTO_REUSE; if `True`, we go into
        reuse mode for this scope as well as all sub-scopes; if
        tf.compat.v1.AUTO_REUSE, we create variables if they do not exist, and
        return them otherwise; if None, we inherit the parent scope's reuse
        flag. When eager execution is enabled, new variables are always created
        unless an EagerVariableStore or template is currently active.
      dtype: type of variables created in this scope (defaults to the type in
        the passed scope, or inherited from parent scope).
      use_resource: If False, all variables will be regular Variables. If True,
        experimental ResourceVariables with well-defined semantics will be used
        instead. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be True.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      auxiliary_name_scope: If `True`, we create an auxiliary name scope with
        the scope. If `False`, we don't create it. Note that the argument is not
        inherited, and it only takes effect for once when creating. You should
        only use it for re-entering a premade variable scope.

    Returns:
      A scope that can be captured and reused.

    Raises:
      ValueError: when trying to reuse within a create scope, or create within
        a reuse scope.
      TypeError: when the types of some arguments are not appropriate.
    """
    self._name_or_scope = name_or_scope
    self._default_name = default_name
    self._values = values
    self._initializer = initializer
    self._regularizer = regularizer
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._reuse = reuse
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    if self._default_name is None and self._name_or_scope is None:
      raise TypeError("If default_name is None then name_or_scope is required")
    if self._reuse is False:
      # We don't allow non-inheriting scopes, False = None here.
      self._reuse = None
    if not (self._reuse is True
            or self._reuse is None
            or self._reuse is AUTO_REUSE):
      raise ValueError("The reuse parameter must be True or False or None.")
    if self._values is None:
      self._values = []
    self._in_graph_mode = not context.executing_eagerly()
    if self._in_graph_mode:
      self._graph = ops._get_graph_from_inputs(self._values)  # pylint: disable=protected-access
    self._cached_pure_variable_scope = None
    self._current_name_scope = None
    if not isinstance(auxiliary_name_scope, bool):
      raise TypeError("The auxiliary_name_scope must be `True` or `False`, "
                      "while get {}".format(auxiliary_name_scope))
    self._auxiliary_name_scope = auxiliary_name_scope

  def __enter__(self):
    # If the default graph is building a function, then we should not replace it
    # with the cached graph.
    if ops.get_default_graph().building_function:
      self._building_function = True
    else:
      self._building_function = False
    if self._in_graph_mode and not self._building_function:
      self._graph_context_manager = self._graph.as_default()
      self._graph_context_manager.__enter__()
    if self._cached_pure_variable_scope is not None:
      # Fast path for re-entering variable_scopes. We've held on to the pure
      # variable scope from a previous successful __enter__, so we avoid some
      # overhead by re-using that object.
      if self._current_name_scope is not None:
        self._current_name_scope.__enter__()
      return self._cached_pure_variable_scope.__enter__()

    try:
      return self._enter_scope_uncached()
    except:
      if (self._in_graph_mode and not self._building_function and
          self._graph_context_manager is not None):
        self._graph_context_manager.__exit__(*sys.exc_info())
      raise

  def _enter_scope_uncached(self):
    """Enters the context manager when there is no cached scope yet.

    Returns:
      The entered variable scope.

    Raises:
      TypeError: A wrong type is passed as `scope` at __init__().
      ValueError: `reuse` is incorrectly set at __init__().
    """
    if self._auxiliary_name_scope:
      # Create a new name scope later
      current_name_scope = None
    else:
      # Reenter the current name scope
      name_scope = ops.get_name_scope()
      if name_scope:
        # Hack to reenter
        name_scope += "/"
        current_name_scope = ops.name_scope(name_scope)
      else:
        # Root scope
        current_name_scope = ops.name_scope(name_scope)

    # IMPORTANT: Only assign to self._cached_pure_variable_scope and
    # self._current_name_scope after successful __enter__() calls.
    if self._name_or_scope is not None:
      if not isinstance(self._name_or_scope,
                        (VariableScope,) + six.string_types):
        raise TypeError("VariableScope: name_or_scope must be a string or "
                        "VariableScope.")
      if isinstance(self._name_or_scope, six.string_types):
        name_scope = self._name_or_scope
      else:
        name_scope = self._name_or_scope.name.split("/")[-1]
      if name_scope or current_name_scope:
        current_name_scope = current_name_scope or ops.name_scope(name_scope)
        try:
          current_name_scope_name = current_name_scope.__enter__()
        except:
          current_name_scope.__exit__(*sys.exc_info())
          raise
        self._current_name_scope = current_name_scope
        if isinstance(self._name_or_scope, six.string_types):
          old_name_scope = current_name_scope_name
        else:
          old_name_scope = self._name_or_scope.original_name_scope
        pure_variable_scope = _pure_variable_scope(
            self._name_or_scope,
            reuse=self._reuse,
            initializer=self._initializer,
            regularizer=self._regularizer,
            caching_device=self._caching_device,
            partitioner=self._partitioner,
            custom_getter=self._custom_getter,
            old_name_scope=old_name_scope,
            dtype=self._dtype,
            use_resource=self._use_resource,
            constraint=self._constraint)
        try:
          entered_pure_variable_scope = pure_variable_scope.__enter__()
        except:
          pure_variable_scope.__exit__(*sys.exc_info())
          raise
        self._cached_pure_variable_scope = pure_variable_scope
        return entered_pure_variable_scope
      else:
        self._current_name_scope = None
        # This can only happen if someone is entering the root variable scope.
        pure_variable_scope = _pure_variable_scope(
            self._name_or_scope,
            reuse=self._reuse,
            initializer=self._initializer,
            regularizer=self._regularizer,
            caching_device=self._caching_device,
            partitioner=self._partitioner,
            custom_getter=self._custom_getter,
            dtype=self._dtype,
            use_resource=self._use_resource,
            constraint=self._constraint)
        try:
          entered_pure_variable_scope = pure_variable_scope.__enter__()
        except:
          pure_variable_scope.__exit__(*sys.exc_info())
          raise
        self._cached_pure_variable_scope = pure_variable_scope
        return entered_pure_variable_scope

    else:  # Here name_or_scope is None. Using default name, but made unique.
      if self._reuse:
        raise ValueError("reuse=True cannot be used without a name_or_scope")
      current_name_scope = current_name_scope or ops.name_scope(
          self._default_name)
      try:
        current_name_scope_name = current_name_scope.__enter__()
      except:
        current_name_scope.__exit__(*sys.exc_info())
        raise
      self._current_name_scope = current_name_scope
      unique_default_name = _get_unique_variable_scope(self._default_name)
      pure_variable_scope = _pure_variable_scope(
          unique_default_name,
          initializer=self._initializer,
          regularizer=self._regularizer,
          caching_device=self._caching_device,
          partitioner=self._partitioner,
          custom_getter=self._custom_getter,
          old_name_scope=current_name_scope_name,
          dtype=self._dtype,
          use_resource=self._use_resource,
          constraint=self._constraint)
      try:
        entered_pure_variable_scope = pure_variable_scope.__enter__()
      except:
        pure_variable_scope.__exit__(*sys.exc_info())
        raise
      self._cached_pure_variable_scope = pure_variable_scope
      return entered_pure_variable_scope

  def __exit__(self, type_arg, value_arg, traceback_arg):
    try:
      self._cached_pure_variable_scope.__exit__(type_arg, value_arg,
                                                traceback_arg)
    finally:
      try:
        if self._current_name_scope:
          self._current_name_scope.__exit__(type_arg, value_arg,
                                            traceback_arg)
      finally:
        if self._in_graph_mode and not self._building_function:
          self._graph_context_manager.__exit__(type_arg, value_arg,
                                               traceback_arg)


# pylint: disable=g-doc-return-or-yield
@tf_export(v1=["variable_op_scope"])
@tf_contextlib.contextmanager
def variable_op_scope(values,
                      name_or_scope,
                      default_name=None,
                      initializer=None,
                      regularizer=None,
                      caching_device=None,
                      partitioner=None,
                      custom_getter=None,
                      reuse=None,
                      dtype=None,
                      use_resource=None,
                      constraint=None):
  """Deprecated: context manager for defining an op that creates variables."""
  logging.warn("tf.variable_op_scope(values, name, default_name) is deprecated,"
               " use tf.variable_scope(name, default_name, values)")
  with variable_scope(
      name_or_scope,
      default_name=default_name,
      values=values,
      initializer=initializer,
      regularizer=regularizer,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=custom_getter,
      reuse=reuse,
      dtype=dtype,
      use_resource=use_resource,
      constraint=constraint) as scope:
    yield scope


def _call_partitioner(partitioner, shape, dtype):
  """Call partitioner validating its inputs/output.

  Args:
    partitioner: a function mapping `Tensor` shape and dtype to a list of
      partitions.
    shape: shape of the `Tensor` to partition, must have at least two
      dimensions.
    dtype: dtype of the elements in the `Tensor`.

  Returns:
    A list with elements >=1 and exactly one >1. The index of that
    element corresponds to the partitioning axis.
  """
  if not shape.is_fully_defined():
    raise ValueError("Shape of a new partitioned variable must be "
                     "fully defined, but instead was %s." % (shape,))
  if shape.ndims < 1:
    raise ValueError("A partitioned Variable must have rank at least 1, "
                     "shape: %s" % shape)

  slicing = partitioner(shape=shape, dtype=dtype)
  if not isinstance(slicing, collections_lib.Sequence):
    raise ValueError("Partitioner must return a sequence, but saw: %s" %
                     slicing)
  if len(slicing) != shape.ndims:
    raise ValueError(
        "Partitioner returned a partition list that does not match the "
        "Variable's rank: %s vs. %s" % (slicing, shape))
  if any(p < 1 for p in slicing):
    raise ValueError("Partitioner returned zero partitions for some axes: %s" %
                     slicing)
  if sum(p > 1 for p in slicing) > 1:
    raise ValueError("Can only slice a variable along one dimension: "
                     "shape: %s, partitioning: %s" % (shape, slicing))
  return slicing


# TODO(slebedev): could be inlined, but
# `_VariableStore._get_partitioned_variable` is too complex even
# without this logic.
def _get_slice_dim_and_num_slices(slicing):
  """Get slicing dimension and number of slices from the partitioner output."""
  for slice_dim, num_slices in enumerate(slicing):
    if num_slices > 1:
      break
  else:
    # Degenerate case: no partitioning applied.
    slice_dim = 0
    num_slices = 1
  return slice_dim, num_slices


def _iter_slices(full_shape, num_slices, slice_dim):
  """Slices a given a shape along the specified dimension."""
  num_slices_with_excess = full_shape[slice_dim] % num_slices
  offset = [0] * len(full_shape)
  min_slice_len = full_shape[slice_dim] // num_slices
  for i in xrange(num_slices):
    shape = full_shape[:]
    shape[slice_dim] = min_slice_len + bool(i < num_slices_with_excess)
    yield offset[:], shape
    offset[slice_dim] += shape[slice_dim]


def default_variable_creator(next_creator=None, **kwargs):
  """Default variable creator."""
  assert next_creator is None
  initial_value = kwargs.get("initial_value", None)
  trainable = kwargs.get("trainable", None)
  collections = kwargs.get("collections", None)
  validate_shape = kwargs.get("validate_shape", True)
  caching_device = kwargs.get("caching_device", None)
  name = kwargs.get("name", None)
  variable_def = kwargs.get("variable_def", None)
  dtype = kwargs.get("dtype", None)
  expected_shape = kwargs.get("expected_shape", None)
  import_scope = kwargs.get("import_scope", None)
  constraint = kwargs.get("constraint", None)
  use_resource = kwargs.get("use_resource", None)
  synchronization = kwargs.get("synchronization", None)
  aggregation = kwargs.get("aggregation", None)
  shape = kwargs.get("shape", None)

  if use_resource is None:
    use_resource = get_variable_scope().use_resource
  if use_resource is None:
    use_resource = _DEFAULT_USE_RESOURCE
  use_resource = use_resource or context.executing_eagerly()
  if use_resource:
    distribute_strategy = kwargs.get("distribute_strategy", None)
    return resource_variable_ops.ResourceVariable(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        dtype=dtype,
        constraint=constraint,
        variable_def=variable_def,
        import_scope=import_scope,
        distribute_strategy=distribute_strategy,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape)
  else:
    return variables.RefVariable(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        dtype=dtype,
        constraint=constraint,
        variable_def=variable_def,
        expected_shape=expected_shape,
        import_scope=import_scope,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape)


def default_variable_creator_v2(next_creator=None, **kwargs):
  """Default variable creator."""
  assert next_creator is None
  initial_value = kwargs.get("initial_value", None)
  trainable = kwargs.get("trainable", None)
  validate_shape = kwargs.get("validate_shape", True)
  caching_device = kwargs.get("caching_device", None)
  name = kwargs.get("name", None)
  variable_def = kwargs.get("variable_def", None)
  dtype = kwargs.get("dtype", None)
  import_scope = kwargs.get("import_scope", None)
  constraint = kwargs.get("constraint", None)
  distribute_strategy = kwargs.get("distribute_strategy", None)
  synchronization = kwargs.get("synchronization", None)
  aggregation = kwargs.get("aggregation", None)
  shape = kwargs.get("shape", None)

  return resource_variable_ops.ResourceVariable(
      initial_value=initial_value,
      trainable=trainable,
      validate_shape=validate_shape,
      caching_device=caching_device,
      name=name,
      dtype=dtype,
      constraint=constraint,
      variable_def=variable_def,
      import_scope=import_scope,
      distribute_strategy=distribute_strategy,
      synchronization=synchronization,
      aggregation=aggregation,
      shape=shape)


variables.default_variable_creator = default_variable_creator
variables.default_variable_creator_v2 = default_variable_creator_v2


def _make_getter(captured_getter, captured_previous):
  """Gets around capturing loop variables in python being broken."""
  return lambda **kwargs: captured_getter(captured_previous, **kwargs)


# TODO(apassos) remove forwarding symbol
variable = variables.VariableV1


@tf_export(v1=["variable_creator_scope"])
@tf_contextlib.contextmanager
def variable_creator_scope_v1(variable_creator):
  """Scope which defines a variable creation function to be used by variable().

  variable_creator is expected to be a function with the following signature:

  ```
    def variable_creator(next_creator, **kwargs)
  ```

  The creator is supposed to eventually call the next_creator to create a
  variable if it does want to create a variable and not call Variable or
  ResourceVariable directly. This helps make creators composable. A creator may
  choose to create multiple variables, return already existing variables, or
  simply register that a variable was created and defer to the next creators in
  line. Creators can also modify the keyword arguments seen by the next
  creators.

  Custom getters in the variable scope will eventually resolve down to these
  custom creators when they do create variables.

  The valid keyword arguments in kwds are:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        `trainable` defaults to `True`, unless `synchronization` is
        set to `ON_READ`, in which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
      constraint: A constraint function to be applied to the variable after
        updates by some algorithms.
      use_resource: if True, a ResourceVariable is always created.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

  This set may grow over time, so it's important the signature of creators is as
  mentioned above.

  Args:
    variable_creator: the passed creator

  Yields:
    A scope in which the creator is active
  """
  with ops.get_default_graph()._variable_creator_scope(variable_creator):  # pylint: disable=protected-access
    yield


# Note: only the docstrings differ between this and v1.
@tf_export("variable_creator_scope", v1=[])
@tf_contextlib.contextmanager
def variable_creator_scope(variable_creator):
  """Scope which defines a variable creation function to be used by variable().

  variable_creator is expected to be a function with the following signature:

  ```
    def variable_creator(next_creator, **kwargs)
  ```

  The creator is supposed to eventually call the next_creator to create a
  variable if it does want to create a variable and not call Variable or
  ResourceVariable directly. This helps make creators composable. A creator may
  choose to create multiple variables, return already existing variables, or
  simply register that a variable was created and defer to the next creators in
  line. Creators can also modify the keyword arguments seen by the next
  creators.

  Custom getters in the variable scope will eventually resolve down to these
  custom creators when they do create variables.

  The valid keyword arguments in kwds are:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, the default, GradientTapes automatically watch
        uses of this Variable.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
      constraint: A constraint function to be applied to the variable after
        updates by some algorithms.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

  This set may grow over time, so it's important the signature of creators is as
  mentioned above.

  Args:
    variable_creator: the passed creator

  Yields:
    A scope in which the creator is active
  """
  with ops.get_default_graph()._variable_creator_scope(variable_creator):  # pylint: disable=protected-access
    yield
