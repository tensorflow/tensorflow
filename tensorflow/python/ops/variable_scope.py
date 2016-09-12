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
import contextlib
import functools
import traceback

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

__all__ = ["VariableScope", "get_variable_scope",
           "get_variable", "get_local_variable", "variable_scope",
           "variable_op_scope", "no_regularizer"]


class _PartitionInfo(object):
  """Holds partition info used by initializer functions.
  """

  def __init__(self, full_shape, var_offset):
    """Constructor.

    Args:
      full_shape: Tuple or list of `int` indicating the full combined shape
        of the partitioned variables.
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

    for i in xrange(len(full_shape)):
      offset = var_offset[i]
      shape = full_shape[i]
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
          "self.full_shape={} is of length {}.".format(shape, len(
              shape), self.full_shape, len(self.full_shape)))

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


class _VariableStore(object):
  """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys
          and the corresponding TensorFlow Variables as values.
  """

  def __init__(self):
    """Create a variable store."""
    self._vars = {}  # A dictionary of the stored TensorFlow variables.
    self._partitioned_vars = {}  # A dict of the stored PartitionedVariables.
    self._variable_scopes_count = {}  # Count re-used variable scopes.

  def open_variable_scope(self, scope_name):
    if scope_name in self._variable_scopes_count:
      self._variable_scopes_count[scope_name] += 1
    else:
      self._variable_scopes_count[scope_name] = 1

  def close_variable_subscopes(self, scope_name):
    for k in self._variable_scopes_count:
      if not scope_name or k.startswith(scope_name + "/"):
        self._variable_scopes_count[k] = 0

  def variable_scope_count(self, scope_name):
    return self._variable_scopes_count.get(scope_name, 0)

  def get_variable(self, name, shape=None, dtype=dtypes.float32,
                   initializer=None, regularizer=None, reuse=None,
                   trainable=True, collections=None, caching_device=None,
                   partitioner=None, validate_shape=True, custom_getter=None):
    """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    If `reuse` is `None` (the default), both new and existing variables are
    returned.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `uniform_unit_scaling_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If a partitioner is provided, first a sharded `Variable` is created
    via `_get_partitioned_variable`, and the return value is a
    `Tensor` composed of the shards concatenated along the partition axis.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: Initializer for the variable.
      regularizer: A (Tensor -> Tensor or None) function; the result of
        applying it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean or `None`. Controls reuse or creation of variables.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      collections: List of graph collections keys to add the Variable to.
        Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method.
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

    Returns:
      The created or existing variable.

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
    """
    if custom_getter is not None and not callable(custom_getter):
      raise ValueError(
          "Passed a custom_getter which is not callable: %s" % custom_getter)

    # This is the main logic of get_variable.  However, custom_getter
    # may override this logic.  So we save it as a callable and pass
    # it to custom_getter.
    # Note: the parameters of _true_getter, and their documentation, match
    # *exactly* item-for-item with the docstring of this method.
    def _true_getter(name, shape=None, dtype=dtypes.float32,  # pylint: disable=missing-docstring
                     initializer=None, regularizer=None, reuse=None,
                     trainable=True, collections=None, caching_device=None,
                     partitioner=None, validate_shape=True):
      # Partitioned variable case
      if partitioner is not None:
        if not callable(partitioner):
          raise ValueError(
              "Partitioner must be callable, but received: %s" % partitioner)
        with ops.name_scope(None):
          return self._get_partitioned_variable(name=name,
                                                shape=shape,
                                                dtype=dtype,
                                                initializer=initializer,
                                                regularizer=regularizer,
                                                reuse=reuse,
                                                trainable=trainable,
                                                collections=collections,
                                                caching_device=caching_device,
                                                partitioner=partitioner,
                                                validate_shape=validate_shape)

      # Special case for partitioned variable to allow reuse without having to
      # specify partitioner.
      if (reuse is True and partitioner is None
          and name in self._partitioned_vars):
        return self._get_partitioned_variable(name=name,
                                              shape=shape,
                                              dtype=dtype,
                                              initializer=initializer,
                                              regularizer=regularizer,
                                              reuse=reuse,
                                              trainable=trainable,
                                              collections=collections,
                                              caching_device=caching_device,
                                              partitioner=None,
                                              validate_shape=validate_shape)

      # Single variable case
      if "%s/part_0" % name in self._vars:
        raise ValueError(
            "No partitioner was provided, but a partitioned version of the "
            "variable was found: %s/part_0. Perhaps a variable of the same "
            "name was already created with partitioning?" % name)

      return self._get_single_variable(
          name=name, shape=shape, dtype=dtype,
          initializer=initializer, regularizer=regularizer, reuse=reuse,
          trainable=trainable, collections=collections,
          caching_device=caching_device, validate_shape=validate_shape)

    if custom_getter is not None:
      return custom_getter(
          getter=_true_getter, name=name, shape=shape, dtype=dtype,
          initializer=initializer, regularizer=regularizer,
          reuse=reuse, trainable=trainable, collections=collections,
          caching_device=caching_device, partitioner=partitioner,
          validate_shape=validate_shape)
    else:
      return _true_getter(
          name, shape=shape, dtype=dtype,
          initializer=initializer, regularizer=regularizer,
          reuse=reuse, trainable=trainable, collections=collections,
          caching_device=caching_device, partitioner=partitioner,
          validate_shape=validate_shape)

  def _get_partitioned_variable(
      self, name, partitioner, shape=None, dtype=dtypes.float32,
      initializer=None, regularizer=None, reuse=None,
      trainable=True, collections=None, caching_device=None,
      validate_shape=True):
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
    If `reuse` is `None` (the default), both new and existing variables are
    returned.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `uniform_unit_scaling_initializer`. If initializer is a Tensor, we use
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
      dtype: type of the new or existing sharded variable
        (defaults to `DT_FLOAT`).
      initializer: initializer for the sharded variable.
      regularizer: a (Tensor -> Tensor or None) function; the result of
        applying it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean or `None`. Controls reuse or creation of variables.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      collections: List of graph collections keys to add the Variable to.
        Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.

    Returns:
      A tuple `(shards, partitions)` where `shards` is the list of `Variable`
      shards and `partitions` is the output of the partitioner on the input
      shape.

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        when violating reuse during variable creation, or if an existing
        sharded variable exists for the given name but with different sharding.
    """

    initializing_from_value = initializer is not None and isinstance(
        initializer, ops.Tensor)
    reuse_without_partition = reuse is True and partitioner is None

    if name in self._vars:
      raise ValueError(
          "A partitioner was provided, but an unpartitioned version of the "
          "variable was found: %s.  Perhaps a variable of the same name was "
          "already created without partitioning?" % name)

    shape = tensor_shape.as_shape(shape)
    if initializing_from_value:
      shape = shape.merge_with(initializer.get_shape())

    if not reuse_without_partition:
      if not shape.is_fully_defined():
        raise ValueError("Shape of a new partitioned variable (%s) must be "
                         "fully defined, but instead was %s." % (name, shape))

      if shape.ndims < 1:
        raise ValueError("A partitioned Variable must have rank at least 1, "
                         "shape: %s" % shape)

      partitions = partitioner(shape=shape, dtype=dtype)

      if not isinstance(partitions, collections_lib.Sequence):
        raise ValueError("Partitioner must return a sequence, but saw: %s"
                         % partitions)

      if len(partitions) != shape.ndims:
        raise ValueError(
            "Partitioner returned a partition list that does not match the "
            "Variable's rank: %s vs. %s" % (partitions, shape))

      if any([p < 1 for p in partitions]):
        raise ValueError(
            "Partitioner returned zero partitions for some axes: %s" %
            partitions)

    should_check = reuse is not None

    if name in self._partitioned_vars:
      if should_check and not reuse:
        raise ValueError(
            "Partitioned variable with name %s already exists. Did you mean to "
            "set reuse=True in VarScope?"
            % name)

      existing_var = self._partitioned_vars[name]
      if not shape.is_compatible_with(existing_var.get_shape()):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified shape %s "
            "and found shape %s."
            % (name, shape, existing_var.get_shape()))
      if not dtype.is_compatible_with(existing_var.dtype):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified dtype %s "
            "and found dtype %s."
            % (name, dtype.name, existing_var.dtype.name))

      # pylint: disable=protected-access
      if (not reuse_without_partition and
          existing_var._get_partitions() != partitions):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified partitions "
            "%s and found partitions %s." %
            (name, partitions, existing_var._get_partitions()))
      # pylint: enable=protected-access

      return existing_var

    if should_check and reuse:
      raise ValueError("PartitionedVariable %s does not exist, or was not "
                       "created with tf.get_variable(). Did you mean to set "
                       "reuse=None in VarScope?" % name)

    slice_dim, slice_shape = _compute_slice_dim_and_shape(
        shape.as_list(), partitions)

    vs = []
    num_slices = partitions[slice_dim]
    num_slices_with_excess = shape[slice_dim].value % num_slices

    slice_offset = [0] * shape.ndims

    if "%s/part_0" % name in self._vars:
      if "%s/part_%d" % (name, num_slices - 1) not in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but %s/part_%d was not."
            % (num_slices, name, name, num_slices - 1))
      if "%s/part_%d" % (name, num_slices) in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but so was the extra shard %s/part_%d."
            % (num_slices, name, name, num_slices))

    for i in xrange(num_slices):
      var_shape = slice_shape[:]
      var_offset = slice_offset[:]
      partition_info = _PartitionInfo(
          full_shape=shape.as_list(), var_offset=var_offset)
      if i < num_slices_with_excess:
        var_shape[slice_dim] += 1
      slice_offset[slice_dim] += var_shape[slice_dim]

      var_full_name = "%s/part_%d" % (name, i)
      with ops.name_scope(var_full_name + "/PartitionedInitializer"):
        if initializer is None:
          init = init_ops.uniform_unit_scaling_initializer()
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
            validate_shape=validate_shape)

      # pylint: disable=protected-access
      var._set_save_slice_info(variables.Variable.SaveSliceInfo(
          name, shape.as_list(), var_offset, var_shape))
      vs.append(var)
      # pylint: enable=protected-access

      # pylint: disable=protected-access
    partitioned_var = variables.PartitionedVariable(name=name,
                                                    shape=shape,
                                                    dtype=dtype,
                                                    variable_list=vs,
                                                    partitions=partitions)
    # pylint: enable=protected-access

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
                           trainable=True,
                           collections=None,
                           caching_device=None,
                           validate_shape=True):
    """Get or create a single Variable (e.g. a shard or entire variable).

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

    should_check = reuse is not None
    dtype = dtypes.as_dtype(dtype)
    shape = tensor_shape.as_shape(shape)

    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if should_check and not reuse:
        tb = self._vars[name].op.traceback[::-1]
        # Throw away internal tf entries and only take a few lines.
        tb = [x for x in tb if "tensorflow/python" not in x[0]][:3]
        raise ValueError("Variable %s already exists, disallowed."
                         " Did you mean to set reuse=True in VarScope? "
                         "Originally defined at:\n\n%s" % (
                             name, "".join(traceback.format_list(tb))))
      found_var = self._vars[name]
      if not shape.is_compatible_with(found_var.get_shape()):
        raise ValueError("Trying to share variable %s, but specified shape %s"
                         " and found shape %s." % (name, shape,
                                                   found_var.get_shape()))
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError("Trying to share variable %s, but specified dtype %s"
                         " and found dtype %s." % (name, dtype_str,
                                                   found_type_str))
      return found_var

    # The code below handles only the case of creating a new variable.
    if should_check and reuse:
      raise ValueError("Variable %s does not exist, or was not created with "
                       "tf.get_variable(). Did you mean to set reuse=None in "
                       "VarScope?" % name)
    if not shape.is_fully_defined() and not initializing_from_value:
      raise ValueError("Shape of a new variable (%s) must be fully defined, "
                       "but instead was %s." % (name, shape))

    # Create the tensor to initialize the variable.
    if initializer is None:
      initializer = init_ops.uniform_unit_scaling_initializer()
    # Clear control dependencies while creating the initializer.
    with ops.control_dependencies(None):
      if initializing_from_value:
        init_val = initializer
        variable_dtype = None
      else:
        init_val = lambda: initializer(
            shape.as_list(), dtype=dtype, partition_info=partition_info)
        variable_dtype = dtype.base_dtype

    # Create the variable.
    v = variables.Variable(initial_value=init_val,
                           name=name,
                           trainable=trainable,
                           collections=collections,
                           caching_device=caching_device,
                           dtype=variable_dtype,
                           validate_shape=validate_shape)
    self._vars[name] = v
    logging.vlog(1, "Created variable %s with shape %s and init %s", v.name,
                 format(shape), initializer)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      with ops.colocate_with(v.op):
        with ops.name_scope(name + "/Regularizer/"):
          loss = regularizer(v)
        if loss is not None:
          logging.vlog(1, "Applied regularizer to %s and added the result %s "
                       "to REGULARIZATION_LOSSES.", v.name, loss.name)
          ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)

    return v


# To stop regularization, use this regularizer
def no_regularizer(_):
  """Use this function to prevent regularization of variables."""
  return None


class VariableScope(object):
  """Variable scope object to carry defaults to provide to get_variable.

  Many of the arguments we need for get_variable in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    regularizer: default regularizer passed to get_variable.
    reuse: Boolean or None, setting the reuse in get_variable.
    caching_device: string, callable, or None: the caching device passed to
      get_variable.
    partitioner: callable or `None`: the partitioner passed to `get_variable`.
    custom_getter: default custom getter passed to get_variable.
    name_scope: The name passed to `tf.name_scope`.
    dtype: default type passed to get_variable (defaults to DT_FLOAT).
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
               dtype=dtypes.float32):
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

  def reuse_variables(self):
    """Reuse variables in this scope."""
    self._reuse = True

  def set_initializer(self, initializer):
    """Set initializer for this scope."""
    self._initializer = initializer

  def set_dtype(self, dtype):
    """Set data type for this scope."""
    self._dtype = dtype

  def set_regularizer(self, regularizer):
    """Set regularizer for this scope."""
    self._regularizer = regularizer

  def set_caching_device(self, caching_device):
    """Set caching_device for this scope."""
    self._caching_device = caching_device

  def set_partitioner(self, partitioner):
    """Set partitioner for this scope."""
    self._partitioner = partitioner

  def set_custom_getter(self, custom_getter):
    """Set custom getter for this scope."""
    self._custom_getter = custom_getter

  def get_variable(self,
                   var_store,
                   name,
                   shape=None,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   collections=None,
                   caching_device=None,
                   partitioner=None,
                   validate_shape=True,
                   custom_getter=None):
    """Gets an existing variable with this name or create a new one."""
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if custom_getter is None:
      custom_getter = self._custom_getter

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
      if dtype is None:
        dtype = self._dtype

      return var_store.get_variable(
          full_name, shape=shape, dtype=dtype, initializer=initializer,
          regularizer=regularizer, reuse=self.reuse, trainable=trainable,
          collections=collections, caching_device=caching_device,
          partitioner=partitioner, validate_shape=validate_shape,
          custom_getter=custom_getter)

  def _get_partitioned_variable(self,
                                var_store,
                                name,
                                shape=None,
                                dtype=None,
                                initializer=None,
                                regularizer=None,
                                trainable=True,
                                collections=None,
                                caching_device=None,
                                partitioner=None,
                                validate_shape=True):
    """Gets an existing variable with this name or create a new one."""
    if initializer is None:
      initializer = self._initializer
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if dtype is None:
      dtype = self._dtype

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
          full_name, shape=shape, dtype=dtype, initializer=initializer,
          regularizer=regularizer, reuse=self.reuse, trainable=trainable,
          collections=collections, caching_device=caching_device,
          partitioner=partitioner, validate_shape=validate_shape)
      # pylint: enable=protected-access


_VARSTORE_KEY = ("__variable_store",)
_VARSCOPE_KEY = ("__varscope",)


def get_variable_scope():
  """Returns the current variable scope."""
  scope = ops.get_collection(_VARSCOPE_KEY)
  if scope:  # This collection has at most 1 element, the default scope at [0].
    return scope[0]
  scope = VariableScope(False)
  ops.add_to_collection(_VARSCOPE_KEY, scope)
  return scope


def _get_default_variable_store():
  store = ops.get_collection(_VARSTORE_KEY)
  if store:
    return store[0]
  store = _VariableStore()
  ops.add_to_collection(_VARSTORE_KEY, store)
  return store


def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 custom_getter=None):
  """Gets an existing variable with these parameters or create a new one.

  This function prefixes the name with the current variable scope
  and performs reuse checks. See the
  [Variable Scope How To](../../how_tos/variable_scope/index.md)
  for an extensive description of how reusing works. Here is a basic example:

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
      w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
  with tf.variable_scope("foo", reuse=True)
      v1 = tf.get_variable("v")  # The same as v above.
  ```

  If initializer is `None` (the default), the default initializer passed in
  the variable scope will be used. If that one is `None` too, a
  `uniform_unit_scaling_initializer` will be used. The initializer can also be
  a Tensor, in which case the variable is initialized to this value and shape.

  Similarly, if the regularizer is `None` (the default), the default regularizer
  passed in the variable scope will be used (if that is `None` too,
  then by default no regularization is performed).

  If a partitioner is provided, first a sharded `Variable` is created
  via `_get_partitioned_variable`, and the return value is a
  `Tensor` composed of the shards concatenated along the partition axis.

  Some useful partitioners are available.  See, e.g.,
  `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: Initializer for the variable if one is created.
    regularizer: A (Tensor -> Tensor or None) function; the result of
      applying it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    collections: List of graph collections keys to add the Variable to.
      Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).
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
        must be known.
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

  Returns:
    The created or existing variable.

  Raises:
    ValueError: when creating a new variable and shape is not declared,
      when violating reuse during variable creation, or when `initializer` dtype
      and `dtype` don't match. Reuse is set inside `variable_scope`.
  """
  return get_variable_scope().get_variable(
      _get_default_variable_store(), name, shape=shape, dtype=dtype,
      initializer=initializer, regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape,
      custom_getter=custom_getter)


@functools.wraps(get_variable)
def get_local_variable(*args, **kwargs):
  kwargs["trainable"] = False
  if "collections" in kwargs:
    kwargs["collections"] += [ops.GraphKeys.LOCAL_VARIABLES]
  else:
    kwargs["collections"] = [ops.GraphKeys.LOCAL_VARIABLES]
  get_local_variable.__doc__ = (
      "Gets an existing local variable or creates a new one.\n\n" +
      get_local_variable.__doc__)
  return get_variable(*args, **kwargs)


def _get_partitioned_variable(name,
                              shape=None,
                              dtype=None,
                              initializer=None,
                              regularizer=None,
                              trainable=True,
                              collections=None,
                              caching_device=None,
                              partitioner=None,
                              validate_shape=True):
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
  If `reuse` is `None` (the default), both new and existing variables are
  returned.

  If initializer is `None` (the default), the default initializer passed in
  the constructor is used. If that one is `None` too, we use a new
  `uniform_unit_scaling_initializer`. If initializer is a Tensor, we use
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
    regularizer: A (Tensor -> Tensor or None) function; the result of
      applying it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    collections: List of graph collections keys to add the Variable to.
      Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).
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
        must be known.

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
      _get_default_variable_store(), name, shape=shape, dtype=dtype,
      initializer=initializer, regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape)
  # pylint: enable=protected-access


@contextlib.contextmanager
def _pure_variable_scope(name_or_scope,
                         reuse=None,
                         initializer=None,
                         regularizer=None,
                         caching_device=None,
                         partitioner=None,
                         custom_getter=None,
                         old_name_scope=None,
                         dtype=dtypes.float32):
  """Creates a context for the variable_scope, see `variable_scope` for docs.

  Note: this does not create a name scope.

  Args:
    name_or_scope: `string` or `VariableScope`: the scope to open.
    reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
      well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
    initializer: default initializer for variables within this scope.
    regularizer: default regularizer for variables within this scope.
    caching_device: default caching device for variables within this scope.
    partitioner: default partitioner for variables within this scope.
    custom_getter: default custom getter for variables within this scope.
    old_name_scope: the original name scope when re-entering a variable scope.
    dtype: type of the variables within this scope (defaults to `DT_FLOAT`).

  Yields:
    A scope that can be to captured and reused.

  Raises:
    ValueError: when trying to reuse within a create scope, or create within
      a reuse scope, or if reuse is not `None` or `True`.
    TypeError: when the types of some arguments are not appropriate.

  """
  get_variable_scope()  # Ensure that a default exists, then get a pointer.
  # Get the reference to the collection as we want to modify it in place.
  default_varscope = ops.get_collection_ref(_VARSCOPE_KEY)
  old = default_varscope[0]
  var_store = _get_default_variable_store()
  if isinstance(name_or_scope, VariableScope):
    new_name = name_or_scope.name
  else:
    new_name = old.name + "/" + name_or_scope if old.name else name_or_scope
  try:
    var_store.open_variable_scope(new_name)
    if isinstance(name_or_scope, VariableScope):
      name_scope = name_or_scope._name_scope  # pylint: disable=protected-access
      # Handler for the case when we jump to a shared scope.
      #   We create a new VariableScope (default_varscope[0]) that contains
      #   a copy of the provided shared scope, possibly with changed reuse
      #   and initializer, if the user requested this.
      default_varscope[0] = VariableScope(
          name_or_scope.reuse if reuse is None else reuse,
          name=new_name,
          initializer=name_or_scope.initializer,
          regularizer=name_or_scope.regularizer,
          caching_device=name_or_scope.caching_device,
          partitioner=name_or_scope.partitioner,
          dtype=name_or_scope.dtype,
          custom_getter=name_or_scope.custom_getter,
          name_scope=name_scope)
      if initializer is not None:
        default_varscope[0].set_initializer(initializer)
      if regularizer is not None:
        default_varscope[0].set_regularizer(regularizer)
      if caching_device is not None:
        default_varscope[0].set_caching_device(caching_device)
      if partitioner is not None:
        default_varscope[0].set_partitioner(partitioner)
      if custom_getter is not None:
        default_varscope[0].set_custom_getter(custom_getter)
      if dtype is not None:
        default_varscope[0].set_dtype(dtype)
      yield default_varscope[0]
    else:
      # Handler for the case when we just prolong current variable scope.
      #   VariableScope with name extended by the provided one, and inherited
      #   reuse and initializer (except if the user provided values to set).
      reuse = reuse or old.reuse  # Re-using is inherited by sub-scopes.
      default_varscope[0] = VariableScope(
          reuse,
          name=new_name,
          initializer=old.initializer,
          regularizer=old.regularizer,
          caching_device=old.caching_device,
          partitioner=old.partitioner,
          dtype=old.dtype,
          custom_getter=old.custom_getter,
          name_scope=old_name_scope or name_or_scope)
      if initializer is not None:
        default_varscope[0].set_initializer(initializer)
      if regularizer is not None:
        default_varscope[0].set_regularizer(regularizer)
      if caching_device is not None:
        default_varscope[0].set_caching_device(caching_device)
      if partitioner is not None:
        default_varscope[0].set_partitioner(partitioner)
      if custom_getter is not None:
        default_varscope[0].set_custom_getter(custom_getter)
      if dtype is not None:
        default_varscope[0].set_dtype(dtype)
      yield default_varscope[0]
  finally:
    var_store.close_variable_subscopes(new_name)
    default_varscope[0] = old


def _get_unique_variable_scope(prefix):
  """Get a name with the given prefix unique in the current variable scope."""
  var_store = _get_default_variable_store()
  current_scope = get_variable_scope()
  name = current_scope.name + "/" + prefix if current_scope.name else prefix
  if var_store.variable_scope_count(name) == 0:
    return prefix
  idx = 1
  while var_store.variable_scope_count(name + ("_%d" % idx)) > 0:
    idx += 1
  return prefix + ("_%d" % idx)


# pylint: disable=g-doc-return-or-yield
@contextlib.contextmanager
def variable_scope(name_or_scope,
                   default_name=None,
                   values=None,
                   initializer=None,
                   regularizer=None,
                   caching_device=None,
                   partitioner=None,
                   custom_getter=None,
                   reuse=None,
                   dtype=None):
  """Returns a context manager for defining ops that creates variables (layers).

  This context manager validates that the (optional) `values` are from
  the same graph, ensures that graph is the default graph, and pushes a
  name scope and a variable scope.

  If `name_or_scope` is not None, it is used as is. If `scope` is None, then
  `default_name` is used.  In that case, if the same name has been previously
  used in the same scope, it will made unique be appending `_N` to it.

  Variable scope allows to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](../../how_tos/variable_scope/index.md),
  here we present only a few basic examples.

  Simple example of how to create a new variable:

  ```python
  with tf.variable_scope("foo"):
      with tf.variable_scope("bar"):
          v = tf.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```

  Basic example of sharing a variable:

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
  with tf.variable_scope("foo", reuse=True):
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.variable_scope("foo") as scope:
      v = tf.get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when
  getting an existing variable in a non-reusing scope.

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
      v1 = tf.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```

  Similarly, we raise an exception when trying to get a variable that
  does not exist in reuse mode.

  ```python
  with tf.variable_scope("foo", reuse=True):
      v = tf.get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope,
  then all its sub-scopes become reusing as well.

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
    reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
      well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
    dtype: type of variables created in this scope (defaults to the type
      in the passed scope, or inherited from parent scope).

  Returns:
    A scope that can be to captured and reused.

  Raises:
    ValueError: when trying to reuse within a create scope, or create within
      a reuse scope, or if reuse is not `None` or `True`.
    TypeError: when the types of some arguments are not appropriate.
  """
  if default_name is None and not name_or_scope:
    raise TypeError("If default_name is None then name_or_scope is required")
  if values is None:
    values = []
  g = ops._get_graph_from_inputs(values)  # pylint: disable=protected-access
  with g.as_default():
    if name_or_scope is not None:
      if not isinstance(name_or_scope, (VariableScope,) + six.string_types):
        raise TypeError("VariableScope: name_or_scope must be a string or "
                        "VariableScope.")
      if isinstance(name_or_scope, six.string_types):
        name_scope = name_or_scope
      else:
        name_scope = name_or_scope.name.split("/")[-1]
      if name_scope:
        with ops.name_scope(name_scope) as cur_name_scope:
          if isinstance(name_or_scope, six.string_types):
            old_name_scope = cur_name_scope
          else:
            old_name_scope = name_or_scope.original_name_scope
          with _pure_variable_scope(
              name_or_scope,
              reuse=reuse,
              initializer=initializer,
              regularizer=regularizer,
              caching_device=caching_device,
              partitioner=partitioner,
              custom_getter=custom_getter,
              old_name_scope=old_name_scope,
              dtype=dtype) as vs:
            yield vs
      else:
        # This can only happen if someone is entering the root variable scope.
        with _pure_variable_scope(
            name_or_scope,
            reuse=reuse,
            initializer=initializer,
            regularizer=regularizer,
            caching_device=caching_device,
            partitioner=partitioner,
            custom_getter=custom_getter,
            dtype=dtype) as vs:
          yield vs
    else:  # Here name_or_scope is None. Using default name, but made unique.
      if reuse:
        raise ValueError("reuse=True cannot be used without a name_or_scope")
      with ops.name_scope(default_name) as scope:
        unique_default_name = _get_unique_variable_scope(default_name)
        with _pure_variable_scope(
            unique_default_name,
            initializer=initializer,
            regularizer=regularizer,
            caching_device=caching_device,
            partitioner=partitioner,
            custom_getter=custom_getter,
            old_name_scope=scope,
            dtype=dtype) as vs:
          yield vs


# pylint: disable=g-doc-return-or-yield
@contextlib.contextmanager
def variable_op_scope(values,
                      name_or_scope,
                      default_name=None,
                      initializer=None,
                      regularizer=None,
                      caching_device=None,
                      partitioner=None,
                      custom_getter=None,
                      reuse=None,
                      dtype=None):
  """Deprecated: context manager for defining an op that creates variables."""
  logging.warn("tf.variable_op_scope(values, name, default_name) is deprecated,"
               " use tf.variable_scope(name, default_name, values)")
  with variable_scope(name_or_scope,
                      default_name=default_name,
                      values=values,
                      initializer=initializer,
                      regularizer=regularizer,
                      caching_device=caching_device,
                      partitioner=partitioner,
                      custom_getter=custom_getter,
                      reuse=reuse,
                      dtype=dtype) as scope:
    yield scope


def _compute_slice_dim_and_shape(full_shape, slicing):
  """Computes which dimension is being sliced and the typical slice shape."""

  slice_shape = [0] * len(full_shape)
  slice_dim = None
  for dim, num_slices in enumerate(slicing):
    dim_size = full_shape[dim]
    if num_slices <= 0 or dim_size < num_slices:
      raise ValueError("Cannot create %d slices for size %d. shape: %s, "
                       "slicing: %s" %
                       (num_slices, full_shape[dim], full_shape, slicing))
    if num_slices == 1:
      # Not slicing in this dimension.
      slice_shape[dim] = dim_size
    elif slice_dim is not None:
      # We only support slicing along one of the dimensions.
      raise ValueError("Can only slice a variable along one dimension: "
                       "shape: %s, slicing: %s" % (full_shape, slicing))
    else:
      # Note: We will add any extras onto the last slice, later.
      slice_dim = dim
      slice_shape[dim] = dim_size // num_slices

  # Degenerate case: If "slicing" was all ones, pretend we are slicing along
  # the first dimension.
  if slice_dim is None:
    slice_dim = 0
  return slice_dim, slice_shape
