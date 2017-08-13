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

"""Variable functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

from tensorflow.contrib.framework.python.ops import add_arg_scope as contrib_add_arg_scope
from tensorflow.contrib.framework.python.ops import gen_variable_ops
from tensorflow.contrib.util import loader
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import resource_loader
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import training_util
from tensorflow.python.util.deprecation import deprecated


__all__ = ['add_model_variable',
           'assert_global_step',
           'assert_or_get_global_step',
           'assign_from_checkpoint',
           'assign_from_checkpoint_fn',
           'assign_from_values',
           'assign_from_values_fn',
           'create_global_step',
           'filter_variables',
           'get_global_step',
           'get_or_create_global_step',
           'get_local_variables',
           'get_model_variables',
           'get_trainable_variables',
           'get_unique_variable',
           'get_variables_by_name',
           'get_variables_by_suffix',
           'get_variable_full_name',
           'get_variables_to_restore',
           'get_variables',
           'local_variable',
           'model_variable',
           'variable',
           'VariableDeviceChooser',
           'zero_initializer']


def zero_initializer(ref, use_locking=True, name="zero_initializer"):
  """Initialize 'ref' with all zeros, ref tensor should be uninitialized.
  If already initialized, you will get ValueError. This op is intended to
  save memory during initialization.
  Args:
    ref: ref of the tensor need to be zero initialized.
    name: optional name for this operation.
  Returns:
    ref that initialized.
  Raises:
    ValueError: If ref tensor is initialized.
  """
  loader.load_op_library(
      resource_loader.get_path_to_datafile("_variable_ops.so"))
  return gen_variable_ops.zero_initializer(ref, name=name)

@deprecated(None, "Please switch to tf.train.assert_global_step")
def assert_global_step(global_step_tensor):
  training_util.assert_global_step(global_step_tensor)


def assert_or_get_global_step(graph=None, global_step_tensor=None):
  """Verifies that a global step tensor is valid or gets one if None is given.

  If `global_step_tensor` is not None, check that it is a valid global step
  tensor (using `assert_global_step`). Otherwise find a global step tensor using
  `get_global_step` and return it.

  Args:
    graph: The graph to find the global step tensor for.
    global_step_tensor: The tensor to check for suitability as a global step.
      If None is given (the default), find a global step tensor.

  Returns:
    A tensor suitable as a global step, or `None` if none was provided and none
    was found.
  """
  if global_step_tensor is None:
    # Get the global step tensor the same way the supervisor would.
    global_step_tensor = get_global_step(graph)
  else:
    assert_global_step(global_step_tensor)
  return global_step_tensor

@deprecated(None, "Please switch to tf.train.get_global_step")
def get_global_step(graph=None):
  return training_util.get_global_step(graph)

@deprecated(None, "Please switch to tf.train.create_global_step")
def create_global_step(graph=None):
  """Create global step tensor in graph.

  This API is deprecated. Use core framework training version instead.

  Args:
    graph: The graph in which to create the global step tensor. If missing,
      use default graph.

  Returns:
    Global step tensor.

  Raises:
    ValueError: if global step tensor is already defined.
  """
  return training_util.create_global_step(graph)

@deprecated(None, "Please switch to tf.train.get_or_create_global_step")
def get_or_create_global_step(graph=None):
  """Returns and create (if necessary) the global step tensor.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    The global step tensor.
  """
  return training_util.get_or_create_global_step(graph)


def local_variable(initial_value, validate_shape=True, name=None):
  """Create variable and add it to `GraphKeys.LOCAL_VARIABLES` collection.

  Args:
    initial_value: See variables.Variable.__init__.
    validate_shape: See variables.Variable.__init__.
    name: See variables.Variable.__init__.
  Returns:
    New variable.
  """
  return variable_scope.variable(
      initial_value, trainable=False,
      collections=[ops.GraphKeys.LOCAL_VARIABLES],
      validate_shape=validate_shape, name=name)


@contrib_add_arg_scope
def variable(name, shape=None, dtype=None, initializer=None,
             regularizer=None, trainable=True, collections=None,
             caching_device=None, device=None,
             partitioner=None, custom_getter=None, use_resource=None):
  """Gets an existing variable with these parameters or creates a new one.

  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of
        applying it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      If None it would default to `tf.GraphKeys.GLOBAL_VARIABLES`.
    caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal
      get_variable method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.

  Returns:
    The created or existing variable.
  """
  collections = list(collections if collections is not None
                     else [ops.GraphKeys.GLOBAL_VARIABLES])

  # Remove duplicates
  collections = set(collections)
  getter = variable_scope.get_variable
  if custom_getter is not None:
    getter = functools.partial(custom_getter,
                               reuse=variable_scope.get_variable_scope().reuse)
  with ops.device(device or ''):
    return getter(name, shape=shape, dtype=dtype,
                  initializer=initializer,
                  regularizer=regularizer,
                  trainable=trainable,
                  collections=collections,
                  caching_device=caching_device,
                  partitioner=partitioner,
                  use_resource=use_resource)


@contrib_add_arg_scope
def model_variable(name, shape=None, dtype=dtypes.float32, initializer=None,
                   regularizer=None, trainable=True, collections=None,
                   caching_device=None, device=None, partitioner=None,
                   custom_getter=None, use_resource=None):
  """Gets an existing model variable with these parameters or creates a new one.

  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of
        applying it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      Note that the variable is always also added to the
      `GraphKeys.GLOBAL_VARIABLES` and `GraphKeys.MODEL_VARIABLES` collections.
    caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal
      get_variable method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.

  Returns:
    The created or existing variable.
  """
  collections = list(collections or [])
  collections += [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
  var = variable(name, shape=shape, dtype=dtype,
                 initializer=initializer, regularizer=regularizer,
                 trainable=trainable, collections=collections,
                 caching_device=caching_device, device=device,
                 partitioner=partitioner, custom_getter=custom_getter,
                 use_resource=use_resource)
  return var


def add_model_variable(var):
  """Adds a variable to the `GraphKeys.MODEL_VARIABLES` collection.

  Args:
    var: a variable.
  """
  if var not in ops.get_collection(ops.GraphKeys.MODEL_VARIABLES):
    ops.add_to_collection(ops.GraphKeys.MODEL_VARIABLES, var)


def get_variables(scope=None, suffix=None,
                  collection=ops.GraphKeys.GLOBAL_VARIABLES):
  """Gets the list of variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return. Can be a
      variable scope or a string.
    suffix: an optional suffix for filtering the variables to return.
    collection: in which collection search for. Defaults to
      `GraphKeys.GLOBAL_VARIABLES`.

  Returns:
    a list of variables in collection with scope and suffix.
  """
  if isinstance(scope, variable_scope.VariableScope):
    scope = scope.name
  if suffix is not None:
    if ':' not in suffix:
      suffix += ':'
    scope = (scope or '') + '.*' + suffix
  return ops.get_collection(collection, scope)


def get_model_variables(scope=None, suffix=None):
  """Gets the list of model variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in collection with scope and suffix.
  """
  return get_variables(scope, suffix, ops.GraphKeys.MODEL_VARIABLES)


def get_local_variables(scope=None, suffix=None):
  """Gets the list of local variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in collection with scope and suffix.
  """
  return get_variables(scope, suffix, ops.GraphKeys.LOCAL_VARIABLES)


def get_trainable_variables(scope=None, suffix=None):
  """Gets the list of trainable variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in the trainable collection with scope and suffix.
  """
  return get_variables(scope, suffix, ops.GraphKeys.TRAINABLE_VARIABLES)


def get_variables_to_restore(include=None, exclude=None):
  """Gets the list of the variables to restore.

  Args:
    include: an optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to include. None would include all
      the variables.
    exclude: an optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to exclude. None it would not
      exclude any.

  Returns:
    a list of variables to restore.

  Raises:
    TypeError: include or exclude is provided but is not a list or a tuple.
  """
  if include is None:
    # Include all variables.
    vars_to_include = get_variables()
  else:
    if not isinstance(include, (list, tuple)):
      raise TypeError('include is provided but is not a list or a tuple.')
    vars_to_include = []
    for scope in include:
      vars_to_include += get_variables(scope)
  vars_to_exclude = set()
  if exclude is not None:
    if not isinstance(exclude, (list, tuple)):
      raise TypeError('exclude is provided but is not a list or a tuple.')
    for scope in exclude:
      vars_to_exclude |= set(get_variables(scope))
  # Exclude the variables in vars_to_exclude
  return [v for v in vars_to_include if v not in vars_to_exclude]


def get_variables_by_suffix(suffix, scope=None):
  """Gets the list of variables that end with the given suffix.

  Args:
    suffix: suffix for filtering the variables to return.
    scope: an optional scope for filtering the variables to return.

  Returns:
    a copied list of variables with the given name and prefix.
  """
  return get_variables(scope=scope, suffix=suffix)


def get_variables_by_name(given_name, scope=None):
  """Gets the list of variables that were given that name.

  Args:
    given_name: name given to the variable without any scope.
    scope: an optional scope for filtering the variables to return.

  Returns:
    a copied list of variables with the given name and scope.
  """
  suffix = '/' + given_name + ':|^' + given_name + ':'
  return get_variables(scope=scope, suffix=suffix)


def get_unique_variable(var_op_name):
  """Gets the variable uniquely identified by that var_op_name.

  Args:
    var_op_name: the full name of the variable op, including the scope.

  Returns:
    a tensorflow variable.

  Raises:
    ValueError: if no variable uniquely identified by the name exists.
  """
  candidates = get_variables(scope=var_op_name)
  if not candidates:
    raise ValueError('Couldnt find variable %s' % var_op_name)

  for candidate in candidates:
    if candidate.op.name == var_op_name:
      return candidate
  raise ValueError('Variable %s does not uniquely identify a variable' %
                   var_op_name)


def assign_from_values(var_names_to_values):
  """Creates an assignment operation from a given mapping.

  This function provides a mechanism for performing assignment of variables
  to values in a way that does not fill the graph with large assignment values.

  Args:
    var_names_to_values: A map from variable names to values.

  Returns:
    assign_op: An `Operation` that assigns each of the given variables to the
      requested values.
    feed_dict: The feed dictionary to use when evaluating `assign_op`.

  Raises:
    ValueError: if any of the given variable names were not found.
  """
  feed_dict = {}
  assign_ops = []

  for var_name in var_names_to_values:
    var_value = var_names_to_values[var_name]
    var = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, var_name)
    if not var:
      raise ValueError('Variable %s wasn\'t found' % var_name)
    elif len(var) > 1:
      # tf.get_collection is just a filter on the prefix: find the exact match:
      found = False
      for v in var:
        if v.op.name == var_name:
          var = v
          found = True
          break

      if not found:
        raise ValueError('Variable %s doesn\'t uniquely identify a variable' %
                         var_name)
    else:
      var = var[0]

    # TODO(nsilberman): ensure placeholder and assign are on the same device.
    # Assign a placeholder to the value that will be filled later.
    placeholder_name = 'placeholder/' + var.op.name
    placeholder_value = array_ops.placeholder(
        dtype=var.dtype.base_dtype,
        shape=var.get_shape(),
        name=placeholder_name)
    assign_ops.append(var.assign(placeholder_value))

    feed_dict[placeholder_value] = var_value.reshape(var.get_shape())

  assign_op = control_flow_ops.group(*assign_ops)
  return assign_op, feed_dict


def assign_from_values_fn(var_names_to_values):
  """Returns a function that assigns specific variables from the given values.

  This function provides a mechanism for performing assignment of variables
  to values in a way that does not fill the graph with large assignment values.

  Args:
    var_names_to_values: A map from variable names to values.

  Returns:
    A function that takes a single argument, a `tf.Session`, that applies the
    assignment operation.

  Raises:
    ValueError: if any of the given variable names were not found.
  """
  assign_op, feed_dict = assign_from_values(var_names_to_values)
  def callback(session):
    return session.run(assign_op, feed_dict)
  return callback


# pylint: disable=protected-access
# Currently variable_scope doesn't provide very good APIs to access
# all variables under scope and retrieve and check existing scopes.
def get_variable_full_name(var):
  """Returns the full name of a variable.

  For normal Variables, this is the same as the var.op.name.  For
  sliced or PartitionedVariables, this name is the same for all the
  slices/partitions. In both cases, this is normally the name used in
  a checkpoint file.

  Args:
    var: A `Variable` object.

  Returns:
    A string that is the full name.
  """
  if var._save_slice_info:
    return var._save_slice_info.full_name
  else:
    return var.op.name


# TODO(nsilberman): add flag to load exponential moving averages instead
#
# TODO(sguada): Update docs in slim/g3doc/index.md to describe
# the new feature where the var_list dictionary can have values that
# are each a list of Variables.
def assign_from_checkpoint(model_path, var_list, ignore_missing_vars=False):
  """Creates an operation to assign specific variables from a checkpoint.

  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
        use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of (possibly partitioned) `Variable` objects
        or a dictionary mapping names in the checkpoint to the
        corresponding variables or list of variables to initialize
        from that checkpoint value. For partitioned Variables, the
        name in the checkpoint must be the full variable, not the
        name of the partitioned variable, eg. "my_var" rather than
        "my_var/part_4". If empty, returns no_op(), {}.
    ignore_missing_vars: Boolean, if True ignore variables missing in the
        checkpoint with a warning instead of failing.

  Returns:
    the restore_op and the feed_dict that need to be run to restore var_list.

  Raises:
    ValueError: If `ignore_missing_vars` is False and the checkpoint specified
        at `model_path` is missing one of the variables in `var_list`.
  """
  # Normalize var_list into a dictionary mapping names in the
  # checkpoint to the list of variables to initialize from that
  # checkpoint variable. Sliced (including partitioned) variables will
  # end up under the same key.
  grouped_vars = {}
  if isinstance(var_list, (tuple, list)):
    for var in var_list:
      ckpt_name = get_variable_full_name(var)
      if ckpt_name not in grouped_vars:
        grouped_vars[ckpt_name] = []
      grouped_vars[ckpt_name].append(var)

  else:
    for ckpt_name, value in var_list.items():
      if isinstance(value, (tuple, list)):
        grouped_vars[ckpt_name] = value
      else:
        grouped_vars[ckpt_name] = [value]

  # Read each checkpoint entry. Create a placeholder variable and
  # add the (possibly sliced) data from the checkpoint to the feed_dict.
  reader = pywrap_tensorflow.NewCheckpointReader(model_path)
  feed_dict = {}
  assign_ops = []
  for ckpt_name in grouped_vars:
    if not reader.has_tensor(ckpt_name):
      log_str = 'Checkpoint is missing variable [%s]' % ckpt_name
      if ignore_missing_vars:
        logging.warning(log_str)
        continue
      else:
        raise ValueError(log_str)
    ckpt_value = reader.get_tensor(ckpt_name)

    for var in grouped_vars[ckpt_name]:
      placeholder_tensor = array_ops.placeholder(
          dtype=var.dtype.base_dtype,
          shape=var.get_shape(),
          name='placeholder/' + var.op.name)
      assign_ops.append(var.assign(placeholder_tensor))

      if not var._save_slice_info:
        if var.get_shape() != ckpt_value.shape:
          raise ValueError(
              'Total size of new array must be unchanged for %s '
              'lh_shape: [%s], rh_shape: [%s]'
              % (ckpt_name, str(ckpt_value.shape), str(var.get_shape())))

        feed_dict[placeholder_tensor] = ckpt_value.reshape(ckpt_value.shape)
      else:
        slice_dims = zip(var._save_slice_info.var_offset,
                         var._save_slice_info.var_shape)
        slice_dims = [(start, start + size) for (start, size) in slice_dims]
        slice_dims = [slice(*x) for x in slice_dims]
        slice_value = ckpt_value[slice_dims]
        slice_value = slice_value.reshape(var._save_slice_info.var_shape)
        feed_dict[placeholder_tensor] = slice_value

  assign_op = control_flow_ops.group(*assign_ops)
  return assign_op, feed_dict
# pylint: enable=protected-access


def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False):
  """Returns a function that assigns specific variables from a checkpoint.

  If ignore_missing_vars is True and no variables are found in the checkpoint
  it returns None.

  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
        use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of `Variable` objects or a dictionary mapping names in the
        checkpoint to the corresponding variables to initialize. If empty or
        `None`, it would return `no_op(), None`.
    ignore_missing_vars: Boolean, if True it would ignore variables missing in
        the checkpoint with a warning instead of failing.
    reshape_variables: Boolean, if True it would automatically reshape variables
        which are of different shape then the ones stored in the checkpoint but
        which have the same number of elements.

  Returns:
    A function that takes a single argument, a `tf.Session`, that applies the
    assignment operation. If no matching variables were found in the checkpoint
    then `None` is returned.

  Raises:
    ValueError: If var_list is empty.
  """
  if not var_list:
    raise ValueError('var_list cannot be empty')
  if ignore_missing_vars:
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
      var_dict = var_list
    else:
      var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:
      if reader.has_tensor(var):
        available_vars[var] = var_dict[var]
      else:
        logging.warning(
            'Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
  if var_list:
    saver = tf_saver.Saver(var_list, reshape=reshape_variables)
    def callback(session):
      saver.restore(session, model_path)
    return callback
  else:
    logging.warning('No Variables to restore')
    return None


class VariableDeviceChooser(object):
  """Device chooser for variables.

  When using a parameter server it will assign them in a round-robin fashion.
  When not using a parameter server it allows GPU or CPU placement.
  """

  def __init__(self,
               num_tasks=0,
               job_name='ps',
               device_type='CPU',
               device_index=0):
    """Initialize VariableDeviceChooser.

    Usage:
      To use with 2 parameter servers:
        VariableDeviceChooser(2)

      To use without parameter servers:
        VariableDeviceChooser()
        VariableDeviceChooser(device_type='GPU') # For GPU placement

    Args:
      num_tasks: number of tasks.
      job_name: String, a name for the parameter server job.
      device_type: Optional device type string (e.g. "CPU" or "GPU")
      device_index: int.  Optional device index.  If left
        unspecified, device represents 'any' device_index.
    """
    self._job_name = job_name
    self._device_type = device_type
    self._device_index = device_index
    self._num_tasks = num_tasks
    self._next_task_id = 0

  def __call__(self, op):
    device_spec = tf_device.DeviceSpec(device_type=self._device_type,
                                       device_index=self._device_index)
    if self._num_tasks > 0:
      task_id = self._next_task_id
      self._next_task_id = (self._next_task_id + 1) % self._num_tasks
      device_spec.job = self._job_name
      device_spec.task = task_id
    return device_spec.to_string()


def filter_variables(var_list, include_patterns=None, exclude_patterns=None,
                     reg_search=True):
  """Filter a list of variables using regular expressions.

  First includes variables according to the list of include_patterns.
  Afterwards, eliminates variables according to the list of exclude_patterns.

  For example, one can obtain a list of variables with the weights of all
  convolutional layers (depending on the network definition) by:

  ```python
  variables = tf.contrib.framework.get_model_variables()
  conv_weight_variables = tf.contrib.framework.filter_variables(
      variables,
      include_patterns=['Conv'],
      exclude_patterns=['biases', 'Logits'])
  ```

  Args:
    var_list: list of variables.
    include_patterns: list of regular expressions to include. Defaults to None,
        which means all variables are selected according to the include rules.
        A variable is included if it matches any of the include_patterns.
    exclude_patterns: list of regular expressions to exclude. Defaults to None,
        which means all variables are selected according to the exclude rules.
        A variable is excluded if it matches any of the exclude_patterns.
    reg_search: boolean. If True (default), performs re.search to find matches
        (i.e. pattern can match any substring of the variable name). If False,
        performs re.match (i.e. regexp should match from the beginning of the
        variable name).

  Returns:
    filtered list of variables.
  """
  if reg_search:
    reg_exp_func = re.search
  else:
    reg_exp_func = re.match

  # First include variables.
  if include_patterns is None:
    included_variables = list(var_list)
  else:
    included_variables = []
    for var in var_list:
      if any(reg_exp_func(ptrn, var.name) for ptrn in include_patterns):
        included_variables.append(var)

  # Afterwards, exclude variables.
  if exclude_patterns is None:
    filtered_variables = included_variables
  else:
    filtered_variables = []
    for var in included_variables:
      if not any(reg_exp_func(ptrn, var.name) for ptrn in exclude_patterns):
        filtered_variables.append(var)

  return filtered_variables
