# Copyright 2015 Google Inc. All Rights Reserved.
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
from tensorflow.contrib.framework.python.ops import add_arg_scope as contrib_add_arg_scope
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


__all__ = ['add_model_variable',
           'assert_global_step',
           'assert_or_get_global_step',
           'create_global_step',
           'get_global_step',
           'get_or_create_global_step',
           'get_local_variables',
           'get_model_variables',
           'get_unique_variable',
           'get_variables_by_name',
           'get_variables_by_suffix',
           'get_variables_to_restore',
           'get_variables',
           'local_variable',
           'model_variable',
           'variable',
           'VariableDeviceChooser']


def assert_global_step(global_step_tensor):
  """Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

  Args:
    global_step_tensor: `Tensor` to test.
  """
  if not (isinstance(global_step_tensor, variables.Variable) or
          isinstance(global_step_tensor, ops.Tensor)):
    raise TypeError('Existing "global_step" must be a Variable or Tensor.')

  if not global_step_tensor.dtype.base_dtype.is_integer:
    raise TypeError(
        'Existing "global_step" does not have integer type: %s' %
        global_step_tensor.dtype)

  if global_step_tensor.get_shape().ndims != 0:
    raise TypeError(
        'Existing "global_step" is not scalar: %s' %
        global_step_tensor.get_shape())


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


# TODO(ptucker): Change supervisor to use this when it's migrated to core.
def get_global_step(graph=None):
  """Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

  Args:
    graph: The graph to find the global step in. If missing, use default graph.

  Returns:
    The global step variable, or `None` if none was found.

  Raises:
    TypeError: If the global step tensor has a non-integer type, or if it is not
      a `Variable`.
  """
  graph = ops.get_default_graph() if graph is None else graph
  global_step_tensor = None
  global_step_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_STEP)
  if len(global_step_tensors) == 1:
    global_step_tensor = global_step_tensors[0]
  elif not global_step_tensors:
    try:
      global_step_tensor = graph.get_tensor_by_name('global_step:0')
    except KeyError:
      return None
  else:
    logging.error('Multiple tensors in global_step collection.')
    return None

  assert_global_step(global_step_tensor)
  return global_step_tensor


def create_global_step(graph=None):
  """Create global step tensor in graph.

  Args:
    graph: The graph in which to create the global step. If missing, use default
        graph.

  Returns:
    Global step tensor.

  Raises:
    ValueError: if global step key is already defined.
  """
  graph = ops.get_default_graph() if graph is None else graph
  if get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    collections = [ops.GraphKeys.VARIABLES, ops.GraphKeys.GLOBAL_STEP]
    return variable(ops.GraphKeys.GLOBAL_STEP, shape=[], dtype=dtypes.int64,
                    initializer=init_ops.zeros_initializer, trainable=False,
                    collections=collections)


def get_or_create_global_step(graph=None):
  """Returns and create (if necessary) the global step variable.

  Args:
    graph: The graph in which to create the global step. If missing, use default
        graph.

  Returns:
    the tensor representing the global step variable.
  """
  graph = ops.get_default_graph() if graph is None else graph
  globalstep = get_global_step(graph)
  if globalstep is None:
    globalstep = create_global_step(graph)
  return globalstep


def local_variable(initial_value, validate_shape=True, name=None):
  """Create variable and add it to `GraphKeys.LOCAL_VARIABLES` collection.

  Args:
    initial_value: See variables.Variable.__init__.
    validate_shape: See variables.Variable.__init__.
    name: See variables.Variable.__init__.
  Returns:
    New variable.
  """
  return variables.Variable(
      initial_value, trainable=False,
      collections=[ops.GraphKeys.LOCAL_VARIABLES],
      validate_shape=validate_shape, name=name)


@contrib_add_arg_scope
def variable(name, shape=None, dtype=dtypes.float32, initializer=None,
             regularizer=None, trainable=True, collections=None,
             caching_device=None, device=None):
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
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    collections: A list of collection names to which the Variable will be added.
      If None it would default to tf.GraphKeys.VARIABLES.
    caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.

  Returns:
    The created or existing variable.
  """
  collections = list(collections or [ops.GraphKeys.VARIABLES])

  # Remove duplicates
  collections = set(collections)
  with ops.device(device or ''):
    return variable_scope.get_variable(name, shape=shape, dtype=dtype,
                                       initializer=initializer,
                                       regularizer=regularizer,
                                       trainable=trainable,
                                       collections=collections,
                                       caching_device=caching_device)

# TODO(sguada) move it to ops.GraphKeys or to contrib.framework.GraphKeys
# Collection containing all the variables created using model_variables.
MODEL_VARIABLES = '_model_variables_'


@contrib_add_arg_scope
def model_variable(name, shape=None, dtype=dtypes.float32, initializer=None,
                   regularizer=None, trainable=True, collections=None,
                   caching_device=None, device=None):
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
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    collections: A list of collection names to which the Variable will be added.
      Note that the variable is always also added to the tf.GraphKeys.VARIABLES
      and MODEL_VARIABLES collections.
    caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.

  Returns:
    The created or existing variable.
  """
  collections = list(collections or [])

  # Make sure variables are added to tf.GraphKeys.VARIABLES and MODEL_VARIABLES
  collections += [ops.GraphKeys.VARIABLES, MODEL_VARIABLES]
  return variable(name, shape=shape, dtype=dtype,
                  initializer=initializer, regularizer=regularizer,
                  trainable=trainable, collections=collections,
                  caching_device=caching_device, device=device)


def add_model_variable(var):
  """Adds a variable to the MODEL_VARIABLES collection.

  Args:
    var: a variable.
  """
  if var not in ops.get_collection(MODEL_VARIABLES):
    ops.add_to_collection(MODEL_VARIABLES, var)


def get_variables(scope=None, suffix=None, collection=ops.GraphKeys.VARIABLES):
  """Gets the list of variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.
    collection: in which collection search for. Defaults to GraphKeys.VARIABLES.

  Returns:
    a list of variables in colelction with scope and suffix.
  """
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
    a list of variables in colelction with scope and suffix.
  """
  return get_variables(scope, suffix, MODEL_VARIABLES)


def get_local_variables(scope=None, suffix=None):
  """Gets the list of model variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in colelction with scope and suffix.
  """
  return get_variables(scope, suffix, ops.GraphKeys.LOCAL_VARIABLES)


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
  raise ValueError('Variable %s does not uniquely identify a variable',
                   var_op_name)


class VariableDeviceChooser(object):
  """Device chooser for variables.

  When using a parameter server it will assign them in a round-robin fashion.
  When not using a parameter server it allows GPU or CPU placement.
  """

  def __init__(self,
               num_tasks=0,
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
      device_type: Optional device type string (e.g. "CPU" or "GPU")
      device_index: int.  Optional device index.  If left
        unspecified, device represents 'any' device_index.
    """
    self._job_name = 'ps' if num_tasks > 0 else None
    self._device_type = device_type
    self._device_index = device_index
    self._num_tasks = num_tasks
    self._next_task_id = 0

  def __call__(self, op):
    device_spec = tf_device.DeviceSpec(job=self._job_name,
                                       device_type=self._device_type,
                                       device_index=self._device_index)
    if self._num_tasks > 0:
      task_id = self._next_task_id
      self._next_task_id = (self._next_task_id + 1) % self._num_tasks
      device_spec.task = task_id
    return device_spec.to_string()
