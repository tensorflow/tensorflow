# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tools to work with checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "load_checkpoint", "load_variable", "list_variables", "init_from_checkpoint"
]


@tf_export("train.load_checkpoint")
def load_checkpoint(ckpt_dir_or_file):
  """Returns `CheckpointReader` for checkpoint found in `ckpt_dir_or_file`.

  If `ckpt_dir_or_file` resolves to a directory with multiple checkpoints,
  reader for the latest checkpoint is returned.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint
      file.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: If `ckpt_dir_or_file` resolves to a directory with no
      checkpoints.
  """
  filename = _get_checkpoint_filename(ckpt_dir_or_file)
  if filename is None:
    raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                     "given directory %s" % ckpt_dir_or_file)
  return pywrap_tensorflow.NewCheckpointReader(filename)


@tf_export("train.load_variable")
def load_variable(ckpt_dir_or_file, name):
  """Returns the tensor value of the given variable in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    name: Name of the variable to return.

  Returns:
    A numpy `ndarray` with a copy of the value of this variable.
  """
  # TODO(b/29227106): Fix this in the right place and remove this.
  if name.endswith(":0"):
    name = name[:-2]
  reader = load_checkpoint(ckpt_dir_or_file)
  return reader.get_tensor(name)


@tf_export("train.list_variables")
def list_variables(ckpt_dir_or_file):
  """Returns list of all variables in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(name, shape)`.
  """
  reader = load_checkpoint(ckpt_dir_or_file)
  variable_map = reader.get_variable_to_shape_map()
  names = sorted(variable_map.keys())
  result = []
  for name in names:
    result.append((name, variable_map[name]))
  return result


@tf_export("train.init_from_checkpoint")
def init_from_checkpoint(ckpt_dir_or_file, assignment_map):
  """Initializes current variables with tensors loaded from given checkpoint.

  Note: This overrides default initialization ops of specified variables and
  redefines dtype.

  Assignment map supports following syntax:

  * `'checkpoint_scope_name/': 'scope_name/'` - will load all variables in
    current `scope_name` from `checkpoint_scope_name` with matching tensor
    names.
  * `'checkpoint_scope_name/some_other_variable': 'scope_name/variable_name'` -
    will initialize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.
  * `'scope_variable_name': variable` - will initialize given `tf.Variable`
    object with tensor 'scope_variable_name' from the checkpoint.
  * `'scope_variable_name': list(variable)` - will initialize list of
    partitioned variables with tensor 'scope_variable_name' from the checkpoint.
  * `'/': 'scope_name/'` - will load all variables in current `scope_name` from
    checkpoint's root (e.g. no scope).

  Supports loading into partitioned variables, which are represented as
  `'<variable>/part_<part #>'`.

  Example:

  ```python

  # Say, '/tmp/model.ckpt' has the following tensors:
  #  -- name='old_scope_1/var1', shape=[20, 2]
  #  -- name='old_scope_1/var2', shape=[50, 4]
  #  -- name='old_scope_2/var3', shape=[100, 100]

  # Create new model's variables
  with tf.variable_scope('new_scope_1'):
    var1 = tf.get_variable('var1', shape=[20, 2],
                           initializer=tf.zeros_initializer())
  with tf.variable_scope('new_scope_2'):
    var2 = tf.get_variable('var2', shape=[50, 4],
                           initializer=tf.zeros_initializer())
    # Partition into 5 variables along the first axis.
    var3 = tf.get_variable(name='var3', shape=[100, 100],
                           initializer=tf.zeros_initializer(),
                           partitioner=lambda shape, dtype: [5, 1])

  # Initialize all variables in `new_scope_1` from `old_scope_1`.
  init_from_checkpoint('/tmp/model.ckpt', {'old_scope_1/', 'new_scope_1'})

  # Use names to specify which variables to initialize from checkpoint.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': 'new_scope_1/var1',
                        'old_scope_1/var2': 'new_scope_2/var2'})

  # Or use tf.Variable objects to identify what to initialize.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': var1,
                        'old_scope_1/var2': var2})

  # Initialize partitioned variables using variable's name
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': 'new_scope_2/var3'})

  # Or specify the list of tf.Variable objects.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': var3._get_variable_list()})

  ```

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    assignment_map: Dict, where keys are names of the variables in the
      checkpoint and values are current variables or names of current variables
      (in default graph).

  Raises:
    tf.errors.OpError: If missing checkpoints or tensors in checkpoints.
    ValueError: If missing variables in current graph.
  """
  ckpt_file = _get_checkpoint_filename(ckpt_dir_or_file)
  reader = load_checkpoint(ckpt_dir_or_file)
  variable_map = reader.get_variable_to_shape_map()
  for tensor_name_in_ckpt, current_var_or_name in sorted(
      six.iteritems(assignment_map)):
    var = None
    # Check if this is Variable object or list of Variable objects (in case of
    # partitioned variables).
    is_var = lambda x: isinstance(x, variables.Variable)
    if is_var(current_var_or_name) or (
        isinstance(current_var_or_name, list)
        and all(is_var(v) for v in current_var_or_name)):
      var = current_var_or_name
    else:
      store_vars = vs._get_default_variable_store()._vars  # pylint:disable=protected-access
      # Check if this variable is in var_store.
      var = store_vars.get(current_var_or_name, None)
      # Also check if variable is partitioned as list.
      if var is None:
        var = _collect_partitioned_variable(current_var_or_name, store_vars)
    if var is not None:
      # If 1 to 1 mapping was provided, find variable in the checkpoint.
      if tensor_name_in_ckpt not in variable_map:
        raise ValueError("Tensor %s is not found in %s checkpoint %s" % (
            tensor_name_in_ckpt, ckpt_dir_or_file, variable_map
        ))
      if is_var(var):
        # Additional at-call-time checks.
        if not var.get_shape().is_compatible_with(
            variable_map[tensor_name_in_ckpt]):
          raise ValueError(
              "Shape of variable %s (%s) doesn't match with shape of "
              "tensor %s (%s) from checkpoint reader." % (
                  var.name, str(var.get_shape()),
                  tensor_name_in_ckpt, str(variable_map[tensor_name_in_ckpt])
              ))
        var_name = var.name
      else:
        var_name = ",".join([v.name for v in var])
      _set_variable_or_list_initializer(var, ckpt_file, tensor_name_in_ckpt)
      logging.info("Initialize variable %s from checkpoint %s with %s",
                   var_name, ckpt_dir_or_file, tensor_name_in_ckpt)
    else:
      scopes = ""
      # TODO(vihanjain): Support list of 'current_var_or_name' here.
      if "/" in current_var_or_name:
        scopes = current_var_or_name[:current_var_or_name.rindex("/")]
      if not tensor_name_in_ckpt.endswith("/"):
        raise ValueError(
            "Assignment map with scope only name {} should map to scope only "
            "{}. Should be 'scope/': 'other_scope/'.".format(
                scopes, tensor_name_in_ckpt))
      # If scope to scope mapping was provided, find all variables in the scope
      # and create variable to variable mapping.
      scope_variables = set()
      for var_name in store_vars:
        if not scopes or var_name.startswith(scopes + "/"):
          # Consume /part_ if partitioned variable.
          if "/part_" in var_name:
            var_name = var_name[:var_name.index("/part_")]
          scope_variables.add(var_name)
      for var_name in sorted(scope_variables):
        # Lookup name with specified prefix and suffix from current variable.
        # If tensor_name given is '/' (root), don't use it for full name.
        full_tensor_name = var_name[len(scopes):]
        if current_var_or_name != "/":
          full_tensor_name = full_tensor_name[1:]
        if tensor_name_in_ckpt != "/":
          full_tensor_name = tensor_name_in_ckpt + full_tensor_name
        # Remove trailing '/', if any, in the full_tensor_name
        if full_tensor_name.endswith("/"):
          full_tensor_name = full_tensor_name[:-1]
        if full_tensor_name not in variable_map:
          raise ValueError(
              "Tensor %s (%s in %s) is not found in %s checkpoint" % (
                  full_tensor_name, var_name[len(scopes) + 1:],
                  tensor_name_in_ckpt, ckpt_dir_or_file
              ))
        var = store_vars.get(var_name, None)
        if var is None:
          var = _collect_partitioned_variable(var_name, store_vars)
        _set_variable_or_list_initializer(var, ckpt_file, full_tensor_name)
        logging.info("Initialize variable %s from checkpoint %s with %s",
                     var_name, ckpt_dir_or_file, full_tensor_name)


def _get_checkpoint_filename(ckpt_dir_or_file):
  """Returns checkpoint filename given directory or specific checkpoint file."""
  if gfile.IsDirectory(ckpt_dir_or_file):
    return saver.latest_checkpoint(ckpt_dir_or_file)
  return ckpt_dir_or_file


def _set_checkpoint_initializer(variable,
                                ckpt_file,
                                tensor_name,
                                slice_spec,
                                name="checkpoint_initializer"):
  """Overrides given variable's initialization op.

  Sets variable initializer to assign op that initializes variable from tensor's
  value in the checkpoint.

  Args:
    variable: `tf.Variable` object.
    ckpt_file: string, full path of the checkpoint.
    tensor_name: Name of the tensor to load from the checkpoint.
    slice_spec: Slice specification for loading partitioned tensors.
    name: Name of the operation.
  """
  base_type = variable.dtype.base_dtype
  # Do not colocate with variable since RestoreV2 op only runs on CPU and
  # colocation will force variable (and other ops that colocate with variable)
  # to be on CPU as well. It is okay to place the variable's initializer op on
  # CPU since it will only be run once at the start.
  with ops.device(variable.device), ops.device("/cpu:0"):
    restore_op = io_ops.restore_v2(
        ckpt_file, [tensor_name], [slice_spec], [base_type], name=name)[0]
    if isinstance(variable, resource_variable_ops.ResourceVariable):
      init_op = variable.assign(restore_op, read_value=False)
    else:
      init_op = state_ops.assign(variable, restore_op)
    variable._initializer_op = init_op  # pylint:disable=protected-access
    restore_op.set_shape(variable.shape)
    variable._initial_value = restore_op  # pylint:disable=protected-access


def _set_variable_or_list_initializer(variable_or_list, ckpt_file,
                                      tensor_name):
  """Overrides initialization op of given variable or list of variables.

  Calls `_set_checkpoint_initializer` for each variable in the given list of
  variables.

  Args:
    variable_or_list: `tf.Variable` object or a list of `tf.Variable` objects.
    ckpt_file: string, full path of the checkpoint.
    tensor_name: Name of the tensor to load from the checkpoint.

  Raises:
    ValueError: if all objects in `variable_or_list` are not partitions of the
      same large variable.
  """
  if isinstance(variable_or_list, (list, tuple)):
    # A set of slices.
    slice_name = None
    for v in variable_or_list:
      slice_info = v._save_slice_info  # pylint:disable=protected-access
      if slice_name is None:
        slice_name = slice_info.full_name
      elif slice_name != slice_info.full_name:
        raise ValueError("Slices must all be from the same tensor: %s != %s" %
                         (slice_name, slice_info.full_name))
      _set_checkpoint_initializer(v, ckpt_file, tensor_name, slice_info.spec)
  else:
    _set_checkpoint_initializer(variable_or_list, ckpt_file, tensor_name, "")


def _collect_partitioned_variable(name, all_vars):
  """Returns list of `tf.Variable` that comprise the partitioned variable."""
  if name + "/part_0" in all_vars:
    var = []
    i = 0
    while name + "/part_%d" % i in all_vars:
      var.append(all_vars[name + "/part_%d" % i])
      i += 1
    return var
  return None
