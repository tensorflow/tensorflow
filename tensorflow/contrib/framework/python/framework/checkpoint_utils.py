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

from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training as train

__all__ = [
    "load_checkpoint",
    "load_variable",
    "list_variables",
    "init_from_checkpoint"]


def _get_checkpoint_filename(filepattern):
  """Returns checkpoint filename given directory or specific filepattern."""
  if gfile.IsDirectory(filepattern):
    return checkpoint_management.latest_checkpoint(filepattern)
  return filepattern


def load_checkpoint(filepattern):
  """Returns CheckpointReader for latest checkpoint.

  Args:
    filepattern: Directory with checkpoints file or path to checkpoint.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: if checkpoint_dir doesn't have 'checkpoint' file or checkpoints.
  """
  filename = _get_checkpoint_filename(filepattern)
  if filename is None:
    raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                     "given directory %s" % filepattern)
  return train.NewCheckpointReader(filename)


def load_variable(checkpoint_dir, name):
  """Returns a Tensor with the contents of the given variable in the checkpoint.

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    name: Name of the tensor to return.

  Returns:
    `Tensor` object.
  """
  # TODO(b/29227106): Fix this in the right place and remove this.
  if name.endswith(":0"):
    name = name[:-2]
  reader = load_checkpoint(checkpoint_dir)
  return reader.get_tensor(name)


def list_variables(checkpoint_dir):
  """Returns list of all variables in the latest checkpoint.

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(name, shape)`.
  """
  reader = load_checkpoint(checkpoint_dir)
  variable_map = reader.get_variable_to_shape_map()
  names = sorted(variable_map.keys())
  result = []
  for name in names:
    result.append((name, variable_map[name]))
  return result


# pylint: disable=protected-access
# Currently variable_scope doesn't provide very good APIs to access
# all variables under scope and retrieve and check existing scopes.
# TODO(ipolosukhin): Refactor variable_scope module to provide nicer APIs.


def _set_checkpoint_initializer(variable, file_pattern, tensor_name, slice_spec,
                                name="checkpoint_initializer"):
  """Sets variable initializer to assign op form value in checkpoint's tensor.

  Args:
    variable: `Variable` object.
    file_pattern: string, where to load checkpoints from.
    tensor_name: Name of the `Tensor` to load from checkpoint reader.
    slice_spec: Slice specification for loading partitioned variables.
    name: Name of the operation.
  """
  base_type = variable.dtype.base_dtype
  with ops.device(variable.device), ops.device("/cpu:0"):
    restore_op = io_ops.restore_v2(
        file_pattern, [tensor_name], [slice_spec], [base_type], name=name)[0]
    variable._initializer_op = state_ops.assign(variable, restore_op)


def _set_variable_or_list_initializer(variable_or_list, file_pattern,
                                      tensor_name):
  if isinstance(variable_or_list, (list, tuple)):
    # A set of slices.
    slice_name = None
    for v in variable_or_list:
      if slice_name is None:
        slice_name = v._save_slice_info.full_name
      elif slice_name != v._save_slice_info.full_name:
        raise ValueError("Slices must all be from the same tensor: %s != %s" %
                         (slice_name, v._save_slice_info.full_name))
      _set_checkpoint_initializer(v, file_pattern, tensor_name,
                                  v._save_slice_info.spec)
  else:
    _set_checkpoint_initializer(variable_or_list, file_pattern, tensor_name, "")


def _collect_partitioned_variable(name, var_scope):
  if name + "/part_0" in var_scope._vars:
    var = []
    i = 0
    while name + "/part_%d" % i in var_scope._vars:
      var.append(var_scope._vars[name + "/part_%d" % i])
      i += 1
    return var
  return None


def init_from_checkpoint(checkpoint_dir, assignment_map):
  """Using assignment map initializes current variables with loaded tensors.

  Note: This overrides default initialization ops of specified variables and
  redefines dtype.

  Assignment map supports following syntax:

  * `'checkpoint_scope_name/': 'scope_name/'` - will load all variables in
    current `scope_name` from `checkpoint_scope_name` with matching variable
    names.
  * `'checkpoint_scope_name/some_other_variable': 'scope_name/variable_name'` -
    will initialize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.
  * `'scope_variable_name': variable` - will initialize given `tf.Variable`
    object with variable from the checkpoint.
  * `'scope_variable_name': list(variable)` - will initialize list of
    partitioned variables with variable from the checkpoint.
  * `'/': 'scope_name/'` - will load all variables in current `scope_name` from
    checkpoint's root (e.g. no scope).

  Supports loading into partitioned variables, which are represented as
  `'<variable>/part_<part #>'`.

  Example:

  ```python
    # Create variables.
    with tf.compat.v1.variable_scope('test'):
      m = tf.compat.v1.get_variable('my_var')
    with tf.compat.v1.variable_scope('test2'):
      var2 = tf.compat.v1.get_variable('my_var')
    var3 = tf.compat.v1.get_variable(name="my1", shape=[100, 100],
                           partitioner=lambda shape, dtype: [5, 1])
    ...
    # Specify which variables to initialize from checkpoint.
    init_from_checkpoint(checkpoint_dir, {
      'some_var': 'test/my_var',
      'some_scope/': 'test2/'})
    ...
    # Or use `Variable` objects to identify what to initialize.
    init_from_checkpoint(checkpoint_dir, {
      'some_scope/var2': var2,
    })
    # Initialize partitioned variables
    init_from_checkpoint(checkpoint_dir, {
      'some_var_from_ckpt': 'part_var',
    })
    # Or specifying the list of `Variable` objects.
    init_from_checkpoint(checkpoint_dir, {
      'some_var_from_ckpt': var3._get_variable_list(),
    })
    ...
    # Initialize variables as usual.
    session.run(tf.get_all_variables())
  ```

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    assignment_map: Dict, where keys are names of the variables in the
      checkpoint and values are current variables or names of current variables
      (in default graph).

  Raises:
    tf.errors.OpError: If missing checkpoints or tensors in checkpoints.
    ValueError: If missing variables in current graph.
  """
  filepattern = _get_checkpoint_filename(checkpoint_dir)
  reader = load_checkpoint(checkpoint_dir)
  variable_map = reader.get_variable_to_shape_map()
  for tensor_name_in_ckpt, current_var_or_name in six.iteritems(assignment_map):
    var = None
    # Check if this is Variable object or list of Variable objects (in case of
    # partitioned variables).
    is_var = lambda x: isinstance(x, variables.Variable)
    if is_var(current_var_or_name) or (
        isinstance(current_var_or_name, list)
        and all(is_var(v) for v in current_var_or_name)):
      var = current_var_or_name
    else:
      var_scope = vs._get_default_variable_store()
      # Check if this variable is in var_store.
      var = var_scope._vars.get(current_var_or_name, None)
      # Also check if variable is partitioned as list.
      if var is None:
        var = _collect_partitioned_variable(current_var_or_name, var_scope)
    if var is not None:
      # If 1 to 1 mapping was provided, find variable in the checkpoint.
      if tensor_name_in_ckpt not in variable_map:
        raise ValueError("Tensor %s is not found in %s checkpoint %s" % (
            tensor_name_in_ckpt, checkpoint_dir, variable_map
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
      _set_variable_or_list_initializer(var, filepattern, tensor_name_in_ckpt)
      logging.info("Initialize variable %s from checkpoint %s with %s" % (
          var_name, checkpoint_dir, tensor_name_in_ckpt
      ))
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
      for var_name in var_scope._vars:
        if not scopes or var_name.startswith(scopes + "/"):
          # Consume /part_ if partitioned variable.
          if "/part_" in var_name:
            var_name = var_name[:var_name.index("/part_")]
          scope_variables.add(var_name)
      for var_name in scope_variables:
        # Lookup name with specified prefix and suffix from current variable.
        # If tensor_name given is '/' (root), don't use it for full name.
        full_tensor_name = var_name[len(scopes):]
        if current_var_or_name != "/":
          full_tensor_name = full_tensor_name[1:]
        if tensor_name_in_ckpt != "/":
          full_tensor_name = tensor_name_in_ckpt + full_tensor_name
        if full_tensor_name not in variable_map:
          raise ValueError(
              "Tensor %s (%s in %s) is not found in %s checkpoint" % (
                  full_tensor_name, var_name[len(scopes) + 1:],
                  tensor_name_in_ckpt, checkpoint_dir
              ))
        var = var_scope._vars.get(var_name, None)
        if var is None:
          var = _collect_partitioned_variable(var_name, var_scope)
        _set_variable_or_list_initializer(var, filepattern, full_tensor_name)
        logging.info("Initialize variable %s from checkpoint %s with %s" % (
            var_name, checkpoint_dir, full_tensor_name
        ))
# pylint: enable=protected-access
