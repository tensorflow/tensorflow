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
"""Tools to work with checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver
from tensorflow.python.training import training as train


def load_checkpoint(filepattern):
  """Returns CheckpointReader for latest checkpoint.

  Args:
    filepattern: Directory with checkpoints file or path to checkpoint.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: if checkpoint_dir doesn't have 'checkpoint' file or checkpoints.
  """
  if gfile.IsDirectory(filepattern):
    filename = saver.latest_checkpoint(filepattern)
    if filename is None:
      raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                       "given directory %s" % filepattern)
    return train.NewCheckpointReader(filename)
  return train.NewCheckpointReader(filepattern)


def load_variable(checkpoint_dir, name):
  """Returns a Tensor with the contents of the given variable in the checkpoint.

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    name: Name of the tensor to return.

  Returns:
    `Tensor` object.
  """
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


def _checkpoint_initializer(variable, checkpoint_reader, tensor_name):
  """Assigns variable to value that will be loaded from checkpoint's tensor.

  Args:
    variable: `Variable` object.
    checkpoint_reader: `CheckpointReader` object.
    tensor_name: Name of the `Tensor` to load from checkpoint reader.

  Returns:
    `Tensor` that returns value of `tensor_name` in checkpoint.

  Raises:
    ValueError: if shape or dtype of `variable` doesn't match with Tensor in
                checkpoint.
  """
  # Currently to avoid putting the whole tensor into the graph, this adds a
  # py_func function to the graph, that will return actual value.
  # TODO(ipolosukhin): Rewrite this as C++ op, that loads checkpoint at time.
  tensor = checkpoint_reader.get_tensor(tensor_name)
  def _tensor():
    return tensor
  if not variable.get_shape().is_compatible_with(tensor.shape):
    raise ValueError(
        "Shape of variable %s (%s) doesn't match with shape of "
        "tensor %s (%s) from checkpoint reader." % (
            variable.name, str(variable.get_shape()),
            tensor_name, str(tensor.shape)
        ))
  if not dtypes.as_dtype(tensor.dtype).is_compatible_with(variable.dtype):
    raise ValueError(
        "DType of variable %s (%s) doesn't match with dtype of "
        "tensor %s (%s) from checkpoint reader." % (
            variable.name, str(variable.dtype),
            tensor_name, str(dtypes.as_dtype(tensor.dtype))
        ))
  return state_ops.assign(
      variable, script_ops.py_func(_tensor, [], [tensor.dtype])[0])


def init_from_checkpoint(checkpoint_dir, assignment_map):
  """Using assingment map initializes current variables with loaded tensors.

  Note: This overrides default initialization ops of specified variables and
  redefines dtype.

  Assignment map supports next syntax:
    `'scope_name/': 'checkpoint_scope_name/'` - will load all variables in
      current `scope_name` from `checkpoint_scope_name` with matching variable
      names.
    `'scope_name/variable_name': 'checkpoint_scope_name/some_other_variable'` -
    will initalize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.

  Example:
  ```python
    # Create variables.
    with tf.variable_scope('test'):
      m = tf.get_variable('my_var')
    with tf.variable_scope('test2'):
      m = tf.get_variable('my_var')
    ...
    # Specify which variables to intialize from checkpoint.
    init_from_checkpoint(checkpoint_dir, {
      'test/my_var': 'some_var',
      'test2/', 'some_scope/'})
    ...
    # Initialize variables as usual.
    session.run(tf.get_all_variables())
  ```

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    assignment_map: Dict, where keys are names of current variables
                    (in default graph) and values are names of the variables
                    in the checkpoint.

  Raises:
    tf.errors.OpError: If missing checkpoints or tensors in checkpoints.
    ValueError: If missing variables in current graph.
  """
  reader = load_checkpoint(checkpoint_dir)
  variable_map = reader.get_variable_to_shape_map()
  for current_name, tensor_name in six.iteritems(assignment_map):
    scopes = ""
    if "/" in current_name:
      scopes = current_name[:current_name.rindex("/")]
      current_name = current_name[current_name.rindex("/") + 1:]
    if current_name:
      # If 1 to 1 mapping was provided, find variable in the scope.
      if tensor_name not in variable_map:
        raise ValueError("Tensor %s is not found in %s checkpoint" % (
            tensor_name, checkpoint_dir
        ))
      with vs.variable_scope(scopes, reuse=True):
        var = vs.get_variable(current_name)
        var._initializer_op = _checkpoint_initializer(var, reader, tensor_name)  # pylint: disable=protected-access
        logging.info("Initialize variable %s from checkpoint %s with %s" % (
            var.name, checkpoint_dir, tensor_name
        ))
    else:
      if not tensor_name.endswith("/"):
        raise ValueError(
            "Assignment map with scope only name (%s) "
            "should map to scope only (%s). "
            "Should be 'scope/': 'other_scope/'." % (
                scopes, tensor_name
            ))
      # If scope to scope mapping was provided, find all variables in the scope.
      # TODO(ipolosukhin): Refactor variable_scope module to provide nicer APIs.
      var_scope = vs._get_default_variable_store()  # pylint: disable=protected-access
      for var_name in var_scope._vars:  # pylint: disable=protected-access
        if var_name.startswith(scopes):
          # Lookup name with specified prefix and suffix from current variable.
          full_tensor_name = tensor_name + var_name[len(scopes) + 1:]
          if full_tensor_name not in variable_map:
            raise ValueError(
                "Tensor %s (%s in %s) is not found in %s checkpoint" % (
                    full_tensor_name, var_name[len(scopes) + 1:], tensor_name,
                    checkpoint_dir
                ))
          var = var_scope._vars[var_name]  # pylint: disable=protected-access
          var._initializer_op = _checkpoint_initializer(  # pylint: disable=protected-access
              var, reader, full_tensor_name)
          logging.info("Initialize variable %s from checkpoint %s with %s" % (
              var_name, checkpoint_dir, tensor_name
          ))
