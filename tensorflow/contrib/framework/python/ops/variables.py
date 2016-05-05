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

@@assert_global_step
@@create_global_step
@@get_global_step
@@assert_or_get_global_step
@@local_variable
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

__all__ = [
    'assert_global_step', 'create_global_step', 'get_global_step',
    'assert_or_get_global_step', 'local_variable']


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
    TypeError: if `dtype` is invalid.
    ValueError: if global step key is already defined.
  """
  graph = ops.get_default_graph() if graph is None else graph
  if get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    result = variables.Variable(
        0, trainable=False, dtype=dtypes.int64, name=ops.GraphKeys.GLOBAL_STEP)
    graph.add_to_collection(ops.GraphKeys.GLOBAL_STEP, result)
    return result


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

