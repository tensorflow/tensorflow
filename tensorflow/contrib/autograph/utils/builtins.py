# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Builtin conversion utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.autograph.utils import py_func
from tensorflow.contrib.autograph.utils import type_check
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_inspect


def dynamic_builtin(f, *args, **kwargs):
  """Converts a builtin function call inline."""
  # Some built-ins may be objects.
  if not tf_inspect.isbuiltin(f) and f not in (range,):
    return f(*args, **kwargs)

  if f is len:
    return dynamic_len(*args, **kwargs)
  if six.PY2 and f is xrange:
    return dynamic_range(*args, **kwargs)
  if f is range:
    return dynamic_range(*args, **kwargs)

  raise NotImplementedError(
      'The "%s" builtin is not yet supported.' % f.__name__)


def dynamic_len(list_or_tensor):
  """Implementation of len using dynamic dispatch."""
  if tensor_util.is_tensor(list_or_tensor):
    shape = list_or_tensor.shape
    if not shape:
      raise ValueError(
          'len requires non-zero rank for tensor "%s"' % list_or_tensor)
    return array_ops.shape(list_or_tensor)[0]
  return len(list_or_tensor)


def dynamic_range(start_or_stop, stop=None, step=None):
  """Implementation of range using dynamic dispatch."""
  if type_check.is_tensor(start_or_stop, stop, step):
    if step is not None:
      return math_ops.range(start_or_stop, stop, step)
    if stop is not None:
      return math_ops.range(start_or_stop, stop)
    return math_ops.range(start_or_stop)

  if step is not None:
    return range(start_or_stop, stop, step)
  elif stop is not None:
    return range(start_or_stop, stop)
  return range(start_or_stop)


def is_tf_print_compatible(value):
  # TODO(mdan): Enable once we can reliably test this.
  # This is currently disabled because we can't capture the output of
  # op kernels from Python.
  del value
  return False


def dynamic_print(*values):
  """Implementartion of print using dynamic dispatch.

  The function attempts to use tf.Print if all the values are compatible.
  Otherwise, it will fall back to py_func.

  Args:
    *values: values to print
  Returns:
    A dummy value indicating the print completed. If tf.
  """

  if all(map(is_tf_print_compatible, values)):
    return logging_ops.Print(1, values)
  return py_func.wrap_py_func(print, None, values, use_dummy_return=True)


def dynamic_dataset(iterated):
  """Implementartion of smart tf.data.Dataset epoch wrapping.

  The function checks if the input is a tf.data.Dataset and if so then wraps it
  so that for each element it returns it also returns the current epoch the
  dataset iteration is in, for two epochs.  If the input is not a
  tf.data.Dataset then it just returns the input.

  Args:
    iterated: The iterable or tf.data.Dataset that is being iterated over.
  Returns:
    Either just the untouched input, or in the case of input being a
    tf.data.Dataset then it returns a wrapped  tf.data.Dataset where for each
    element it returns it also returns the current epoch the dataset iteration
    is in.
  """
  if not isinstance(iterated, dataset_ops.Dataset):
    return iterated

  def epoch_dataset_number_helper(i):
    return dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.from_tensors(i).repeat(), iterated))

  epoch_numbers = dataset_ops.Dataset.range(2)
  return epoch_numbers.flat_map(epoch_dataset_number_helper)


def dynamic_for_cond(iteration, iterated):
  """Implementartion of smart while-loop condition using dynamic dispatch.

  The function checks if it is iterating over a tf.data.Dataset or not, and in
  the case it is not then it simply returns if we are still in range of the
  iterated and the next element.  If it is iterating over a dataset then it only
  iterates for a single epoch.

  Args:
    iteration: The current iteration of the loop.
    iterated: The iterable or tf.data.Dataset that is being iterated over.
  Returns:
    A tuple of a bool that indicates whether the loop should continue, and the
    next element in iterated.
  """
  # TODO(znado): Clean up.
  # TODO(znado): This won't work for unpacked iterates. Fix.
  if isinstance(iterated, dataset_ops.Dataset):
    curr_epoch, next_elem = iterated.make_one_shot_iterator().get_next()
    return math_ops.less(curr_epoch, 1), next_elem
  elif tensor_util.is_tensor(iterated):
    if iterated.shape.ndims > 1:
      elem_shape = array_ops.shape(iterated)[1:]
    else:
      elem_shape = ()
    if iterated.shape.ndims == 0 or iterated.shape[0] == 0:
      return False, array_ops.zeros(elem_shape, iterated.dtype)
    return control_flow_ops.cond(
        math_ops.less(iteration, dynamic_len(iterated)),
        lambda: (True, iterated[iteration]),
        lambda: (False, array_ops.zeros(elem_shape, iterated.dtype)))
  elif hasattr(iterated, '__len__'):
    if iteration < len(iterated):
      return True, iterated[iteration]
    return False, None
  else:
    raise NotImplementedError('Python iterators not yet supported.')
