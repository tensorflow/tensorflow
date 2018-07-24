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

import sys

import six

from tensorflow.contrib.autograph.utils import py_func
from tensorflow.contrib.autograph.utils import type_check
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops


def dynamic_builtin(f, *args, **kwargs):
  """Converts a builtin function call inline."""
  if f is len:
    return dynamic_len(*args, **kwargs)
  if six.PY2 and f is xrange:
    return dynamic_range(*args, **kwargs)
  if f is range:
    return dynamic_range(*args, **kwargs)
  if f is int:
    return dynamic_int(*args, **kwargs)
  if f is float:
    return dynamic_float(*args, **kwargs)

  raise NotImplementedError(
      'The "%s" builtin is not yet supported.' % f.__name__)


def dynamic_len(list_or_tensor):
  """Implementation of len using dynamic dispatch."""
  if tensor_util.is_tensor(list_or_tensor):
    shape = list_or_tensor.shape
    if not shape.ndims:
      raise ValueError(
          'len requires non-zero rank for tensor "%s"' % list_or_tensor)
    return array_ops.shape(list_or_tensor)[0]
  return len(list_or_tensor)


def dynamic_int(num_or_tensor, **kwargs):
  """Implementation of int() using dynamic dispatch."""
  if tensor_util.is_tensor(num_or_tensor):
    return math_ops.cast(num_or_tensor, dtype=dtypes.int32, **kwargs)
  return int(num_or_tensor)


def dynamic_float(num_or_tensor, **kwargs):
  """Implementation of float() using dynamic dispatch."""
  if tensor_util.is_tensor(num_or_tensor):
    return math_ops.cast(num_or_tensor, dtype=dtypes.float32, **kwargs)
  return float(num_or_tensor)


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
  """Implementation of print using dynamic dispatch.

  The function attempts to use tf.Print if all the values are compatible.
  Otherwise, it will fall back to py_func.

  Args:
    *values: values to print
  Returns:
    A dummy value indicating the print completed. If tf.
  """

  if all(map(is_tf_print_compatible, values)):
    return logging_ops.Print(1, values)

  def print_wrapper(*vals):
    if six.PY3:
      # TensorFlow doesn't seem to generate Unicode when passing strings to
      # py_func. This causes the print to add a "b'" wrapper to the output,
      # which is probably never what you want.
      vals = tuple(v.decode() if isinstance(v, bytes) else v for v in vals)
    print(*vals)
    # The flush helps avoid garbled output in IPython.
    sys.stdout.flush()

  return py_func.wrap_py_func(
      print_wrapper, None, values, use_dummy_return=True)
