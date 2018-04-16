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
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
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

  def flushed_print(*vals):
    print(*vals)
    sys.stdout.flush()

  return py_func.wrap_py_func(
      flushed_print, None, values, use_dummy_return=True)
