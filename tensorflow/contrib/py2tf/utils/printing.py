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
"""TensorFlow printing support utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.utils import py_func
from tensorflow.python.ops import logging_ops


def is_tf_print_compatible(value):
  # TODO(mdan): Enable once we can reliably test this.
  # This is currently disabled because we can't capture the output of
  # op kernels from Python.
  del value
  return False


def call_print(*values):
  """Compiled counterpart of the print builtin.

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
