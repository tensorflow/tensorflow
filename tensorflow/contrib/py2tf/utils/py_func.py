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
"""Pyfunc creation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import script_ops


def wrap_py_func(f, return_dtypes, arguments, use_dummy_return=False):
  """Helper that wraps a callable to py_func.

  The helper passes tensor arguments through the py_func interface. Non-tensor
  arguments are allowed, and will be passed to f directly. Note that non-tensor
  arguments are captured by f will not update every time the wrapper is
  called (this is consistent with its argument list, which only includes
  the tensor arguments). In general, it's safest not to reuse this wrapper.

  Args:
    f: Callable
    return_dtypes: DType, tuple, list or None, the data type for each of f's
        return value. None if f has no return values or use_dummy_return is
        True.
    arguments: Arguments for f
    use_dummy_return: If True, the function will return a dummy value of 1
        and discard its actual return value.
  Returns:
    The return values of f converted to tensor.
  Raises:
    ValueError: if the arguments are incorrect.
  """

  if return_dtypes and use_dummy_return:
    raise ValueError('if use_dummy_return is True, return_dtypes must be empty')

  n = len(arguments)
  arg_is_tensor = tuple(map(tensor_util.is_tensor, arguments))
  index_in_tensor_list = [0] * n
  i = 0
  for j in range(n):
    index_in_tensor_list[j] = i
    if arg_is_tensor[j]:
      i += 1

  def f_wrapper(*tensor_args):
    f_args = tuple(tensor_args[index_in_tensor_list[i]]
                   if arg_is_tensor[i] else arguments[i] for i in range(n))
    retval = f(*f_args)
    return 1 if use_dummy_return else retval

  return script_ops.py_func(
      f_wrapper, tuple(arguments[i] for i in range(n) if arg_is_tensor[i]),
      dtypes.int64 if use_dummy_return else return_dtypes)
