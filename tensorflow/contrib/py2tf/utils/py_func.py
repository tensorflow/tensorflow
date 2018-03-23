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

from collections import namedtuple

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import script_ops


class MatchDType(namedtuple('MatchDType', ('arg_number',))):
  """Allows matching the dtype of an argument.

  Used in conjunction with function calls. For example, MatchDType(0) will
  match the DType of the first argument.
  """

  pass


def wrap_py_func(f, return_dtypes, args, kwargs=None, use_dummy_return=False):
  """Helper that wraps a callable to py_func.

  The helper passes tensor arguments through the py_func interface. Non-tensor
  arguments are allowed, and will be passed to f directly. Note that non-tensor
  arguments are captured by f will not update every time the wrapper is
  called (this is consistent with its argument list, which only includes
  the tensor arguments). In general, it's safest not to reuse this wrapper.

  Args:
    f: Callable
    return_dtypes: None, individual of tuple/list of DType or MatchDType, the
        data type for each of f's return value(s). Set to None if f has no
        return values or use_dummy_return is True. Use MatchDType to define a
        dtype identical to that of `i`th argument (argument 0 is the first);
        an argument must of Tensor type if it is to be used with MatchDType.
    args: Positional arguments for f, as list or tuple.
    kwargs: Keyword arguments for f, as dict with string keys. May be None.
    use_dummy_return: If True, the function will return a dummy value of 1
        and discard its actual return value.
  Returns:
    The return values of f converted to tensor.
  Raises:
    ValueError: if any of the arguments are incorrect.
  """

  if return_dtypes and use_dummy_return:
    raise ValueError('if use_dummy_return is True, return_dtypes must be empty')

  tensor_args = []
  tensor_args_idx = {}

  # Of the positional arguments, only grab the tensor ones to be passed through
  # the py_func.
  n_args = len(args)
  arg_is_tensor = tuple(map(tensor_util.is_tensor, args))
  for i in range(n_args):
    if arg_is_tensor[i]:
      tensor_args_idx[i] = len(tensor_args)
      tensor_args.append(args[i])

  # We essentially take the tensor kwargs, if any, and add them to the list of
  # positional arguments. The kwargs are then reconstructed inside the py_func.
  #
  # For example, if
  #
  #     args = [Tensor(1), 'foo']
  #     kwargs = {'a': Tensor(2), 'b': 'bar'}
  #
  # Then
  #
  #     tensor_args = (Tensor(1), Tensor(2))
  #     kwarg_keys = ('a', 'b')
  if kwargs:
    kwarg_keys = tuple(kwargs.keys())
    kwarg_is_tensor = {k: tensor_util.is_tensor(kwargs[k]) for k in kwarg_keys}
    for k in kwarg_keys:
      if kwarg_is_tensor[k]:
        tensor_args_idx[k] = len(tensor_args)
        tensor_args.append(kwargs[k])
  else:
    kwarg_keys = ()

  # Set up return dtypes.
  def match_arg_dtype(arg_number):
    arg = args[arg_number]
    if not arg_is_tensor[arg_number]:
      raise ValueError(
          'argument %d was used with MatchDType and must be a tf.Tensor, but '
          'was %s instead' % (arg_number, type(arg)))
    return arg.dtype

  if return_dtypes:
    if isinstance(return_dtypes, MatchDType):
      return_dtypes = match_arg_dtype(return_dtypes.arg_number)
    elif isinstance(return_dtypes, (list, tuple)):
      return_dtypes = tuple(
          match_arg_dtype(a.arg_number) if isinstance(a, MatchDType) else a
          for a in return_dtypes)
    else:
      assert isinstance(return_dtypes, dtypes.DType)

  def f_wrapper(*tensor_args):
    f_args = tuple(
        tensor_args[tensor_args_idx[i]] if arg_is_tensor[i] else a
        for i, a in enumerate(args))
    f_kwargs = {
        k: tensor_args[tensor_args_idx[k]] if kwarg_is_tensor[k] else kwargs[k]
        for i, k in enumerate(kwarg_keys)
    }
    retval = f(*f_args, **f_kwargs)
    return 1 if use_dummy_return else retval

  return script_ops.py_func(f_wrapper, tensor_args, dtypes.int64
                            if use_dummy_return else return_dtypes)
