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
"""Operators corresponding to Python builtin functions.

List of built-in functions: https://docs.python.org/3/library/functions.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.autograph.utils import py_func
from tensorflow.contrib.autograph.utils import tensors
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops


UNDEFINED = object()


def overload_of(f):
  if f in SUPPORTED_BUILTINS:
    return BUILTIN_FUINCTIONS_MAP[f.__name__]
  return f


def abs_(x):
  if tensor_util.is_tensor(x):
    return _tf_abs(x)
  return _py_abs(x)


def _tf_abs(x):
  return math_ops.abs(x)


def _py_abs(x):
  return abs(x)


def float_(x=0):
  if tensor_util.is_tensor(x):
    return _tf_float(x)
  return _py_float(x)


def _tf_float(x):
  # TODO(mdan): We shouldn't assume float32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.float32)
  return math_ops.cast(x, dtype=dtypes.float32)


def _py_float(x):
  return float(x)


def int_(x=0, base=UNDEFINED):
  if tensor_util.is_tensor(x):
    return _tf_int(x, base)
  return _py_int(x, base)


def _tf_int(x, base):
  if base not in (10, UNDEFINED):
    raise NotImplementedError('base {} not supported for int'.format(base))

  # TODO(mdan): We shouldn't assume int32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.int32)
  return math_ops.cast(x, dtype=dtypes.int32)


def _py_int(x, base):
  if base is UNDEFINED:
    return int(x)
  return int(x, base)


def len_(s):
  if tensors.is_tensor_array(s):
    return _tf_tensor_array_len(s)
  elif tensors.is_tensor_list(s):
    return _tf_tensor_list_len(s)
  elif tensor_util.is_tensor(s):
    return _tf_tensor_len(s)
  return _py_len(s)


def _tf_tensor_array_len(s):
  return s.size()


def _tf_tensor_list_len(s):
  return list_ops.tensor_list_length(s)


def _tf_tensor_len(s):
  """Overload of len_ for Tensor arguments."""
  # Statically shaped tensors: length is known ahead of time.
  if s.shape.ndims and s.shape[0].value is not None:
    return s.shape[0].value

  # Static shape of unknown dimensions: use dynamic shape but statically
  # chech that it's a scalar.
  shape = array_ops.shape(s)

  assert shape.shape, 'shape tensor of zero size? {}'.format(shape)

  if shape.shape[0] == 0:
    raise ValueError(
        'len requires a non-scalar tensor, got one of shape {}'.format(shape))

  if shape.shape[0].value is not None:
    return array_ops.shape(s)[0]

  # Fully dynamic shape: use ops.
  rank = array_ops.rank(s)

  def raise_zero_rank_error():
    msg = gen_string_ops.string_join(
        ['len requires non-zero rank, got ',
         gen_string_ops.as_string(rank)])
    with ops.control_dependencies([control_flow_ops.Assert(False, [msg])]):
      return constant_op.constant(0, dtype=dtypes.int32)

  return control_flow_ops.cond(rank > 0, lambda: array_ops.shape(s)[0],
                               raise_zero_rank_error)


def _py_len(s):
  return len(s)


def print_(*objects, **kwargs):
  # Note: Python 2.6 doesn't support explicit keywords after starargs.
  unknown_kwargs = tuple(
      set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
  if unknown_kwargs:
    raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))

  # TODO(mdan): use logging_ops.Print when py_func is not supported.
  return _tf_py_func_print(objects, kwargs)


def _tf_py_func_print(objects, kwargs):
  """Overload of print_ as a py_func implementation."""
  override_kwargs = {k: v for k, v in kwargs.items() if v is not UNDEFINED}
  if 'flush' not in override_kwargs:
    # Defaulting to flushing the console in graph mode, which helps reduce
    # garbled output in IPython.
    override_kwargs['flush'] = True

  def print_wrapper(*vals):
    if six.PY3:
      # TensorFlow doesn't seem to generate Unicode when passing strings to
      # py_func. This causes the print to add a "b'" wrapper to the output,
      # which is probably never what you want.
      vals = tuple(
          v.decode('utf-8') if isinstance(v, bytes) else v for v in vals)
    six.print_(*vals, **override_kwargs)

  return py_func.wrap_py_func(
      print_wrapper, None, objects, use_dummy_return=True)


def range_(start_or_stop, stop=UNDEFINED, step=UNDEFINED):
  if any(tensor_util.is_tensor(s) for s in (start_or_stop, stop, step)):
    return _tf_range(start_or_stop, stop, step)
  return _py_range(start_or_stop, stop, step)


def _tf_range(start_or_stop, stop, step):
  # TODO(mdan): We should optimize this when a full tensor is not required.
  if step is not UNDEFINED:
    return math_ops.range(start_or_stop, stop, step)
  if stop is not UNDEFINED:
    return math_ops.range(start_or_stop, stop)
  return math_ops.range(start_or_stop)


def _py_range(start_or_stop, stop, step):
  if step is not UNDEFINED:
    return range(start_or_stop, stop, step)
  if stop is not UNDEFINED:
    return range(start_or_stop, stop)
  return range(start_or_stop)


SUPPORTED_BUILTINS = set((abs, float, int, len, print, range))

if six.PY2:
  SUPPORTED_BUILTINS.add(xrange)

BUILTIN_FUINCTIONS_MAP = {
    'abs': abs_,
    'float': float_,
    'int': int_,
    'len': len_,
    'print': print_,
    'range': range_,
    'xrange': range_,
}
