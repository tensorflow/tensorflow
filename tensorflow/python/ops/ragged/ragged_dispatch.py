# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Operator dispatch for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect

# @TODO(edloper): Set this to True in the CL that exports RaggedTensors.
_UPDATE_DOCSTRINGS = False

# Information about an argument to an operation: The name of the argument, its
# position in the argument list, and a boolean flag indicating whether it
# expects a list of tensors.
_ArgInfo = collections.namedtuple('ArgInfo', ['name', 'position', 'is_list'])


def _get_arg_infos(func, arg_names):
  """Returns an `_ArgInfo` for each argument of `func` specified by `arg_names`.

  Args:
    func: The function whose arguments should be described.
    arg_names: The names of the arguments to get info for.

  Returns:
    A tuple of `_ArgInfo`s.
  """
  arg_infos = []

  # Inspect the func's argspec to find the position of each arg.
  arg_spec = tf_inspect.getargspec(func)
  for argname in arg_names:
    assert isinstance(argname, str)
    is_list = argname.startswith('[') and argname.endswith(']')
    if is_list:
      argname = argname[1:-1]
    if argname not in arg_spec.args:
      raise ValueError('Argument %r not found function in %s.  Args=%s' %
                       (argname, func, arg_spec.args))
    arg_infos.append(_ArgInfo(argname, arg_spec.args.index(argname), is_list))
  return arg_infos


def _is_convertible_to_tensor(value):
  """Returns true if `value` is convertible to a `Tensor`."""
  if isinstance(value,
                (ops.Tensor, variables.Variable, np.ndarray, int, float, str)):
    return True
  elif isinstance(value, (sparse_tensor.SparseTensor,)):
    return False
  else:
    try:
      ops.convert_to_tensor(value)
      return True
    except (TypeError, ValueError):
      return False


class UnaryRaggedElementwiseDispatcher(dispatch.OpDispatcher):
  """OpDispatcher for unary ops that map a base op across ragged values."""

  def __init__(self, original_op, arg_is_list=False):
    self._original_op = original_op
    self._arg_is_list = arg_is_list
    arg_names = tf_inspect.getfullargspec(original_op)[0]
    self._x = arg_names[0]
    if _UPDATE_DOCSTRINGS:
      original_op.__doc__ = (
          original_op.__doc__.rstrip() + '\n\n' +
          '    `{x}` may be a `tf.RaggedTensor`.\n'.format(x=self._x))

  def handle(self, args, kwargs):
    if args:
      x, args = args[0], args[1:]
    else:
      kwargs = kwargs.copy()
      x = kwargs.pop(self._x, None)
    if x is None:
      return self.NOT_SUPPORTED
    if self._arg_is_list:
      found_ragged = False
      for elt in x:
        if ragged_tensor.is_ragged(elt):
          found_ragged = True
        elif not _is_convertible_to_tensor(elt):
          return self.NOT_SUPPORTED
      if found_ragged:
        nested_splits_lists = [
            elt.nested_row_splits for elt in x if ragged_tensor.is_ragged(elt)
        ]
        flat_values = [
            elt.flat_values if ragged_tensor.is_ragged(elt) else elt
            for elt in x
        ]
        with ops.control_dependencies(
            ragged_util.assert_splits_match(nested_splits_lists)):
          return ragged_tensor.RaggedTensor.from_nested_row_splits(
              self._original_op(flat_values, *args, **kwargs),
              nested_splits_lists[0])
      else:
        return self.NOT_SUPPORTED
    else:
      found_ragged = ragged_tensor.is_ragged(x)
      if found_ragged:
        mapped_values = self._original_op(x.flat_values, *args, **kwargs)
        return x.with_flat_values(mapped_values)
      else:
        return self.NOT_SUPPORTED


class BinaryRaggedElementwiseDispatcher(dispatch.OpDispatcher):
  """OpDispatcher for binary ops that map a base op across ragged values.

  Supports broadcasting.
  """

  def __init__(self, original_op):
    self._original_op = original_op
    arg_names = tf_inspect.getfullargspec(original_op)[0]
    self._x = arg_names[0]
    self._y = arg_names[1]
    if _UPDATE_DOCSTRINGS:
      original_op.__doc__ = (
          original_op.__doc__.rstrip() + '\n\n' +
          '    `{x}` and `{y}` may be a `tf.RaggedTensor`.\n'.format(
              x=self._x, y=self._y))

  def handle(self, args, kwargs):
    # Extract the binary args.
    if len(args) > 1:
      x = args[0]
      y = args[1]
      args = args[2:]
    elif args:
      kwargs = kwargs.copy()
      x = args[0]
      y = kwargs.pop(self._y, None)
      args = args[1:]
    else:
      kwargs = kwargs.copy()
      x = kwargs.pop(self._x, None)
      y = kwargs.pop(self._y, None)

    # Bail if we don't have at least one ragged argument.
    x_is_ragged = ragged_tensor.is_ragged(x)
    y_is_ragged = ragged_tensor.is_ragged(y)
    if not (x_is_ragged or y_is_ragged):
      return self.NOT_SUPPORTED

    # Convert args to tensors.  Bail if conversion fails.
    try:
      if not x_is_ragged:
        x = ops.convert_to_tensor(x, name=self._x, preferred_dtype=y.dtype)
      if not y_is_ragged:
        y = ops.convert_to_tensor(y, name=self._y, preferred_dtype=x.dtype)
    except (TypeError, ValueError):
      return self.NOT_SUPPORTED

    if ((x_is_ragged and y_is_ragged) or
        (x_is_ragged and x.flat_values.shape.ndims <= y.shape.ndims) or
        (y_is_ragged and y.flat_values.shape.ndims <= x.shape.ndims)):
      bcast_shape = ragged_tensor_shape.broadcast_dynamic_shape(
          ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(x),
          ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(y))
      x = ragged_tensor_shape.broadcast_to(
          x, bcast_shape, broadcast_inner_dimensions=False)
      y = ragged_tensor_shape.broadcast_to(
          y, bcast_shape, broadcast_inner_dimensions=False)

    x_values = x.flat_values if ragged_tensor.is_ragged(x) else x
    y_values = y.flat_values if ragged_tensor.is_ragged(y) else y
    mapped_values = self._original_op(x_values, y_values, *args, **kwargs)
    if ragged_tensor.is_ragged(x):
      return x.with_flat_values(mapped_values)
    else:
      return y.with_flat_values(mapped_values)


class RaggedDispatcher(dispatch.OpDispatcher):
  """OpDispatcher for ragged ops.

  Dispatches to a wrapped op-handler if at least one of the `tensor_args`
  arguments is a RaggedTensor or a RaggedTensorValue; and all of the
  `tensor_args` arguments are convertible to Tensor or RaggedTensor.
  """

  def __init__(self, original_op, ragged_op, ragged_args):
    op_arg_names = tf_inspect.getfullargspec(original_op)[0]
    ragged_arg_names = tf_inspect.getfullargspec(ragged_op)[0]
    if op_arg_names != ragged_arg_names:
      raise AssertionError(
          'Signature must exactly match when overriding %s with %s: %s vs %s' %
          (original_op, ragged_op, op_arg_names, ragged_arg_names))
    self._ragged_op = ragged_op
    self._ragged_args = _get_arg_infos(ragged_op, ragged_args)
    if _UPDATE_DOCSTRINGS:
      arg_list = ' and '.join('`%s`' % arg for arg in ragged_args)
      original_op.__doc__ = (
          original_op.__doc__.rstrip() + '\n\n' +
          '    {0} may be a `tf.RaggedTensor`.\n'.format(arg_list))

  def handle(self, args, kwargs):
    if self.is_supported(args, kwargs):
      return self._ragged_op(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED

  def is_supported(self, args, kwargs):
    found_ragged = False
    for arg_info in self._ragged_args:
      if arg_info.position < len(args):
        arg = args[arg_info.position]
      else:
        arg = kwargs.get(arg_info.name, None)

      if arg_info.is_list:
        if not isinstance(arg, (list, tuple)):
          return False
        for elt in arg:
          if ragged_tensor.is_ragged(elt):
            found_ragged = True
          elif not _is_convertible_to_tensor(elt):
            return False
      else:
        if ragged_tensor.is_ragged(arg):
          found_ragged = True
        elif not _is_convertible_to_tensor(arg):
          return False
    return found_ragged


def ragged_dispatch(original_op, tensor_args):

  def decorator(ragged_op):
    dispatch.RaggedDispatcher(original_op, ragged_op,
                              tensor_args).register(original_op)
    return ragged_op

  return decorator


_UNARY_ELEMENTWISE_OPS = [
    array_ops.check_numerics,
    array_ops.identity,
    array_ops.ones_like,
    array_ops.ones_like_v2,
    array_ops.zeros_like,
    array_ops.zeros_like_v2,
    clip_ops.clip_by_value,
    math_ops.abs,
    math_ops.acos,
    math_ops.acosh,
    math_ops.angle,
    math_ops.asin,
    math_ops.asinh,
    math_ops.atan,
    math_ops.atanh,
    math_ops.cast,
    math_ops.ceil,
    math_ops.conj,
    math_ops.cos,
    math_ops.cosh,
    math_ops.digamma,
    math_ops.erf,
    math_ops.erfc,
    math_ops.exp,
    math_ops.expm1,
    math_ops.floor,
    math_ops.imag,
    math_ops.is_finite,
    math_ops.is_inf,
    math_ops.is_nan,
    math_ops.lgamma,
    math_ops.log,
    math_ops.log1p,
    math_ops.log_sigmoid,
    math_ops.logical_not,
    math_ops.negative,
    math_ops.real,
    math_ops.reciprocal,
    math_ops.rint,
    math_ops.round,
    math_ops.rsqrt,
    math_ops.saturate_cast,
    math_ops.sign,
    math_ops.sin,
    math_ops.sinh,
    math_ops.sqrt,
    math_ops.square,
    math_ops.tan,
    parsing_ops.decode_compressed,
    string_ops.string_to_number,
    string_ops.string_to_hash_bucket,
    string_ops.as_string,
    string_ops.decode_base64,
    string_ops.encode_base64,
    string_ops.regex_full_match,
    string_ops.regex_replace,
    string_ops.string_strip,
    string_ops.string_to_hash_bucket,
    string_ops.string_to_hash_bucket_fast,
    string_ops.string_to_hash_bucket_strong,
    string_ops.substr,
    string_ops.substr_v2,
    string_ops.string_length,
    string_ops.string_length_v2,
    string_ops.unicode_script,
]

_UNARY_LIST_ELEMENTWISE_OPS = [
    math_ops.add_n,
    string_ops.string_join,
]

_BINARY_ELEMENTWISE_OPS = [
    math_ops.add,
    math_ops.atan2,
    math_ops.complex,
    math_ops.div_no_nan,
    math_ops.divide,
    math_ops.equal,
    math_ops.floordiv,
    math_ops.floormod,
    math_ops.greater,
    math_ops.greater_equal,
    math_ops.less,
    math_ops.less_equal,
    math_ops.logical_and,
    math_ops.logical_or,
    math_ops.logical_xor,
    math_ops.maximum,
    math_ops.minimum,
    math_ops.multiply,
    math_ops.not_equal,
    math_ops.pow,
    math_ops.realdiv,
    math_ops.squared_difference,
    math_ops.subtract,
    math_ops.truediv,
    math_ops.truncatediv,
    math_ops.truncatemod,
]

# (original_op, ragged_op, ragged_args)
_RAGGED_DISPATCH_OPS = [
    (array_ops.batch_gather, ragged_array_ops.batch_gather,
     ['params', 'indices']),
    (array_ops.concat, ragged_array_ops.concat, ['values']),
    (array_ops.expand_dims_v2, ragged_array_ops.expand_dims, ['input']),
    (array_ops.gather_v2, ragged_array_ops.gather, ['params', 'indices']),
    (array_ops.gather_nd, ragged_array_ops.gather_nd, ['params', 'indices']),
    (array_ops.stack, ragged_array_ops.stack, ['values']),
    (array_ops.tile, ragged_array_ops.tile, ['input']),
    (array_ops.where, ragged_array_ops.where, ['condition', 'x', 'y']),
    (math_ops.unsorted_segment_sum, ragged_math_ops.segment_sum,
     ['data', 'segment_ids']),
    (math_ops.unsorted_segment_prod, ragged_math_ops.segment_prod,
     ['data', 'segment_ids']),
    (math_ops.unsorted_segment_min, ragged_math_ops.segment_min,
     ['data', 'segment_ids']),
    (math_ops.unsorted_segment_max, ragged_math_ops.segment_max,
     ['data', 'segment_ids']),
    (math_ops.unsorted_segment_mean, ragged_math_ops.segment_mean,
     ['data', 'segment_ids']),
    (math_ops.unsorted_segment_sqrt_n, ragged_math_ops.segment_sqrt_n,
     ['data', 'segment_ids']),
    (math_ops.reduce_sum, ragged_math_ops.reduce_sum, ['input_tensor']),
    (math_ops.reduce_prod, ragged_math_ops.reduce_prod, ['input_tensor']),
    (math_ops.reduce_min, ragged_math_ops.reduce_min, ['input_tensor']),
    (math_ops.reduce_max, ragged_math_ops.reduce_max, ['input_tensor']),
    (math_ops.reduce_mean, ragged_math_ops.reduce_mean, ['input_tensor']),
    (math_ops.reduce_any, ragged_math_ops.reduce_any, ['input_tensor']),
    (math_ops.reduce_all, ragged_math_ops.reduce_all, ['input_tensor']),
]


def register_dispatchers():
  """Constructs & registers OpDispatchers for ragged ops."""

  op_list = (
      _UNARY_ELEMENTWISE_OPS + _UNARY_LIST_ELEMENTWISE_OPS +
      _BINARY_ELEMENTWISE_OPS + [x[0] for x in _RAGGED_DISPATCH_OPS])
  for op in op_list:
    _, undecorated_op = tf_decorator.unwrap(op)
    if not hasattr(undecorated_op, tf_export.API_ATTRS['tensorflow'].names):
      raise AssertionError('Expected %s to be an exported symbol '
                           '(while adding a RaggedTensor dispatcher)')

  for op in _UNARY_ELEMENTWISE_OPS:
    UnaryRaggedElementwiseDispatcher(op).register(op)

  for op in _UNARY_LIST_ELEMENTWISE_OPS:
    UnaryRaggedElementwiseDispatcher(op, True).register(op)

  for op in _BINARY_ELEMENTWISE_OPS:
    BinaryRaggedElementwiseDispatcher(op).register(op)

  for (original_op, ragged_op, args) in _RAGGED_DISPATCH_OPS:
    RaggedDispatcher(original_op, ragged_op, args).register(original_op)

  docstring = (
      '\n\n### Additional ops that support `RaggedTensor`\n\n' + '\n'.join([
          '* `tf.%s`' % tf_export.get_canonical_name_for_symbol(op)
          for op in op_list
      ]))

  return docstring
