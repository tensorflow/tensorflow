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
"""Experimental library that exposes XLA operations directly in TensorFlow.

It is sometimes useful to be able to build HLO programs directly from
TensorFlow. This file provides Tensorflow operators that mirror the semantics of
HLO operators as closely as possible.

Note: There is no promise of backward or forward compatibility for operators
defined in this module. This is primarily because the underlying HLO operators
do not promise backward or forward compatibility.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# TODO(phawkins): provide wrappers for all XLA operators. Currently the missing
# ops include:
# infeed/outfeed (available via tf.contrib.tpu)
# collectives, e.g., cross-replica-sum (available via tf.contrib.tpu)
# conditional
# gather/scatter
# collapse

# This file reuses builtin names (following XLA's names, so we can call things
# like xla.max), so we capture the builtin versions here.
# pylint: disable=redefined-builtin
_max = max
_min = min
_slice = slice  # pylint: disable=invalid-name

constant = constant_op.constant

# Unary operators.

# For most arithmetic operators there is a TensorFlow operator
# that exactly corresponds to each XLA operator. Rather than defining
# XLA-specific variants, we reuse the corresponding TensorFlow operator.
# TODO(phawkins): It would be even better to have TensorFlow operators that 1:1
# wrap every HLO operator, because that would allow us to be confident that the
# semantics match.


def _unary_op(fn):
  """Wrapper that restricts `fn` to have the correct signature."""

  def unary_op_wrapper(x, name=None):
    return fn(x, name=name)

  return unary_op_wrapper


abs = _unary_op(math_ops.abs)
# TODO(phawkins): implement clz.
conj = _unary_op(math_ops.conj)
cos = _unary_op(math_ops.cos)
ceil = _unary_op(math_ops.ceil)
digamma = _unary_op(math_ops.digamma)
erf = _unary_op(math_ops.erf)
erfc = _unary_op(math_ops.erfc)
# TODO(phawkins): implement erfinv
exp = _unary_op(math_ops.exp)
expm1 = _unary_op(math_ops.expm1)
floor = _unary_op(math_ops.floor)
imag = _unary_op(math_ops.imag)
is_finite = _unary_op(math_ops.is_finite)
lgamma = _unary_op(math_ops.lgamma)
log = _unary_op(math_ops.log)
log1p = _unary_op(math_ops.log1p)
logical_not = _unary_op(math_ops.logical_not)
neg = _unary_op(math_ops.neg)
real = _unary_op(math_ops.real)
# TODO(phawkins): unlike xla::Round, this rounds to even instead of zero for
# numbers halfway between two integers.
round = _unary_op(math_ops.round)
sin = _unary_op(math_ops.sin)
sign = _unary_op(math_ops.sign)
tanh = _unary_op(math_ops.tanh)

# Binary operators

# The main difference between TensorFlow and XLA binary ops is the broadcasting
# semantics. TensorFlow uses Numpy-style broadcasting semantics, whereas XLA
# requires an explicit specification of which dimensions to broadcast if the
# arguments have different ranks.


def _broadcasting_binary_op(fn):
  """Wraps a binary Tensorflow operator and performs XLA-style broadcasting."""

  def broadcasting_binary_op_wrapper(x, y, broadcast_dims=None, name=None):
    """Inner wrapper function."""
    broadcast_dims = broadcast_dims or []
    broadcast_dims = ops.convert_to_tensor(broadcast_dims, dtypes.int64)
    # Rather than relying on having static shape information in the TensorFlow
    # graph, we use an XlaBroadcastHelper op that can compute the correct shapes
    # at JIT compilation time.
    x, y = gen_xla_ops.xla_broadcast_helper(x, y, broadcast_dims)
    return fn(x, y, name=name)

  return broadcasting_binary_op_wrapper


# Map from TF signed types to TF unsigned types.
_SIGNED_TO_UNSIGNED_TABLE = {
    dtypes.int8: dtypes.uint8,
    dtypes.int16: dtypes.uint16,
    dtypes.int32: dtypes.uint32,
    dtypes.int64: dtypes.uint64,
}

# Map from TF unsigned types to TF signed types.
_UNSIGNED_TO_SIGNED_TABLE = {
    dtypes.uint8: dtypes.int8,
    dtypes.uint16: dtypes.int16,
    dtypes.uint32: dtypes.int32,
    dtypes.uint64: dtypes.int64,
}


def _shift_right_logical_helper(x, y, name=None):
  """Performs an integer right logical shift irrespective of input type."""
  assert y.dtype == x.dtype
  dtype = x.dtype
  signed = dtype in _SIGNED_TO_UNSIGNED_TABLE
  if signed:
    unsigned_dtype = _SIGNED_TO_UNSIGNED_TABLE[dtype]
    x = math_ops.cast(x, unsigned_dtype)
    y = math_ops.cast(y, unsigned_dtype)
  output = bitwise_ops.right_shift(x, y, name=name)
  if signed:
    output = math_ops.cast(output, dtype)
  return output


def _shift_right_arithmetic_helper(x, y, name=None):
  """Performs an integer right arithmetic shift irrespective of input type."""
  assert y.dtype == x.dtype
  dtype = x.dtype
  unsigned = dtype in _UNSIGNED_TO_SIGNED_TABLE
  if unsigned:
    signed_dtype = _UNSIGNED_TO_SIGNED_TABLE[dtype]
    x = math_ops.cast(x, signed_dtype)
    y = math_ops.cast(y, signed_dtype)
  output = bitwise_ops.right_shift(x, y, name=name)
  if unsigned:
    output = math_ops.cast(output, dtype)
  return output


add = _broadcasting_binary_op(math_ops.add)
sub = _broadcasting_binary_op(math_ops.sub)
mul = _broadcasting_binary_op(math_ops.mul)
div = _broadcasting_binary_op(math_ops.div)
rem = _broadcasting_binary_op(gen_math_ops.mod)
max = _broadcasting_binary_op(math_ops.maximum)
min = _broadcasting_binary_op(math_ops.minimum)
atan2 = _broadcasting_binary_op(math_ops.atan2)
complex = _broadcasting_binary_op(math_ops.complex)
logical_and = _broadcasting_binary_op(math_ops.logical_and)
logical_or = _broadcasting_binary_op(math_ops.logical_or)
logical_xor = _broadcasting_binary_op(math_ops.logical_xor)
eq = _broadcasting_binary_op(math_ops.equal)
ne = _broadcasting_binary_op(math_ops.not_equal)
ge = _broadcasting_binary_op(math_ops.greater_equal)
gt = _broadcasting_binary_op(math_ops.greater)
le = _broadcasting_binary_op(math_ops.less_equal)
lt = _broadcasting_binary_op(math_ops.less)
pow = _broadcasting_binary_op(math_ops.pow)
shift_left = _broadcasting_binary_op(bitwise_ops.left_shift)
shift_right_logical = _broadcasting_binary_op(_shift_right_logical_helper)
shift_right_arithmetic = _broadcasting_binary_op(_shift_right_arithmetic_helper)


def _binary_op(fn):
  """Wrapper that restricts `fn` to have the correct signature."""

  def binary_op_wrapper(x, y, name=None):
    return fn(x, y, name=name)

  return binary_op_wrapper


transpose = _binary_op(array_ops.transpose)
rev = _binary_op(array_ops.reverse)

bitcast_convert_type = array_ops.bitcast


def broadcast(x, dims, name=None):
  x = ops.convert_to_tensor(x)
  shape = array_ops.concat([constant_op.constant(dims),
                            array_ops.shape(x)],
                           axis=0)
  return array_ops.broadcast_to(x, shape, name=name)


def clamp(a, x, b, name=None):
  return min(max(a, x, name=name), b, name=name)


concatenate = array_ops.concat


def conv(lhs,
         rhs,
         window_strides,
         padding,
         lhs_dilation,
         rhs_dilation,
         dimension_numbers,
         feature_group_count=1,
         precision_config=None,
         name=None):
  """Wraps the XLA ConvGeneralDilated operator.

  ConvGeneralDilated is the most general form of XLA convolution and is
  documented at
  https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution

  Args:
    lhs: the input tensor
    rhs: the kernel tensor
    window_strides: the inter-window strides
    padding: the padding to apply at the start and end of each input dimensions
    lhs_dilation: dilation to apply between input elements
    rhs_dilation: dilation to apply between kernel elements
    dimension_numbers: a `ConvolutionDimensionNumbers` proto.
    feature_group_count: number of feature groups for grouped convolution.
    precision_config: a `xla.PrecisionConfig` proto.
    name: an optional name for the operator

  Returns:
    A tensor representing the output of the convolution.
  """
  precision_config_proto = ""
  if precision_config:
    precision_config_proto = precision_config.SerializeToString()
  return gen_xla_ops.xla_conv(
      lhs,
      rhs,
      window_strides=window_strides,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      feature_group_count=feature_group_count,
      dimension_numbers=dimension_numbers.SerializeToString(),
      precision_config=precision_config_proto,
      name=name)


convert_element_type = math_ops.cast


def dot(lhs, rhs, name=None):
  return math_ops.tensordot(lhs, rhs, axes=1, name=name)


def dot_general(lhs, rhs, dimension_numbers, precision_config=None, name=None):
  precision_config_proto = ""
  if precision_config:
    precision_config_proto = precision_config.SerializeToString()
  return gen_xla_ops.xla_dot(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers.SerializeToString(),
      precision_config=precision_config_proto,
      name=name)


dynamic_slice = gen_xla_ops.xla_dynamic_slice
dynamic_update_slice = gen_xla_ops.xla_dynamic_update_slice

# TODO(phawkins): generalize tf.pad to support interior padding, and then remove
# the XLA-specific pad operator.
pad = gen_xla_ops.xla_pad


def random_normal(mu, sigma, dims, name=None):
  mu = ops.convert_to_tensor(mu)
  return random_ops.random_normal(
      dims, mean=mu, stddev=sigma, dtype=mu.dtype, name=name)


def random_uniform(minval, maxval, dims, name=None):
  minval = ops.convert_to_tensor(minval)
  return random_ops.random_uniform(
      dims, minval, maxval, dtype=minval.dtype, name=name)


recv = gen_xla_ops.xla_recv
reduce = gen_xla_ops.xla_reduce


def reduce_window(operand,
                  init,
                  reducer,
                  window_dimensions,
                  window_strides=None,
                  base_dilations=None,
                  window_dilations=None,
                  padding=None,
                  name=None):
  """Wraps the XLA ReduceWindow operator.

  ReduceWindow is documented at
  https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

  Args:
    operand: the input tensor
    init: a scalar tensor representing the initial value for the reduction
    reducer: a reduction function that combines a pair of scalars.
    window_dimensions: shape of the window, as a list of integers
    window_strides: inter-window strides, as a list of integers. Optional; if
      omitted, defaults to strides of 1.
    padding: padding to apply to 'operand'. List of (low, high) pairs of
      integers that specify the padding to apply before and after each
      dimension. Optional; if omitted, defaults to no padding.
    name: the operator name, or None.

  Returns:
    A tensor that represents the output of the reduce_window operator.
  """
  window_strides = window_strides or [1] * len(window_dimensions)
  base_dilations = base_dilations or [1] * len(window_dimensions)
  window_dilations = window_dilations or [1] * len(window_dimensions)
  padding = padding or [(0, 0)] * len(window_dimensions)
  return gen_xla_ops.xla_reduce_window(
      input=operand,
      init_value=init,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      base_dilations=base_dilations,
      window_dilations=window_dilations,
      padding=padding,
      computation=reducer,
      name=name)


def reshape(x, new_sizes, dimensions=None, name=None):
  if dimensions is not None:
    x = array_ops.transpose(x, dimensions)
  x = array_ops.reshape(x, new_sizes, name=name)
  return x


def select(condition, x, y, name=None):
  return array_ops.where(condition, x, y, name)


select_and_scatter = gen_xla_ops.xla_select_and_scatter
send = gen_xla_ops.xla_send


def slice(x, start_dims, limit_dims, strides):
  spec = [
      _slice(start, limit, stride)
      for (start, limit, stride) in zip(start_dims, limit_dims, strides)
  ]
  return x[tuple(spec)]


sort = gen_xla_ops.xla_sort
key_value_sort = gen_xla_ops.xla_key_value_sort
while_loop = gen_xla_ops.xla_while
dequantize = gen_xla_ops.xla_dequantize
