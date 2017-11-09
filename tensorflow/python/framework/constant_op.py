# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Operations that generate constants.

See the @{$python/constant_op$constants guide}.

@@zeros
@@zeros_like
@@ones
@@ones_like
@@fill
@@constant
@@linspace
@@range
@@random_normal
@@truncated_normal
@@random_uniform
@@random_shuffle
@@random_crop
@@multinomial
@@random_gamma
@@random_poisson
@@set_random_seed
"""

# Must be separate from array_ops to avoid a cyclic dependency.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util


def _eager_reshape(tensor, shape, ctx):
  """Eager-only version of Reshape op; requires tensor is an eager Tensor."""
  attr_t = tensor.dtype.as_datatype_enum
  attr_tshape, (shape,) = execute.args_to_matching_eager(
      [shape], ctx, dtypes.int32)
  attr_tshape = attr_tshape.as_datatype_enum
  inputs_flat = [tensor, shape]
  attrs = ("T", attr_t, "Tshape", attr_tshape)
  result, = execute.execute(
      b"Reshape", 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
  return result


def _eager_fill(dims, value, ctx):
  """Eager-only version of Fill op; requires value is an eager Tensor."""
  attr_t = value.dtype.as_datatype_enum
  dims = convert_to_eager_tensor(dims, ctx, dtypes.int32)
  inputs_flat = [dims, value]
  attrs = ("T", attr_t)
  result, = execute.execute(
      b"Fill", 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
  return result


def _eager_identity(tensor, ctx):
  """Eager-only version of Identity op; requires tensor is an eager Tensor."""
  attrs = ("T", tensor.dtype.as_datatype_enum)
  result, = execute.execute(
      b"Identity", 1, inputs=[tensor], attrs=attrs, ctx=ctx)
  return result


def convert_to_eager_tensor(value, ctx, dtype=None):
  """Converts the given `value` to an `EagerTensor`.

  Note that this function could return cached copies of created constants for
  performance reasons.

  Args:
    value: value to convert to EagerTensor.
    ctx: value of context.context().
    dtype: optional desired dtype of the converted EagerTensor.

  Returns:
    EagerTensor created from value.

  Raises:
    TypeError: if `dtype` is not compatible with the type of t.
  """
  if isinstance(value, ops.EagerTensor):
    if dtype is not None and value.dtype != dtype:
      raise TypeError("Expected tensor with type %r not %r" % (
          dtype, value.dtype))
    return value
  if dtype is not None:
    try:
      dtype = dtype.as_datatype_enum
    except AttributeError:
      dtype = dtypes.as_dtype(dtype).as_datatype_enum
  device = ctx.device_name
  handle = ctx._handle  # pylint: disable=protected-access
  if isinstance(value, (float,) + six.integer_types):
    # Use a scalar cache. This will put each scalar of each type only once on
    # each device. Scalars don't use much device memory but copying scalars can
    # trigger memcpys which are slow.
    cache_key = device, value, dtype, type(value)
    scalar_cache = ctx.scalar_cache()
    tensor = scalar_cache.get(cache_key, None)
    if tensor is not None:
      return tensor
    t = ops.EagerTensor(value, context=handle, device=device, dtype=dtype)
    scalar_cache[cache_key] = t
    return t
  else:
    return ops.EagerTensor(value, context=handle, device=device, dtype=dtype)


def constant(value, dtype=None, shape=None, name="Const", verify_shape=False):
  """Creates a constant tensor.

  The resulting tensor is populated with values of type `dtype`, as
  specified by arguments `value` and (optionally) `shape` (see examples
  below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the `shape` argument (if
  specified). In the case where the list length is less than the number of
  elements specified by `shape`, the last element in the list will be used
  to fill the remaining entries.

  The argument `shape` is optional. If present, it specifies the dimensions of
  the resulting tensor. If not present, the shape of `value` is used.

  If the argument `dtype` is not specified, then the type is inferred from
  the type of `value`.

  For example:

  ```python
  # Constant 1-D Tensor populated with value list.
  tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

  # Constant 2-D tensor populated with scalar value -1.
  tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                               [-1. -1. -1.]]
  ```

  Args:
    value:          A constant value (or list) of output type `dtype`.

    dtype:          The type of the elements of the resulting tensor.

    shape:          Optional dimensions of resulting tensor.

    name:           Optional name for the tensor.

    verify_shape:   Boolean that enables verification of a shape of values.

  Returns:
    A Constant Tensor.

  Raises:
    TypeError: if shape is incorrectly specified or unsupported.
  """
  ctx = context.context()
  if not ctx.in_graph_mode():
    t = convert_to_eager_tensor(value, ctx, dtype)
    if shape is None:
      return t
    shape = tensor_shape.as_shape(shape)
    if shape == t.shape:
      return t
    if verify_shape:
      raise TypeError("Expected Tensor's shape: %s, got %s." % (tuple(shape),
                                                                tuple(t.shape)))
    num_t = t.shape.num_elements()
    # TODO(josh11b): Implement shape -> eager tensor conversion.
    if num_t == shape.num_elements():
      return _eager_reshape(t, shape.as_list(), ctx)
    if num_t == 1:
      if t.dtype == dtypes.bool:
        # We don't have a Fill kernel for bool dtype on GPU. So we first run
        # Fill on CPU and then copy to GPU if needed.
        with ops.device("/device:CPU:0"):
          x = _eager_fill(shape.as_list(), t.cpu(), ctx)
        return _eager_identity(x, ctx)
      else:
        return _eager_fill(shape.as_list(), t, ctx)
    raise TypeError("Eager execution of tf.constant with unsupported shape "
                    "(value has %d elements, shape is %s with %d elements)." %
                    (num_t, shape, shape.num_elements()))
  g = ops.get_default_graph()
  tensor_value = attr_value_pb2.AttrValue()
  tensor_value.tensor.CopyFrom(
      tensor_util.make_tensor_proto(
          value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  dtype_value = attr_value_pb2.AttrValue(type=tensor_value.tensor.dtype)
  const_tensor = g.create_op(
      "Const", [], [dtype_value.type],
      attrs={"value": tensor_value,
             "dtype": dtype_value},
      name=name).outputs[0]
  return const_tensor


def is_constant(tensor_or_op):
  if isinstance(tensor_or_op, ops.Tensor):
    op = tensor_or_op.op
  else:
    op = tensor_or_op
  return op.type == "Const"


def _constant_tensor_conversion_function(v, dtype=None, name=None,
                                         as_ref=False):
  _ = as_ref
  return constant(v, dtype=dtype, name=name)


ops.register_tensor_conversion_function(
    (list, tuple), _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    np.ndarray, _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    np.generic, _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    object, _constant_tensor_conversion_function, 200)


def _tensor_shape_tensor_conversion_function(s,
                                             dtype=None,
                                             name=None,
                                             as_ref=False):
  """Function to convert TensorShape to Tensor."""
  _ = as_ref
  if not s.is_fully_defined():
    raise ValueError(
        "Cannot convert a partially known TensorShape to a Tensor: %s" % s)
  s_list = s.as_list()
  int64_value = 0
  for dim in s_list:
    if dim >= 2**31:
      int64_value = dim
      break

  if dtype is not None:
    if dtype not in (dtypes.int32, dtypes.int64):
      raise TypeError("Cannot convert a TensorShape to dtype: %s" % dtype)
    if dtype == dtypes.int32 and int64_value:
      raise ValueError("Cannot convert a TensorShape to dtype int32; "
                       "a dimension is too large (%s)" % int64_value)
  else:
    dtype = dtypes.int64 if int64_value else dtypes.int32
  if name is None:
    name = "shape_as_tensor"
  return constant(s_list, dtype=dtype, name=name)


ops.register_tensor_conversion_function(
    tensor_shape.TensorShape, _tensor_shape_tensor_conversion_function, 100)


def _dimension_tensor_conversion_function(d,
                                          dtype=None,
                                          name=None,
                                          as_ref=False):
  """Function to convert Dimension to Tensor."""
  _ = as_ref
  if d.value is None:
    raise ValueError("Cannot convert an unknown Dimension to a Tensor: %s" % d)
  if dtype is not None:
    if dtype not in (dtypes.int32, dtypes.int64):
      raise TypeError("Cannot convert a TensorShape to dtype: %s" % dtype)
  else:
    dtype = dtypes.int32
  if name is None:
    name = "shape_as_tensor"
  return constant(d.value, dtype=dtype, name=name)


ops.register_tensor_conversion_function(
    tensor_shape.Dimension, _dimension_tensor_conversion_function, 100)
