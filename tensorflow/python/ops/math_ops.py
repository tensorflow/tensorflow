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
"""Basic arithmetic operators.

See the [python/math_ops](python/math_ops) guide.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_math_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
linspace = gen_math_ops.lin_space

arg_max = deprecation.deprecated(None, "Use `tf.math.argmax` instead")(arg_max)  # pylint: disable=used-before-assignment
arg_min = deprecation.deprecated(None, "Use `tf.math.argmin` instead")(arg_min)  # pylint: disable=used-before-assignment
tf_export(v1=["arg_max"])(arg_max)
tf_export(v1=["arg_min"])(arg_min)

# This is set by resource_variable_ops.py. It is included in this way since
# there is a circular dependency between math_ops and resource_variable_ops
_resource_variable_type = None


def _set_doc(doc):

  def _decorator(func):
    func.__doc__ = doc
    return func

  return _decorator


# pylint: disable=redefined-builtin
@tf_export(v1=["math.argmax", "argmax"])
@deprecation.deprecated_args(None, "Use the `axis` argument instead",
                             "dimension")
@_set_doc(
    gen_math_ops.arg_max.__doc__.replace("dimensions", "axes").replace(
        "dimension", "axis"))
def argmax(input,
           axis=None,
           name=None,
           dimension=None,
           output_type=dtypes.int64):
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "dimension", dimension)
  return argmax_v2(input, axis, output_type, name)


@tf_export("math.argmax", "argmax", v1=[])
def argmax_v2(input,
              axis=None,
              output_type=dtypes.int64,
              name=None):
  """Returns the index with the largest value across axes of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
    `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`,
    `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `-rank(input), rank(input))`.
      Describes which axis of the input Tensor to reduce across. For vectors,
      use axis = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`.
      Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  """
  if axis is None:
    axis = 0
  return gen_math_ops.arg_max(input, axis, name=name, output_type=output_type)


@tf_export(v1=["math.argmin", "argmin"])
@deprecation.deprecated_args(None, "Use the `axis` argument instead",
                             "dimension")
@_set_doc(
    gen_math_ops.arg_min.__doc__.replace("dimensions", "axes").replace(
        "dimension", "axis"))
def argmin(input,
           axis=None,
           name=None,
           dimension=None,
           output_type=dtypes.int64):
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "dimension", dimension)
  return argmin_v2(input, axis, output_type, name)


@tf_export("math.argmin", "argmin", v1=[])
def argmin_v2(input,
              axis=None,
              output_type=dtypes.int64,
              name=None):
  """Returns the index with the smallest value across axes of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
    `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`,
    `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `-rank(input), rank(input))`.
      Describes which axis of the input Tensor to reduce across. For vectors,
      use axis = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`.
      Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  """
  if axis is None:
    axis = 0
  return gen_math_ops.arg_min(input, axis, name=name, output_type=output_type)


# pylint: enable=redefined-builtin


# pylint: disable=anomalous-backslash-in-string,protected-access
# pylint: disable=g-docstring-has-escape
@tf_export("math.abs", "abs")
@dispatch.add_dispatch_support
def abs(x, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float32` or `float64` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\(a + bj\\). The
  absolute value is computed as \\( \sqrt{a^2 + b^2}\\).  For example:
  ```python
  x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
  tf.abs(x)  # [5.25594902, 6.60492229]
  ```

  Args:
    x: A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
      `int32`, `int64`, `complex64` or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` the same size and type as `x` with absolute
      values.
    Note, for `complex64` or `complex128` input, the returned `Tensor` will be
      of type `float32` or `float64`, respectively.
  """
  with ops.name_scope(name, "Abs", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.is_complex:
      return gen_math_ops.complex_abs(x, Tout=x.dtype.real_dtype, name=name)
    return gen_math_ops._abs(x, name=name)
# pylint: enable=g-docstring-has-escape


# pylint: disable=redefined-builtin
def _bucketize(input, boundaries, name=None):
  return gen_math_ops.bucketize(input=input, boundaries=boundaries, name=name)


# pylint: enable=redefined-builtin


class DivideDelegateWithName(object):
  """Use Python2/Python3 division delegation to implement divide for tensors."""

  def __init__(self, x, name):
    """Construct DivideDelegateWithName.

    Args:
      x: Tensor to use as left operand in operator overloads
      name: The name that is preferred for the op created.
    """
    self.x = x
    self.name = name

  def __truediv__(self, y):
    return _truediv_python3(self.x, y, self.name)

  def __floordiv__(self, y):
    return floordiv(self.x, y, self.name)

  def __div__(self, y):
    return _div_python2(self.x, y, self.name)


@tf_export("math.divide", "divide")
@dispatch.add_dispatch_support
def divide(x, y, name=None):
  """Computes Python style division of `x` by `y`."""

  if name is not None:
    # Cannot use tensors operator overload, because it has no way to track
    # override names. Use a dummy class to track the runtime division behavior
    return DivideDelegateWithName(x, name) / y
  else:
    return x / y


@tf_export("math.multiply", "multiply")
@dispatch.add_dispatch_support
def multiply(x, y, name=None):
  return gen_math_ops.mul(x, y, name)


multiply.__doc__ = gen_math_ops.mul.__doc__.replace("Multiply", "`tf.multiply`")


# TODO(aselle): put deprecation in after another round of global code changes
@deprecation.deprecated(
    "2016-12-30",
    "`tf.mul(x, y)` is deprecated, please use `tf.multiply(x, y)` or `x * y`")
def _mul(x, y, name=None):
  return gen_math_ops.mul(x, y, name)


_mul.__doc__ = (
    gen_math_ops.mul.__doc__ + ("" if _mul.__doc__ is None else _mul.__doc__))


@tf_export("math.subtract", "subtract")
@dispatch.add_dispatch_support
def subtract(x, y, name=None):
  return gen_math_ops.sub(x, y, name)


subtract.__doc__ = gen_math_ops.sub.__doc__.replace("`Sub`", "`tf.subtract`")


# TODO(aselle): put deprecation in after another round of global code changes
@deprecation.deprecated(
    "2016-12-30",
    "`tf.sub(x, y)` is deprecated, please use `tf.subtract(x, y)` or `x - y`")
def _sub(x, y, name=None):
  return gen_math_ops.sub(x, y, name)


_sub.__doc__ = (
    gen_math_ops.sub.__doc__ + ("" if _sub.__doc__ is None else _sub.__doc__))


negative = gen_math_ops.neg


# pylint: disable=g-docstring-has-escape
@deprecation.deprecated(
    "2016-12-30",
    "`tf.neg(x)` is deprecated, please use `tf.negative(x)` or `-x`")
def _neg(x, name=None):
  """Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  return negative(x, name)


# pylint: enable=g-docstring-has-escape


@tf_export(v1=["math.scalar_mul", "scalar_mul"])
def scalar_mul(scalar, x, name=None):
  """Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

  Intended for use in gradient code which might deal with `IndexedSlices`
  objects, which are easy to multiply by a scalar but more expensive to
  multiply with arbitrary tensors.

  Args:
    scalar: A 0-D scalar `Tensor`. Must have known shape.
    x: A `Tensor` or `IndexedSlices` to be scaled.
    name: A name for the operation (optional).

  Returns:
    `scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.

  Raises:
    ValueError: if scalar is not a 0-D `scalar`.
  """
  scalar = ops.convert_to_tensor(
      scalar, dtype=x.dtype.base_dtype, name="scalar")
  shape = scalar.get_shape()
  if shape.ndims == 0:
    if isinstance(x, ops.IndexedSlices):
      return ops.IndexedSlices(gen_math_ops.mul(scalar, x.values, name),
                               x.indices, x.dense_shape)
    else:
      return gen_math_ops.mul(scalar, x, name)
  else:
    raise ValueError("Only scalar multiply works, got shape %s" % shape)


@tf_export("math.scalar_mul", "scalar_mul", v1=[])
@_set_doc(scalar_mul.__doc__)
def scalar_mul_v2(scalar, x, name=None):
  with ops.name_scope(name, "scalar_mul", [x]) as name:
    return scalar_mul(scalar, x, name)


@tf_export("math.pow", "pow")
@dispatch.add_dispatch_support
def pow(x, y, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```python
  x = tf.constant([[2, 2], [3, 3]])
  y = tf.constant([[8, 16], [2, 3]])
  tf.pow(x, y)  # [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
     `complex64`, or `complex128`.
    y: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
     `complex64`, or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.
  """
  with ops.name_scope(name, "Pow", [x]) as name:
    return gen_math_ops._pow(x, y, name=name)


# pylint: disable=redefined-builtin,redefined-outer-name
@tf_export("dtypes.complex", "complex")
@dispatch.add_dispatch_support
def complex(real, imag, name=None):
  r"""Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```python
  real = tf.constant([2.25, 3.25])
  imag = tf.constant([4.75, 5.75])
  tf.complex(real, imag)  # [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`,
      `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64` or `complex128`.
  """
  real = ops.convert_to_tensor(real, name="real")
  imag = ops.convert_to_tensor(imag, name="imag")
  with ops.name_scope(name, "Complex", [real, imag]) as name:
    input_types = (real.dtype, imag.dtype)
    if input_types == (dtypes.float64, dtypes.float64):
      Tout = dtypes.complex128
    elif input_types == (dtypes.float32, dtypes.float32):
      Tout = dtypes.complex64
    else:
      raise TypeError("real and imag have incorrect types: "
                      "{} {}".format(real.dtype.name, imag.dtype.name))
    return gen_math_ops._complex(real, imag, Tout=Tout, name=name)


@tf_export("math.real", v1=["math.real", "real"])
@deprecation.deprecated_endpoints("real")
@dispatch.add_dispatch_support
def real(input, name=None):
  r"""Returns the real part of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the real part of each element in `input` considered as a complex number.

  For example:

  ```python
  x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  tf.real(x)  # [-2.25, 3.25]
  ```

  If `input` is already real, it is returned unchanged.

  Args:
    input: A `Tensor`. Must have numeric type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Real", [input]) as name:
    if input.dtype.is_complex:
      real_dtype = input.dtype.real_dtype
      return gen_math_ops.real(input, Tout=real_dtype, name=name)
    else:
      return input


@tf_export("math.imag", v1=["math.imag", "imag"])
@deprecation.deprecated_endpoints("imag")
@dispatch.add_dispatch_support
def imag(input, name=None):
  r"""Returns the imaginary part of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the imaginary part of each element in `input` considered as a complex
  number. If `input` is real, a tensor of all zeros is returned.

  For example:

  ```python
  x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  tf.imag(x)  # [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float`, `double`,
      `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Imag", [input]) as name:
    if input.dtype.is_complex:
      return gen_math_ops.imag(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.zeros_like(input)


@tf_export("math.angle", v1=["math.angle", "angle"])
@deprecation.deprecated_endpoints("angle")
@dispatch.add_dispatch_support
def angle(input, name=None):
  r"""Returns the element-wise argument of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the argument of each element in `input` considered as a complex number.

  The elements in `input` are considered to be complex numbers of the form
  \\(a + bj\\), where *a* is the real part and *b* is the imaginary part.
  If `input` is real then *b* is zero by definition.

  The argument returned by this function is of the form \\(atan2(b, a)\\).
  If `input` is real, a tensor of all zeros is returned.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.angle(input) ==> [2.0132, 1.056]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float`, `double`,
      `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Angle", [input]) as name:
    if input.dtype.is_complex:
      return gen_math_ops.angle(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.zeros_like(input)


# pylint: enable=redefined-outer-name,redefined-builtin


@tf_export("math.round", "round")
@dispatch.add_dispatch_support
def round(x, name=None):  # pylint: disable=redefined-builtin
  """Rounds the values of a tensor to the nearest integer, element-wise.

  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use tf::cint.
  For example:

  ```python
  x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
  tf.round(x)  # [ 1.0, 2.0, 2.0, 2.0, -4.0 ]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, or `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.
  """
  x = ops.convert_to_tensor(x, name="x")
  if x.dtype.is_integer:
    return x
  else:
    return gen_math_ops.round(x, name=name)


@tf_export("dtypes.cast", "cast")
@dispatch.add_dispatch_support
def cast(x, dtype, name=None):
  """Casts a tensor to a new type.

  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor` or `IndexedSlices`) to `dtype`.

  For example:

  ```python
  x = tf.constant([1.8, 2.2], dtype=tf.float32)
  tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
  ```

  The operation supports data types (for `x` and `dtype`) of
  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `complex64`, `complex128`, `bfloat16`.
  In case of casting from complex types (`complex64`, `complex128`) to real
  types, only the real part of `x` is returned. In case of casting from real
  types to complex types (`complex64`, `complex128`), the imaginary part of the
  returned value is set to `0`. The handling of complex types here matches the
  behavior of numpy.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices` of numeric type. It could
      be `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`,
      `int64`, `float16`, `float32`, `float64`, `complex64`, `complex128`,
      `bfloat16`.
    dtype: The destination type. The list of supported dtypes is the same as
      `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and
      same type as `dtype`.

  Raises:
    TypeError: If `x` cannot be cast to the `dtype`.
  """
  base_type = dtypes.as_dtype(dtype).base_dtype
  if isinstance(x,
                (ops.Tensor, _resource_variable_type)) and base_type == x.dtype:
    return x
  with ops.name_scope(name, "Cast", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      values_cast = cast(x.values, base_type, name=name)
      x = sparse_tensor.SparseTensor(x.indices, values_cast, x.dense_shape)
    elif isinstance(x, ops.IndexedSlices):
      values_cast = cast(x.values, base_type, name=name)
      x = ops.IndexedSlices(values_cast, x.indices, x.dense_shape)
    else:
      # TODO(josh11b): If x is not already a Tensor, we could return
      # ops.convert_to_tensor(x, dtype=dtype, ...)  here, but that
      # allows some conversions that cast() can't do, e.g. casting numbers to
      # strings.
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype.base_dtype != base_type:
        x = gen_math_ops.cast(x, base_type, name=name)
    if x.dtype.is_complex and base_type.is_floating:
      logging.warn("Casting complex to real discards imaginary part.")
    return x


@tf_export("dtypes.saturate_cast", "saturate_cast")
@dispatch.add_dispatch_support
def saturate_cast(value, dtype, name=None):
  """Performs a safe saturating cast of `value` to `dtype`.

  This function casts the input to `dtype` without applying any scaling.  If
  there is a danger that values would over or underflow in the cast, this op
  applies the appropriate clamping before the cast.

  Args:
    value: A `Tensor`.
    dtype: The desired output `DType`.
    name: A name for the operation (optional).

  Returns:
    `value` safely cast to `dtype`.
  """
  # When casting to a type with smaller representable range, clamp.
  # Note that this covers casting to unsigned types as well.
  with ops.name_scope(name, "saturate_cast", [value]) as name:
    value = ops.convert_to_tensor(value, name="value")
    dtype = dtypes.as_dtype(dtype).base_dtype
    if value.dtype.min < dtype.min:
      value = gen_math_ops.maximum(value,
                                   ops.convert_to_tensor(
                                       dtype.min, dtype=value.dtype,
                                       name="min"))
    if value.dtype.max > dtype.max:
      value = gen_math_ops.minimum(value,
                                   ops.convert_to_tensor(
                                       dtype.max, dtype=value.dtype,
                                       name="max"))
    return cast(value, dtype, name=name)

@deprecation.deprecated(date=None, instructions="Use tf.cast instead.")
@tf_export(v1=["to_float"])
def to_float(x, name="ToFloat"):
  """Casts a tensor to type `float32`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `float32`.

  Raises:
    TypeError: If `x` cannot be cast to the `float32`.
  """
  return cast(x, dtypes.float32, name=name)


@deprecation.deprecated(date=None, instructions="Use tf.cast instead.")
@tf_export(v1=["to_double"])
def to_double(x, name="ToDouble"):
  """Casts a tensor to type `float64`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `float64`.

  Raises:
    TypeError: If `x` cannot be cast to the `float64`.
  """
  return cast(x, dtypes.float64, name=name)


@deprecation.deprecated(date=None, instructions="Use tf.cast instead.")
@tf_export(v1=["to_int32"])
def to_int32(x, name="ToInt32"):
  """Casts a tensor to type `int32`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `int32`.

  Raises:
    TypeError: If `x` cannot be cast to the `int32`.
  """
  return cast(x, dtypes.int32, name=name)


@deprecation.deprecated(date=None, instructions="Use tf.cast instead.")
@tf_export(v1=["to_int64"])
def to_int64(x, name="ToInt64"):
  """Casts a tensor to type `int64`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `int64`.

  Raises:
    TypeError: If `x` cannot be cast to the `int64`.
  """
  return cast(x, dtypes.int64, name=name)


@deprecation.deprecated(date=None, instructions="Use tf.cast instead.")
@tf_export(v1=["to_bfloat16"])
def to_bfloat16(x, name="ToBFloat16"):
  """Casts a tensor to type `bfloat16`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `bfloat16`.

  Raises:
    TypeError: If `x` cannot be cast to the `bfloat16`.
  """
  return cast(x, dtypes.bfloat16, name=name)


@deprecation.deprecated(date=None, instructions="Use tf.cast instead.")
@tf_export(v1=["to_complex64"])
def to_complex64(x, name="ToComplex64"):
  """Casts a tensor to type `complex64`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `complex64`.

  Raises:
    TypeError: If `x` cannot be cast to the `complex64`.
  """
  return cast(x, dtypes.complex64, name=name)


@deprecation.deprecated(date=None, instructions="Use tf.cast instead.")
@tf_export(v1=["to_complex128"])
def to_complex128(x, name="ToComplex128"):
  """Casts a tensor to type `complex128`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `complex128`.

  Raises:
    TypeError: If `x` cannot be cast to the `complex128`.
  """
  return cast(x, dtypes.complex128, name=name)


ops.Tensor._override_operator("__neg__", gen_math_ops.neg)
ops.Tensor._override_operator("__abs__", abs)
# __invert__ corresponds to the ~ operator.  Here we follow the numpy convention
# ~ marks an elementwise bit-wise inverse.  This is only implemented for boolean
# tensors and will throw a TypeError if used on nonboolean arrays
ops.Tensor._override_operator("__invert__", gen_math_ops.logical_not)


def _OverrideBinaryOperatorHelper(func, op_name, clazz_object=ops.Tensor):
  """Register operators with different tensor and scalar versions.

  If `clazz_object` is `SparseTensor`, assumes `func` takes `(sp_indices,
  sp_values, sp_shape, dense)` and outputs `(new_sp_values)`.

  Args:
    func: the operator
    op_name: name of the operator being overridden
    clazz_object: class to override for.  Either `Tensor` or `SparseTensor`.
  """

  def binary_op_wrapper(x, y):
    with ops.name_scope(None, op_name, [x, y]) as name:
      if isinstance(x, ops.Tensor) and isinstance(y, ops.Tensor):
        return func(x, y, name=name)
      elif not isinstance(y, sparse_tensor.SparseTensor):
        try:
          y = ops.convert_to_tensor(y, dtype=x.dtype.base_dtype, name="y")
        except TypeError:
          # If the RHS is not a tensor, it might be a tensor aware object
          # that can implement the operator with knowledge of itself
          # and the tensor.
          if hasattr(type(y), "__r%s__" % op_name):
            return NotImplemented
          else:
            raise
      return func(x, y, name=name)

  def binary_op_wrapper_sparse(sp_x, y):
    with ops.name_scope(None, op_name, [sp_x, y]) as name:
      y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name="y")
      return sparse_tensor.SparseTensor(sp_x.indices,
                                        func(
                                            sp_x.indices,
                                            sp_x.values,
                                            sp_x.dense_shape,
                                            y,
                                            name=name), sp_x.dense_shape)

  def r_binary_op_wrapper(y, x):
    with ops.name_scope(None, op_name, [x, y]) as name:
      x = ops.convert_to_tensor(x, dtype=y.dtype.base_dtype, name="x")
      return func(x, y, name=name)

  # Propagate func.__doc__ to the wrappers
  try:
    doc = func.__doc__
  except AttributeError:
    doc = None
  binary_op_wrapper.__doc__ = doc
  r_binary_op_wrapper.__doc__ = doc
  binary_op_wrapper_sparse.__doc__ = doc

  if clazz_object is ops.Tensor:
    clazz_object._override_operator("__%s__" % op_name, binary_op_wrapper)
    del binary_op_wrapper
    clazz_object._override_operator("__r%s__" % op_name, r_binary_op_wrapper)
    del r_binary_op_wrapper
  else:
    clazz_object._override_operator("__%s__" % op_name,
                                    binary_op_wrapper_sparse)
    del binary_op_wrapper_sparse


# Conversion table for __truediv__.  None entries mean no conversion required.
_TRUEDIV_TABLE = {
    dtypes.uint8: dtypes.float32,
    dtypes.int8: dtypes.float32,
    dtypes.uint16: dtypes.float32,
    dtypes.int16: dtypes.float32,
    dtypes.int32: dtypes.float64,
    dtypes.int64: dtypes.float64,
    dtypes.bfloat16: None,
    dtypes.float16: None,
    dtypes.float32: None,
    dtypes.float64: None,
    dtypes.complex64: None,
    dtypes.complex128: None,
}


# NOTE: the support of "sparse (true)div dense" is currently not baked in into
# "tf.(true_)div()".  Until such an API decision is made, the supported usage is
# to explicitly use the "/" operator to invoke either truediv or div.
def _sparse_dense_truediv(sp_indices, sp_values, sp_shape, y, name=None):
  """Internal helper function for 'sp_t / dense_t'."""
  with ops.name_scope(name, "truediv",
                      [sp_indices, sp_values, sp_shape, y]) as name:
    sp_values = ops.convert_to_tensor(sp_values, name="sp_values")
    y = ops.convert_to_tensor(y, name="y")
    x_dtype = sp_values.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError:
      raise TypeError("Invalid dtype %r in __truediv__" % x_dtype)
    if dtype is not None:
      sp_values = cast(sp_values, dtype)
      y = cast(y, dtype)
    return gen_sparse_ops.sparse_dense_cwise_div(
        sp_indices, sp_values, sp_shape, y, name=name)


def _truediv_python3(x, y, name=None):
  with ops.name_scope(name, "truediv", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y")
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError:
      raise TypeError("Invalid dtype %r in __truediv__" % x_dtype)
    if dtype is not None:
      x = cast(x, dtype)
      y = cast(y, dtype)
    return gen_math_ops.real_div(x, y, name=name)


def _div_python2(x, y, name=None):
  """Divide two values using Python 2 semantics. Used for Tensor.__div__.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).
  Returns:
    `x / y` returns the quotient of x and y.
  """

  with ops.name_scope(name, "div", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    if x_dtype.is_floating or x_dtype.is_complex:
      return gen_math_ops.real_div(x, y, name=name)
    else:
      return gen_math_ops.floor_div(x, y, name=name)


@tf_export("math.truediv", "truediv")
@dispatch.add_dispatch_support
def truediv(x, y, name=None):
  """Divides x / y elementwise (using Python 3 division operator semantics).

  NOTE: Prefer using the Tensor operator or tf.divide which obey Python
  division operator semantics.

  This function forces Python 3 division operator semantics where all integer
  arguments are cast to floating types first.   This op is generated by normal
  `x / y` division in Python 3 and in Python 2.7 with
  `from __future__ import division`.  If you want integer division that rounds
  down, use `x // y` or `tf.math.floordiv`.

  `x` and `y` must have the same numeric type.  If the inputs are floating
  point, the output will have the same type.  If the inputs are integral, the
  inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
  and `int64` (matching the behavior of Numpy).

  Args:
    x: `Tensor` numerator of numeric type.
    y: `Tensor` denominator of numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` evaluated in floating point.

  Raises:
    TypeError: If `x` and `y` have different dtypes.
  """
  return _truediv_python3(x, y, name)


@deprecation.deprecated(
    date=None,
    instructions="Deprecated in favor of operator or tf.math.divide.")
@tf_export(v1=["div"])
def div(x, y, name=None):
  """Divides x / y elementwise (using Python 2 division operator semantics).

  NOTE: Prefer using the Tensor division operator or tf.divide which obey Python
  division operator semantics.

  This function divides `x` and `y`, forcing Python 2.7 semantics. That is,
  if one of `x` or `y` is a float, then the result will be a float.
  Otherwise, the output will be an integer type. Flooring semantics are used
  for integer division.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).
  Returns:
    `x / y` returns the quotient of x and y.
  """
  return _div_python2(x, y, name)


@tf_export("div_no_nan")
@dispatch.add_dispatch_support
def div_no_nan(x, y, name=None):
  """Computes an unsafe divide which returns 0 if the y is zero.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    y: A `Tensor` whose dtype is compatible with `x`.
    name: A name for the operation (optional).
  Returns:
    The element-wise value of the x divided by y.
  """

  with ops.name_scope(name, "div_no_nan", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    return gen_math_ops.div_no_nan(x, y, name=name)


# TODO(aselle): This should be removed
mod = gen_math_ops.floor_mod


# TODO(aselle): Deprecate this once all internal functionality uses
# tf.truncatediv
@tf_export("math.floordiv", v1=["math.floordiv", "floordiv"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("floordiv")
def floordiv(x, y, name=None):
  """Divides `x / y` elementwise, rounding toward the most negative integer.

  The same as `tf.div(x,y)` for integers, but uses `tf.floor(tf.div(x,y))` for
  floating point arguments so that the result is always an integer (though
  possibly an integer represented as floating point).  This op is generated by
  `x // y` floor division in Python 3 and in Python 2.7 with
  `from __future__ import division`.

  `x` and `y` must have the same type, and the result will have the same type
  as well.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` rounded down.

  Raises:
    TypeError: If the inputs are complex.
  """
  with ops.name_scope(name, "floordiv", [x, y]) as name:
    return gen_math_ops.floor_div(x, y, name=name)


realdiv = gen_math_ops.real_div
truncatediv = gen_math_ops.truncate_div
# TODO(aselle): Rename this to floordiv when we can.
floor_div = gen_math_ops.floor_div
truncatemod = gen_math_ops.truncate_mod
floormod = gen_math_ops.floor_mod


def _mul_dispatch(x, y, name=None):
  """Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse"."""
  is_tensor_y = isinstance(y, ops.Tensor)
  if is_tensor_y:
    return gen_math_ops.mul(x, y, name=name)
  else:
    assert isinstance(y, sparse_tensor.SparseTensor)  # Case: Dense * Sparse.
    new_vals = gen_sparse_ops.sparse_dense_cwise_mul(y.indices, y.values,
                                                     y.dense_shape, x, name)
    return sparse_tensor.SparseTensor(y.indices, new_vals, y.dense_shape)


# NOTE(aselle): When integer division is added for sparse_dense_cwise,
# div, truediv, and floordiv should be delegated appropriately for
# Python sematnics, analogous to dense cwise tensor operations.
_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_div, "div",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(_sparse_dense_truediv, "truediv",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_mul, "mul",
                              sparse_tensor.SparseTensor)

_OverrideBinaryOperatorHelper(gen_math_ops.add, "add")
_OverrideBinaryOperatorHelper(gen_math_ops.sub, "sub")
_OverrideBinaryOperatorHelper(_mul_dispatch, "mul")
_OverrideBinaryOperatorHelper(_div_python2, "div")
_OverrideBinaryOperatorHelper(_truediv_python3, "truediv")
_OverrideBinaryOperatorHelper(floordiv, "floordiv")
_OverrideBinaryOperatorHelper(gen_math_ops.floor_mod, "mod")
_OverrideBinaryOperatorHelper(pow, "pow")


@tf_export("math.logical_xor", v1=["math.logical_xor", "logical_xor"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("logical_xor")
def logical_xor(x, y, name="LogicalXor"):
  """x ^ y = (x | y) & ~(x & y)."""
  # TODO(alemi) Make this a cwise op if people end up relying on it.
  return gen_math_ops.logical_and(
      gen_math_ops.logical_or(x, y),
      gen_math_ops.logical_not(gen_math_ops.logical_and(x, y)),
      name=name)


_OverrideBinaryOperatorHelper(gen_math_ops.logical_and, "and")
_OverrideBinaryOperatorHelper(gen_math_ops.logical_or, "or")
_OverrideBinaryOperatorHelper(logical_xor, "xor")

ops.Tensor._override_operator("__lt__", gen_math_ops.less)
ops.Tensor._override_operator("__le__", gen_math_ops.less_equal)
ops.Tensor._override_operator("__gt__", gen_math_ops.greater)
ops.Tensor._override_operator("__ge__", gen_math_ops.greater_equal)


@tf_export("range")
def range(start, limit=None, delta=1, dtype=None, name="range"):  # pylint: disable=redefined-builtin
  """Creates a sequence of numbers.

  Creates a sequence of numbers that begins at `start` and extends by
  increments of `delta` up to but not including `limit`.

  The dtype of the resulting tensor is inferred from the inputs unless
  it is provided explicitly.

  Like the Python builtin `range`, `start` defaults to 0, so that
  `range(n) = range(0, n)`.

  For example:

  ```python
  start = 3
  limit = 18
  delta = 3
  tf.range(start, limit, delta)  # [3, 6, 9, 12, 15]

  start = 3
  limit = 1
  delta = -0.5
  tf.range(start, limit, delta)  # [3, 2.5, 2, 1.5]

  limit = 5
  tf.range(limit)  # [0, 1, 2, 3, 4]
  ```

  Args:
    start: A 0-D `Tensor` (scalar). Acts as first entry in the range if
      `limit` is not None; otherwise, acts as range limit and first entry
      defaults to 0.
    limit: A 0-D `Tensor` (scalar). Upper limit of sequence,
      exclusive. If None, defaults to the value of `start` while the first
      entry of the range defaults to 0.
    delta: A 0-D `Tensor` (scalar). Number that increments
      `start`. Defaults to 1.
    dtype: The type of the elements of the resulting tensor.
    name: A name for the operation. Defaults to "range".

  Returns:
    An 1-D `Tensor` of type `dtype`.

  @compatibility(numpy)
  Equivalent to np.arange
  @end_compatibility
  """
  if limit is None:
    start, limit = 0, start

  with ops.name_scope(name, "Range", [start, limit, delta]) as name:
    start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    limit = ops.convert_to_tensor(limit, dtype=dtype, name="limit")
    delta = ops.convert_to_tensor(delta, dtype=dtype, name="delta")

    # infer dtype if not explicitly provided
    if dtype is None:
      dtype_hierarchy = [
          dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
      ]
      assert all(arg.dtype in dtype_hierarchy for arg in [start, limit, delta])
      inferred_dtype = max(
          [arg.dtype for arg in [start, limit, delta]],
          key=dtype_hierarchy.index)

      start = cast(start, inferred_dtype)
      limit = cast(limit, inferred_dtype)
      delta = cast(delta, inferred_dtype)

    return gen_math_ops._range(start, limit, delta, name=name)


# Reduction operations
def _ReductionDims(x, axis, reduction_indices=None):  # pylint: disable=invalid-name
  """Returns range(0, rank(x)) if reduction_indices is None."""
  # TODO(aselle): Remove this after deprecation
  if reduction_indices is not None:
    if axis is not None:
      raise ValueError("Can't specify both axis' and 'reduction_indices'.")
    axis = reduction_indices
  if axis is not None:
    return axis
  else:
    # Fast path: avoid creating Rank and Range ops if ndims is known.
    rank = common_shapes.rank(x)
    if rank is not None:
      return constant_op.constant(np.arange(rank), dtype=dtypes.int32)
    if (isinstance(x, sparse_tensor.SparseTensor) and
        x.dense_shape.shape.is_fully_defined()):
      rank = x.dense_shape.shape.dims[0].value  # sparse.dense_shape is 1-D.
      return constant_op.constant(np.arange(rank), dtype=dtypes.int32)

    # Otherwise, we rely on Range and Rank to do the right thing at run-time.
    return range(0, array_ops.rank(x))


def _may_reduce_to_scalar(keepdims, axis, output):
  """Set a reduction's output shape to be a scalar if we are certain."""
  if not common_shapes.has_fully_defined_shape(output) and (not keepdims) and (
      axis is None):
    output.set_shape(())
  return output


@tf_export(v1=["math.reduce_sum", "reduce_sum"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_sum_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes the sum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1, 1, 1], [1, 1, 1]])
  tf.reduce_sum(x)  # 6
  tf.reduce_sum(x, 0)  # [2, 2, 2]
  tf.reduce_sum(x, 1)  # [3, 3]
  tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
  tf.reduce_sum(x, [0, 1])  # 6
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default),
      reduces all dimensions. Must be in the range
      `[-rank(input_tensor), rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while tensorflow returns the same dtype as the input.
  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_sum(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_sum", "reduce_sum", v1=[])
@dispatch.add_dispatch_support
def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the sum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1, 1, 1], [1, 1, 1]])
  tf.reduce_sum(x)  # 6
  tf.reduce_sum(x, 0)  # [2, 2, 2]
  tf.reduce_sum(x, 1)  # [3, 3]
  tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
  tf.reduce_sum(x, [0, 1])  # 6
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while tensorflow returns the same dtype as the input.
  @end_compatibility
  """
  keepdims = False if keepdims is None else keepdims
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._sum(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.count_nonzero", "count_nonzero"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def count_nonzero(input_tensor,
                  axis=None,
                  keepdims=None,
                  dtype=dtypes.int64,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes number of nonzero elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  **NOTE** Floating point comparison to zero is done by exact floating point
  equality check.  Small values are **not** rounded to zero for purposes of
  the nonzero check.

  For example:

  ```python
  x = tf.constant([[0, 1, 0], [1, 1, 0]])
  tf.count_nonzero(x)  # 3
  tf.count_nonzero(x, 0)  # [1, 2, 0]
  tf.count_nonzero(x, 1)  # [1, 2]
  tf.count_nonzero(x, 1, keepdims=True)  # [[1], [2]]
  tf.count_nonzero(x, [0, 1])  # 3
  ```

  **NOTE** Strings are compared against zero-length empty string `""`. Any
  string with a size greater than zero is already considered as nonzero.

  For example:
  ```python
  x = tf.constant(["", "a", "  ", "b", ""])
  tf.count_nonzero(x) # 3, with "a", "  ", and "b" as nonzero strings.
  ```

  Args:
    input_tensor: The tensor to reduce. Should be of numeric type, `bool`,
      or `string`.
    axis: The dimensions to reduce. If `None` (the default),
      reduces all dimensions. Must be in the range
      `[-rank(input_tensor), rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    dtype: The output dtype; defaults to `tf.int64`.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor (number of nonzero values).
  """
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis,
      "reduction_indices", reduction_indices
      )

  return count_nonzero_v2(input_tensor, axis, keepdims, dtype, name)


@tf_export("math.count_nonzero", v1=[])
def count_nonzero_v2(input,  # pylint: disable=redefined-builtin
                     axis=None,
                     keepdims=None,
                     dtype=dtypes.int64,
                     name=None):
  """Computes number of nonzero elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  **NOTE** Floating point comparison to zero is done by exact floating point
  equality check.  Small values are **not** rounded to zero for purposes of
  the nonzero check.

  For example:

  ```python
  x = tf.constant([[0, 1, 0], [1, 1, 0]])
  tf.count_nonzero(x)  # 3
  tf.count_nonzero(x, 0)  # [1, 2, 0]
  tf.count_nonzero(x, 1)  # [1, 2]
  tf.count_nonzero(x, 1, keepdims=True)  # [[1], [2]]
  tf.count_nonzero(x, [0, 1])  # 3
  ```

  **NOTE** Strings are compared against zero-length empty string `""`. Any
  string with a size greater than zero is already considered as nonzero.

  For example:
  ```python
  x = tf.constant(["", "a", "  ", "b", ""])
  tf.count_nonzero(x) # 3, with "a", "  ", and "b" as nonzero strings.
  ```

  Args:
    input: The tensor to reduce. Should be of numeric type, `bool`,
      or `string`.
    axis: The dimensions to reduce. If `None` (the default),
      reduces all dimensions. Must be in the range
      `[-rank(input), rank(input))`.
    keepdims: If true, retains reduced dimensions with length 1.
    dtype: The output dtype; defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor (number of nonzero values).
  """
  if keepdims is None:
    keepdims = False
  with ops.name_scope(name, "count_nonzero", [input]):
    input = ops.convert_to_tensor(input, name="input")
    # A scalar of 'zero' is enough as `not_equal` will broadcast.
    zero = array_ops.zeros([], dtype=input.dtype)
    return cast(
        reduce_sum(
            # int64 reduction happens on GPU
            to_int64(gen_math_ops.not_equal(input, zero)),
            axis=axis,
            keepdims=keepdims),
        dtype=dtype)


@tf_export(v1=["math.reduce_mean", "reduce_mean"])
def reduce_mean_v1(input_tensor,
                   axis=None,
                   keepdims=None,
                   name=None,
                   reduction_indices=None,
                   keep_dims=None):
  """Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1., 1.], [2., 2.]])
  tf.reduce_mean(x)  # 1.5
  tf.reduce_mean(x, 0)  # [1.5, 1.5]
  tf.reduce_mean(x, 1)  # [1.,  2.]
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default),
      reduces all dimensions. Must be in the range
      `[-rank(input_tensor), rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.mean

  Please note that `np.mean` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `tf.reduce_mean` has an aggressive type inference from `input_tensor`,
  for example:

  ```python
  x = tf.constant([1, 0, 1, 0])
  tf.reduce_mean(x)  # 0
  y = tf.constant([1., 0., 1., 0.])
  tf.reduce_mean(y)  # 0.5
  ```

  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_mean(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_mean", "reduce_mean", v1=[])
@dispatch.add_dispatch_support
def reduce_mean(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1., 1.], [2., 2.]])
  tf.reduce_mean(x)  # 1.5
  tf.reduce_mean(x, 0)  # [1.5, 1.5]
  tf.reduce_mean(x, 1)  # [1.,  2.]
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.mean

  Please note that `np.mean` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `tf.reduce_mean` has an aggressive type inference from `input_tensor`,
  for example:

  ```python
  x = tf.constant([1, 0, 1, 0])
  tf.reduce_mean(x)  # 0
  y = tf.constant([1., 0., 1., 0.])
  tf.reduce_mean(y)  # 0.5
  ```

  @end_compatibility
  """
  keepdims = False if keepdims is None else keepdims
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops.mean(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export("math.reduce_variance")
def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the variance of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1., 2.], [3., 4.]])
  tf.reduce_variance(x)  # 1.25
  tf.reduce_variance(x, 0)  # [1., 1.]
  tf.reduce_variance(x, 1)  # [0.25,  0.25]
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name scope for the associated operations (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.var

  Please note that `np.var` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `tf.reduce_variance` has an aggressive type inference from
  `input_tensor`,
  @end_compatibility
  """
  name = name if name else "reduce_variance"
  with ops.name_scope(name):
    means = reduce_mean(input_tensor, axis=axis, keepdims=True)
    squared_deviations = gen_math_ops.square(input_tensor - means)
    return reduce_mean(squared_deviations, axis=axis, keepdims=keepdims)


@tf_export("math.reduce_std")
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the standard deviation of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1., 2.], [3., 4.]])
  tf.reduce_std(x)  # 1.1180339887498949
  tf.reduce_std(x, 0)  # [1., 1.]
  tf.reduce_std(x, 1)  # [0.5,  0.5]
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name scope for the associated operations (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.std

  Please note that `np.std` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `tf.reduce_std` has an aggressive type inference from `input_tensor`,
  @end_compatibility
  """
  name = name if name else "reduce_std"
  with ops.name_scope(name):
    variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
    return gen_math_ops.sqrt(variance)


@tf_export("math.reduce_prod", "reduce_prod", v1=[])
@dispatch.add_dispatch_support
def reduce_prod(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the product of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default),
      reduces all dimensions. Must be in the range
      `[-rank(input_tensor), rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.prod
  @end_compatibility
  """
  keepdims = False if keepdims is None else keepdims
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops.prod(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_prod", "reduce_prod"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_prod_v1(input_tensor,
                   axis=None,
                   keepdims=None,
                   name=None,
                   reduction_indices=None,
                   keep_dims=None):
  """Computes the product of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.prod
  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_prod(input_tensor, axis, keepdims, name)


@tf_export(v1=["math.reduce_min", "reduce_min"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_min_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes the minimum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.min
  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_min(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_min", "reduce_min", v1=[])
@dispatch.add_dispatch_support
def reduce_min(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the minimum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.min
  @end_compatibility
  """
  keepdims = False if keepdims is None else keepdims
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._min(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_max", "reduce_max"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_max_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes the maximum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default),
      reduces all dimensions. Must be in the range
      `[-rank(input_tensor), rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.max
  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_max(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_max", "reduce_max", v1=[])
@dispatch.add_dispatch_support
def reduce_max(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the maximum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.max
  @end_compatibility
  """
  keepdims = False if keepdims is None else keepdims
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._max(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_all", "reduce_all"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_all_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes the "logical and" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[True,  True], [False, False]])
  tf.reduce_all(x)  # False
  tf.reduce_all(x, 0)  # [False, False]
  tf.reduce_all(x, 1)  # [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.all
  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_all(input_tensor, axis, keepdims, name)


@tf_export("reduce_all", "math.reduce_all", v1=[])
@dispatch.add_dispatch_support
def reduce_all(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the "logical and" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[True,  True], [False, False]])
  tf.reduce_all(x)  # False
  tf.reduce_all(x, 0)  # [False, False]
  tf.reduce_all(x, 1)  # [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.all
  @end_compatibility
  """
  keepdims = False if keepdims is None else keepdims
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._all(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_any", "reduce_any"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_any_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes the "logical or" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[True,  True], [False, False]])
  tf.reduce_any(x)  # True
  tf.reduce_any(x, 0)  # [True, True]
  tf.reduce_any(x, 1)  # [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.any
  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_any(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_any", "reduce_any", v1=[])
@dispatch.add_dispatch_support
def reduce_any(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the "logical or" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[True,  True], [False, False]])
  tf.reduce_any(x)  # True
  tf.reduce_any(x, 0)  # [True, True]
  tf.reduce_any(x, 1)  # [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.any
  @end_compatibility
  """
  keepdims = False if keepdims is None else keepdims
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._any(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_logsumexp", "reduce_logsumexp"])
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_logsumexp_v1(input_tensor,
                        axis=None,
                        keepdims=None,
                        name=None,
                        reduction_indices=None,
                        keep_dims=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.

  For example:

  ```python
  x = tf.constant([[0., 0., 0.], [0., 0., 0.]])
  tf.reduce_logsumexp(x)  # log(6)
  tf.reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
  tf.reduce_logsumexp(x, 1)  # [log(3), log(3)]
  tf.reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
  tf.reduce_logsumexp(x, [0, 1])  # log(6)
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor.
  """
  axis = deprecation.deprecated_argument_lookup(
      "axis", axis, "reduction_indices", reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_logsumexp(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_logsumexp", "reduce_logsumexp", v1=[])
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.

  For example:

  ```python
  x = tf.constant([[0., 0., 0.], [0., 0., 0.]])
  tf.reduce_logsumexp(x)  # log(6)
  tf.reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
  tf.reduce_logsumexp(x, 1)  # [log(3), log(3)]
  tf.reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
  tf.reduce_logsumexp(x, [0, 1])  # log(6)
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  keepdims = False if keepdims is None else keepdims
  input_tensor = ops.convert_to_tensor(input_tensor)
  with ops.name_scope(name, "ReduceLogSumExp", [input_tensor]) as name:
    raw_max = reduce_max(
        input_tensor,
        axis=axis,
        keepdims=True)
    my_max = array_ops.stop_gradient(
        array_ops.where(
            gen_math_ops.is_finite(raw_max), raw_max,
            array_ops.zeros_like(raw_max)))
    result = gen_math_ops.log(
        reduce_sum(
            gen_math_ops.exp(gen_math_ops.sub(input_tensor, my_max)),
            axis,
            keepdims=keepdims))
    if not keepdims:
      my_max = array_ops.reshape(my_max, array_ops.shape(result))
    result = gen_math_ops.add(result, my_max)
    return _may_reduce_to_scalar(keepdims, axis, result)


@tf_export("linalg.trace", v1=["linalg.trace", "trace"])
@deprecation.deprecated_endpoints("trace")
def trace(x, name=None):
  """Compute the trace of a tensor `x`.

  `trace(x)` returns the sum along the main diagonal of each inner-most matrix
  in x. If x is of rank `k` with shape `[I, J, K, ..., L, M, N]`, then output
  is a tensor of rank `k-2` with dimensions `[I, J, K, ..., L]` where

  `output[i, j, k, ..., l] = trace(x[i, j, i, ..., l, :, :])`

  For example:

  ```python
  x = tf.constant([[1, 2], [3, 4]])
  tf.linalg.trace(x)  # 5

  x = tf.constant([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
  tf.linalg.trace(x)  # 15

  x = tf.constant([[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],
                   [[-1, -2, -3],
                    [-4, -5, -6],
                    [-7, -8, -9]]])
  tf.linalg.trace(x)  # [15, -15]
  ```

  Args:
    x: tensor.
    name: A name for the operation (optional).

  Returns:
    The trace of input tensor.
  """
  with ops.name_scope(name, "Trace", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return reduce_sum(array_ops.matrix_diag_part(x), [-1], name=name)


@tf_export("linalg.matmul", "matmul")
def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           name=None):
  """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

  The inputs must, following any transpositions, be tensors of rank >= 2
  where the inner 2 dimensions specify valid matrix multiplication arguments,
  and any further outer dimensions match.

  Both matrices must be of the same type. The supported types are:
  `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

  Either matrix can be transposed or adjointed (conjugated and transposed) on
  the fly by setting one of the corresponding flag to `True`. These are `False`
  by default.

  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
  This optimization is only available for plain matrices (rank-2 tensors) with
  datatypes `bfloat16` or `float32`.

  For example:

  ```python
  # 2-D tensor `a`
  # [[1, 2, 3],
  #  [4, 5, 6]]
  a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

  # 2-D tensor `b`
  # [[ 7,  8],
  #  [ 9, 10],
  #  [11, 12]]
  b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])

  # `a` * `b`
  # [[ 58,  64],
  #  [139, 154]]
  c = tf.matmul(a, b)


  # 3-D tensor `a`
  # [[[ 1,  2,  3],
  #   [ 4,  5,  6]],
  #  [[ 7,  8,  9],
  #   [10, 11, 12]]]
  a = tf.constant(np.arange(1, 13, dtype=np.int32),
                  shape=[2, 2, 3])

  # 3-D tensor `b`
  # [[[13, 14],
  #   [15, 16],
  #   [17, 18]],
  #  [[19, 20],
  #   [21, 22],
  #   [23, 24]]]
  b = tf.constant(np.arange(13, 25, dtype=np.int32),
                  shape=[2, 3, 2])

  # `a` * `b`
  # [[[ 94, 100],
  #   [229, 244]],
  #  [[508, 532],
  #   [697, 730]]]
  c = tf.matmul(a, b)

  # Since python >= 3.5 the @ operator is supported (see PEP 465).
  # In TensorFlow, it simply calls the `tf.matmul()` function, so the
  # following lines are equivalent:
  d = a @ b @ [[10.], [11.]]
  d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])
  ```

  Args:
    a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
      `complex128` and rank > 1.
    b: `Tensor` with same type and rank as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated and transposed before
      multiplication.
    adjoint_b: If `True`, `b` is conjugated and transposed before
      multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
    the product of the corresponding matrices in `a` and `b`, e.g. if all
    transpose or adjoint attributes are `False`:

    `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
    for all indices i, j.

    Note: This is matrix product, not element-wise product.


  Raises:
    ValueError: If transpose_a and adjoint_a, or transpose_b and adjoint_b
      are both set to True.
  """
  with ops.name_scope(name, "MatMul", [a, b]) as name:
    if transpose_a and adjoint_a:
      raise ValueError("Only one of transpose_a and adjoint_a can be True.")
    if transpose_b and adjoint_b:
      raise ValueError("Only one of transpose_b and adjoint_b can be True.")

    if context.executing_eagerly():
      if not isinstance(a, (ops.EagerTensor, _resource_variable_type)):
        a = ops.convert_to_tensor(a, name="a")
      if not isinstance(b, (ops.EagerTensor, _resource_variable_type)):
        b = ops.convert_to_tensor(b, name="b")
    else:
      a = ops.convert_to_tensor(a, name="a")
      b = ops.convert_to_tensor(b, name="b")

    # TODO(apassos) remove _shape_tuple here when it is not needed.
    a_shape = a._shape_tuple()  # pylint: disable=protected-access
    b_shape = b._shape_tuple()  # pylint: disable=protected-access
    if (not a_is_sparse and
        not b_is_sparse) and ((a_shape is None or len(a_shape) > 2) and
                              (b_shape is None or len(b_shape) > 2)):
      # BatchMatmul does not support transpose, so we conjugate the matrix and
      # use adjoint instead. Conj() is a noop for real matrices.
      if transpose_a:
        a = conj(a)
        adjoint_a = True
      if transpose_b:
        b = conj(b)
        adjoint_b = True
      return gen_math_ops.batch_mat_mul(
          a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name)

    # Neither matmul nor sparse_matmul support adjoint, so we conjugate
    # the matrix and use transpose instead. Conj() is a noop for real
    # matrices.
    if adjoint_a:
      a = conj(a)
      transpose_a = True
    if adjoint_b:
      b = conj(b)
      transpose_b = True

    use_sparse_matmul = False
    if a_is_sparse or b_is_sparse:
      sparse_matmul_types = [dtypes.bfloat16, dtypes.float32]
      use_sparse_matmul = (
          a.dtype in sparse_matmul_types and b.dtype in sparse_matmul_types)
    if ((a.dtype == dtypes.bfloat16 or b.dtype == dtypes.bfloat16) and
        a.dtype != b.dtype):
      # matmul currently doesn't handle mixed-precision inputs.
      use_sparse_matmul = True
    if use_sparse_matmul:
      ret = sparse_matmul(
          a,
          b,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          a_is_sparse=a_is_sparse,
          b_is_sparse=b_is_sparse,
          name=name)
      # sparse_matmul always returns float32, even with
      # bfloat16 inputs. This prevents us from configuring bfloat16 training.
      # casting to bfloat16 also matches non-sparse matmul behavior better.
      if a.dtype == dtypes.bfloat16 and b.dtype == dtypes.bfloat16:
        ret = cast(ret, dtypes.bfloat16)
      return ret
    else:
      return gen_math_ops.mat_mul(
          a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)


@tf_export("linalg.matvec")
def matvec(a,
           b,
           transpose_a=False,
           adjoint_a=False,
           a_is_sparse=False,
           b_is_sparse=False,
           name=None):
  """Multiplies matrix `a` by vector `b`, producing `a` * `b`.

  The matrix `a` must, following any transpositions, be a tensor of rank >= 2,
  and we must have `shape(b) = shape(a)[:-2] + [shape(a)[-1]]`.

  Both `a` and `b` must be of the same type. The supported types are:
  `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

  Matrix `a` can be transposed or adjointed (conjugated and transposed) on
  the fly by setting one of the corresponding flag to `True`. These are `False`
  by default.

  If one or both of the inputs contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
  This optimization is only available for plain matrices/vectors (rank-2/1
  tensors) with datatypes `bfloat16` or `float32`.

  For example:

  ```python
  # 2-D tensor `a`
  # [[1, 2, 3],
  #  [4, 5, 6]]
  a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

  # 1-D tensor `b`
  # [7, 9, 11]
  b = tf.constant([7, 9, 11], shape=[3])

  # `a` * `b`
  # [ 58,  64]
  c = tf.matvec(a, b)


  # 3-D tensor `a`
  # [[[ 1,  2,  3],
  #   [ 4,  5,  6]],
  #  [[ 7,  8,  9],
  #   [10, 11, 12]]]
  a = tf.constant(np.arange(1, 13, dtype=np.int32),
                  shape=[2, 2, 3])

  # 2-D tensor `b`
  # [[13, 14, 15],
  #  [16, 17, 18]]
  b = tf.constant(np.arange(13, 19, dtype=np.int32),
                  shape=[2, 3])

  # `a` * `b`
  # [[ 86, 212],
  #  [410, 563]]
  c = tf.matvec(a, b)
  ```

  Args:
    a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
      `complex128` and rank > 1.
    b: `Tensor` with same type and rank = `rank(a) - 1`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated and transposed before
      multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a` and `b` where each inner-most vector is
    the product of the corresponding matrices in `a` and vectors in `b`, e.g. if
    all transpose or adjoint attributes are `False`:

    `output`[..., i] = sum_k (`a`[..., i, k] * `b`[..., k]), for all indices i.

    Note: This is matrix-vector product, not element-wise product.


  Raises:
    ValueError: If transpose_a and adjoint_a are both set to True.
  """
  with ops.name_scope(name, "MatVec", [a, b]) as name:
    output = matmul(
        a,
        array_ops.expand_dims(b, axis=-1),
        transpose_a=transpose_a,
        adjoint_a=adjoint_a,
        a_is_sparse=a_is_sparse,
        b_is_sparse=b_is_sparse)
    return array_ops.squeeze(output, axis=-1)


_OverrideBinaryOperatorHelper(matmul, "matmul")

sparse_matmul = deprecation.deprecated(None, "Use `tf.linalg.matmul` instead")(
    gen_math_ops.sparse_mat_mul)
tf_export(v1=["sparse_matmul"])(sparse_matmul)


@ops.RegisterStatistics("MatMul", "flops")
def _calc_mat_mul_flops(graph, node):
  """Calculates the compute resources needed for MatMul."""
  transpose_a = node.attr["transpose_a"].b
  a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  a_shape.assert_is_fully_defined()
  if transpose_a:
    k = int(a_shape[0])
  else:
    k = int(a_shape[1])
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (k * output_count * 2))


def _as_indexed_slices(x, optimize=True):
  """Convert 'x' to IndexedSlices.

  Convert a dense Tensor to a block-sparse IndexedSlices.

  Args:
    x: Either a Tensor object, or an IndexedSlices object.
    optimize: if true, attempt to optimize the conversion of 'x'.

  Returns:
    An IndexedSlices object.

  Raises:
    TypeError: If 'x' is not a Tensor or an IndexedSlices object.
  """
  # TODO(touts): op_scope
  if not isinstance(x, (ops.Tensor, ops.IndexedSlices)):
    raise TypeError("Not a Tensor or IndexedSlices: %s" % type(x))
  if isinstance(x, ops.IndexedSlices):
    return x
  x_shape = array_ops.shape_internal(x, optimize=optimize)
  return ops.IndexedSlices(x, range(0, x_shape[0]), x_shape)


def _as_indexed_slices_list(inputs, optimize=True):
  """Convert all elements of 'inputs' to IndexedSlices.

  Additionally, homogenize the types of all the indices to
  either int32 or int64.

  Args:
    inputs: List containing either Tensor or IndexedSlices objects.
    optimize: if true, attempt to optimize the conversion of each input.

  Returns:
    A list of IndexedSlices objects.

  Raises:
    TypeError: If 'inputs' is not a list or a tuple.
  """
  if not isinstance(inputs, (list, tuple)):
    raise TypeError("Expected a list or tuple, not a %s" % type(inputs))
  outputs = [_as_indexed_slices(i, optimize=optimize) for i in inputs]
  with_int32_index = [
      o.indices for o in outputs if o.indices.dtype == dtypes.int32
  ]
  if not with_int32_index or len(with_int32_index) == len(outputs):
    return outputs
  casted_outputs = []
  for o in outputs:
    if o.indices.dtype == dtypes.int32:
      casted_outputs.append(
          ops.IndexedSlices(o.values, cast(o.indices, dtypes.int64),
                            o.dense_shape))
    else:
      casted_outputs.append(o)
  return casted_outputs


@tf_export("math.add_n", "add_n")
@dispatch.add_dispatch_support
def add_n(inputs, name=None):
  """Adds all input tensors element-wise.

  Args:
    inputs: A list of `Tensor` or `IndexedSlices` objects, each with same shape
      and type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """
  if not inputs or not isinstance(inputs, (list, tuple)):
    raise ValueError("inputs must be a list of at least one"
                     "Tensor/IndexedSlices with the same dtype and shape")
  inputs = ops.convert_n_to_tensor_or_indexed_slices(inputs)
  if not all(isinstance(x, (ops.Tensor, ops.IndexedSlices)) for x in inputs):
    raise ValueError("inputs must be a list of at least one"
                     "Tensor/IndexedSlices with the same dtype and shape")

  if len(inputs) == 1:
    if isinstance(inputs[0], ops.IndexedSlices):
      values = inputs[0].values
    else:
      values = inputs[0]
    if name:
      return array_ops.identity(values, name=name)
    return values
  return gen_math_ops.add_n(inputs, name=name)


@tf_export("math.accumulate_n", v1=["math.accumulate_n", "accumulate_n"])
@deprecation.deprecated_endpoints("accumulate_n")
def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
  """Returns the element-wise sum of a list of tensors.

  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
  otherwise, these are inferred.

  `tf.math.accumulate_n` performs the same operation as `tf.add_n`, but does not
  wait for all of its inputs to be ready before beginning to sum. This can
  save memory if inputs are ready at different times, since minimum temporary
  storage is proportional to the output size rather than the inputs size.

  `accumulate_n` is differentiable (but wasn't previous to TensorFlow 1.7).

  For example:

  ```python
  a = tf.constant([[1, 2], [3, 4]])
  b = tf.constant([[5, 0], [0, 6]])
  tf.math.accumulate_n([a, b, a])  # [[7, 4], [6, 14]]

  # Explicitly pass shape and type
  tf.math.accumulate_n([a, b, a], shape=[2, 2], tensor_dtype=tf.int32)
                                                                 # [[7,  4],
                                                                 #  [6, 14]]
  ```

  Args:
    inputs: A list of `Tensor` objects, each with same shape and type.
    shape: Shape of elements of `inputs`.
    tensor_dtype: The type of `inputs`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """

  def _input_error():
    return ValueError("inputs must be a list of at least one Tensor with the "
                      "same dtype and shape")

  if not inputs or not isinstance(inputs, (list, tuple)):
    raise _input_error()
  inputs = ops.convert_n_to_tensor_or_indexed_slices(inputs)
  if not all(isinstance(x, ops.Tensor) for x in inputs):
    raise _input_error()
  if not all(x.dtype == inputs[0].dtype for x in inputs):
    raise _input_error()
  if shape is not None:
    shape = tensor_shape.as_shape(shape)
  else:
    shape = tensor_shape.unknown_shape()
  for input_tensor in inputs:
    if isinstance(input_tensor, ops.Tensor):
      shape = shape.merge_with(input_tensor.get_shape())

  # tensor_dtype is for safety only; operator's output type computed in C++
  if tensor_dtype is not None and tensor_dtype != inputs[0].dtype:
    raise TypeError("tensor_dtype is {}, but input is of type {}".format(
        tensor_dtype, inputs[0].dtype))

  if len(inputs) == 1 and name is None:
    return inputs[0]
  elif len(inputs) == 1 and name is not None:
    return array_ops.identity(inputs[0], name=name)
  elif context.executing_eagerly():
    # TemporaryVariable not currently supported in eager mode; fall back
    # onto AddN for now.
    # TODO(frreiss) remove this once the lifetime of eager variables gets
    # addressed
    return add_n(inputs, name=name)
  else:
    return gen_math_ops.accumulate_nv2(inputs, name=name, shape=shape)  # pylint: disable=protected-access


@ops.RegisterGradient("AccumulateNV2")
def _accumulate_n_grad(op, grad):
  """Same as gradient for AddN. Copies the gradient to all inputs."""
  # Not broadcasting.
  return [grad] * len(op.inputs)


@tf_export("math.sigmoid", "nn.sigmoid", "sigmoid")
def sigmoid(x, name=None):
  """Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A Tensor with type `float16`, `float32`, `float64`, `complex64`,
      or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.expit
  @end_compatibility
  """
  with ops.name_scope(name, "Sigmoid", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.sigmoid(x, name=name)


@tf_export("math.log_sigmoid", v1=["math.log_sigmoid", "log_sigmoid"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("log_sigmoid")
def log_sigmoid(x, name=None):
  """Computes log sigmoid of `x` element-wise.

  Specifically, `y = log(1 / (1 + exp(-x)))`.  For numerical stability,
  we use `y = -tf.nn.softplus(-x)`.

  Args:
    x: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x`.
  """
  with ops.name_scope(name, "LogSigmoid", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.neg(gen_nn_ops.softplus(-x), name=name)


@tf_export("math.bincount", v1=[])
def bincount(arr,
             weights=None,
             minlength=None,
             maxlength=None,
             dtype=dtypes.int32,
             name=None):
  """Counts the number of occurrences of each value in an integer array.

  If `minlength` and `maxlength` are not given, returns a vector with length
  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.
  If `weights` are non-None, then index `i` of the output stores the sum of the
  value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Args:
    arr: An int32 tensor of non-negative values.
    weights: If non-None, must be the same shape as arr. For each value in
      `arr`, the bin will be incremented by the corresponding weight instead of
      1.
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `arr` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    dtype: If `weights` is None, determines the type of the output bins.
    name: A name scope for the associated operations (optional).

  Returns:
    A vector with the same dtype as `weights` or the given `dtype`. The bin
    values.
  """
  name = "bincount" if name is None else name
  with ops.name_scope(name):
    arr = ops.convert_to_tensor(arr, name="arr", dtype=dtypes.int32)
    array_is_nonempty = reduce_prod(array_ops.shape(arr)) > 0
    output_size = cast(array_is_nonempty, dtypes.int32) * (reduce_max(arr) + 1)
    if minlength is not None:
      minlength = ops.convert_to_tensor(
          minlength, name="minlength", dtype=dtypes.int32)
      output_size = gen_math_ops.maximum(minlength, output_size)
    if maxlength is not None:
      maxlength = ops.convert_to_tensor(
          maxlength, name="maxlength", dtype=dtypes.int32)
      output_size = gen_math_ops.minimum(maxlength, output_size)
    if weights is not None:
      weights = ops.convert_to_tensor(weights, name="weights")
      return gen_math_ops.unsorted_segment_sum(weights, arr, output_size)
    weights = constant_op.constant([], dtype)
    return gen_math_ops.bincount(arr, output_size, weights)


@tf_export(v1=["math.bincount", "bincount"])
@deprecation.deprecated_endpoints("bincount")
def bincount_v1(arr,
                weights=None,
                minlength=None,
                maxlength=None,
                dtype=dtypes.int32):
  """Counts the number of occurrences of each value in an integer array.

  If `minlength` and `maxlength` are not given, returns a vector with length
  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.
  If `weights` are non-None, then index `i` of the output stores the sum of the
  value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Args:
    arr: An int32 tensor of non-negative values.
    weights: If non-None, must be the same shape as arr. For each value in
      `arr`, the bin will be incremented by the corresponding weight instead of
      1.
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `arr` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    dtype: If `weights` is None, determines the type of the output bins.

  Returns:
    A vector with the same dtype as `weights` or the given `dtype`. The bin
    values.
  """
  return bincount(arr, weights, minlength, maxlength, dtype)


@tf_export("math.cumsum", "cumsum")
def cumsum(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumsum([a, b, c])  # [a, a + b, a + b + c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is performed
  instead:

  ```python
  tf.cumsum([a, b, c], exclusive=True)  # [0, a, a + b]
  ```

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:

  ```python
  tf.cumsum([a, b, c], reverse=True)  # [a + b + c, b + c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumsum([a, b, c], exclusive=True, reverse=True)  # [b + c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: If `True`, perform exclusive cumsum.
    reverse: A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Cumsum", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.cumsum(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)


@tf_export("math.cumprod", v1=["math.cumprod", "cumprod"])
@deprecation.deprecated_endpoints("cumprod")
def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumprod, which means that the
  first element of the input is identical to the first element of the output:

  ```python
  tf.math.cumprod([a, b, c])  # [a, a * b, a * b * c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  performed
  instead:

  ```python
  tf.math.cumprod([a, b, c], exclusive=True)  # [1, a, a * b]
  ```

  By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  opposite direction:

  ```python
  tf.math.cumprod([a, b, c], reverse=True)  # [a * b * c, b * c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.
  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.math.cumprod([a, b, c], exclusive=True, reverse=True)  # [b * c, c, 1]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: If `True`, perform exclusive cumprod.
    reverse: A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Cumprod", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.cumprod(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)


@tf_export("math.conj", v1=["math.conj", "conj"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("conj")
def conj(x, name=None):
  r"""Returns the complex conjugate of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `input`. The
  complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
  real part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

      # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
      tf.math.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]

  If `x` is real, it is returned unchanged.

  Args:
    x: `Tensor` to conjugate.  Must have numeric or variant type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that is the conjugate of `x` (with the same type).

  Raises:
    TypeError: If `x` is not a numeric tensor.
  """
  if isinstance(x, ops.Tensor):
    dt = x.dtype
    if dt.is_floating or dt.is_integer:
      return x
  with ops.name_scope(name, "Conj", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.is_complex or x.dtype == dtypes.variant:
      return gen_math_ops.conj(x, name=name)
    elif x.dtype.is_floating or x.dtype.is_integer:
      return x
    else:
      raise TypeError(
          "Expected numeric or variant tensor, got dtype %r" % x.dtype)


def _BroadcastShape(op):
  """Common shape function for binary operators that broadcast their inputs."""
  return [
      common_shapes.broadcast_shape(op.inputs[0].get_shape(),
                                    op.inputs[1].get_shape())
  ]


def reduced_shape(input_shape, axes):
  """Helper function for reduction ops.

  Args:
    input_shape: 1-D Tensor, the shape of the Tensor being reduced.
    axes: 1-D Tensor, the reduction axes.
  Returns:
    A 1-D Tensor, the output shape as if keepdims were set to True.
  """
  # Example:
  # cast needed for SparseTensor reductions
  if context.executing_eagerly():
    input_shape = input_shape.numpy()
    axes = axes.numpy()
    input_shape[axes] = 1
    return input_shape

  input_shape = cast(input_shape, dtypes.int32)  # [2, 3, 5, 7]
  axes = cast(axes, dtypes.int32)  # [1, 2]

  input_rank = array_ops.size(input_shape)  # 4
  axes = (axes + input_rank) % input_rank
  axes_shape = array_ops.shape(axes)  # [2]
  return gen_data_flow_ops.dynamic_stitch(  # [2, 1, 1, 7]
      [
          range(input_rank),  # [0, 1, 2, 3]
          axes
      ],  # [1, 2]
      [
          input_shape,  # [2, 3, 5, 7]
          array_ops.fill(axes_shape, 1)
      ])  # [1, 1]


def _unsorted_segment_N(data, segment_ids, num_segments):
  """ Helper function for unsorted_segment_mean/_sqrtN. Computes the number
      of segment entries with 0-entries set to 1 to allow division by N.
  """
  # bincount doesn't support negative indices so we use unsorted_segment_sum
  segment_ids_shape = array_ops.shape_internal(segment_ids)
  ones_tensor = array_ops.ones(segment_ids_shape, dtype=data.dtype)
  N = gen_math_ops.unsorted_segment_sum(ones_tensor, segment_ids, num_segments)
  # add dimensions for all non-reduced axes
  ndims_output = data.shape.ndims - segment_ids.shape.ndims
  broadcast_shape = [num_segments] + [1] * ndims_output
  N = array_ops.reshape(N, broadcast_shape)
  return gen_math_ops.maximum(N, 1)


@tf_export(
    "math.unsorted_segment_mean",
    v1=["math.unsorted_segment_mean", "unsorted_segment_mean"])
@deprecation.deprecated_endpoints("unsorted_segment_mean")
@dispatch.add_dispatch_support
def unsorted_segment_mean(data, segment_ids, num_segments, name=None):
  r"""Computes the mean along segments of a tensor.

  Read [the section on
  segmentation](https://tensorflow.org/api_guides/python/math_ops#segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [here](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Instead of computing the sum over segments, it computes the mean of all
  entries belonging to a segment such that:

  \\(output_i = 1/N_i \sum_{j...} data[j...]\\) where the sum is over tuples
  `j...` such that `segment_ids[j...] == i` with \\N_i\\ being the number of
  occurrences of id \\i\\.

  If there is no entry for a given segment ID `i`, it outputs 0.

  If the given segment ID `i` is negative, the value is dropped and will not
  be added to the sum of the segment.

  Args:
    data: A `Tensor` with floating point or complex dtype.
    segment_ids: An integer tensor whose shape is a prefix of `data.shape`.
    num_segments: An integer scalar `Tensor`.  The number of distinct
      segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.  Has same shape as data, except for the first `segment_ids.rank`
    dimensions, which are replaced with a single dimension which has size
   `num_segments`.
  """
  with ops.name_scope(name, "UnsortedSegmentMean"):
    data = ops.convert_to_tensor(data)
    segment_ids = ops.convert_to_tensor(segment_ids)
    N = _unsorted_segment_N(data, segment_ids, num_segments)
    summed = gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments)
    return summed / N


@tf_export(
    "math.unsorted_segment_sqrt_n",
    v1=["math.unsorted_segment_sqrt_n", "unsorted_segment_sqrt_n"])
@deprecation.deprecated_endpoints("unsorted_segment_sqrt_n")
@dispatch.add_dispatch_support
def unsorted_segment_sqrt_n(data, segment_ids, num_segments, name=None):
  r"""Computes the sum along segments of a tensor divided by the sqrt(N).

  Read [the section on
  segmentation](https://tensorflow.org/api_guides/python/math_ops#segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [here](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Additionally to computing the sum over segments, it divides the results by
  sqrt(N).

  \\(output_i = 1/sqrt(N_i) \sum_{j...} data[j...]\\) where the sum is over
  tuples `j...` such that `segment_ids[j...] == i` with \\N_i\\ being the
  number of occurrences of id \\i\\.

  If there is no entry for a given segment ID `i`, it outputs 0.

  Note that this op only supports floating point and complex dtypes,
  due to tf.sqrt only supporting these types.

  If the given segment ID `i` is negative, the value is dropped and will not
  be added to the sum of the segment.

  Args:
    data: A `Tensor` with floating point or complex dtype.
    segment_ids: An integer tensor whose shape is a prefix of `data.shape`.
    num_segments: An integer scalar `Tensor`.  The number of distinct
      segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.  Has same shape as data, except for the first `segment_ids.rank`
    dimensions, which are replaced with a single dimension which has size
   `num_segments`.
  """
  with ops.name_scope(name, "UnsortedSegmentSqrtN"):
    data = ops.convert_to_tensor(data)
    segment_ids = ops.convert_to_tensor(segment_ids)
    N = _unsorted_segment_N(data, segment_ids, num_segments)
    summed = gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments)
    return summed / gen_math_ops.sqrt(N)


@tf_export(v1=["sparse.segment_sum", "sparse_segment_sum"])
@deprecation.deprecated_endpoints("sparse_segment_sum")
def sparse_segment_sum(data, indices, segment_ids, name=None,
                       num_segments=None):
  r"""Computes the sum along sparse segments of a tensor.

  Read [the section on
  segmentation](https://tensorflow.org/api_guides/python/math_ops#Segmentation)
  for an explanation of segments.

  Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.
  `segment_ids` is allowed to have missing ids, in which case the output will
  be zeros at those indices. In those cases `num_segments` is used to determine
  the size of the output.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  # => [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  # => [[ 1  2  3  4]
  #     [-1 -2 -3 -4]]

  # With missing segment ids.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 2]),
                        num_segments=4)
  # => [[ 1  2  3  4]
  #     [ 0  0  0  0]
  #     [-1 -2 -3 -4]
  #     [ 0  0  0  0]]

  # Select all rows, two segments.
  tf.sparse.segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  # => [[0 0 0 0]
  #     [5 6 7 8]]

  # Which is equivalent to:
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`.
      Values should be sorted and can be repeated.
    name: A name for the operation (optional).
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.

  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
  if num_segments is not None:
    return gen_math_ops.sparse_segment_sum_with_num_segments(
        data=data,
        indices=indices,
        segment_ids=segment_ids,
        num_segments=num_segments,
        name=name)
  else:
    return gen_math_ops.sparse_segment_sum(
        data=data, indices=indices, segment_ids=segment_ids, name=name)


@tf_export("sparse.segment_sum", v1=[])
def sparse_segment_sum_v2(data,
                          indices,
                          segment_ids,
                          num_segments=None,
                          name=None):
  return sparse_segment_mean(
      data, indices, segment_ids, name=name, num_segments=num_segments)


@tf_export(v1=["sparse.segment_mean", "sparse_segment_mean"])
@deprecation.deprecated_endpoints("sparse_segment_mean")
def sparse_segment_mean(data,
                        indices,
                        segment_ids,
                        name=None,
                        num_segments=None):
  r"""Computes the mean along sparse segments of a tensor.

  Read [the section on
  segmentation](https://tensorflow.org/api_guides/python/math_ops#Segmentation)
  for an explanation of segments.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.
  `segment_ids` is allowed to have missing ids, in which case the output will
  be zeros at those indices. In those cases `num_segments` is used to determine
  the size of the output.

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`.
      Values should be sorted and can be repeated.
    name: A name for the operation (optional).
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.

  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
  if num_segments is not None:
    return gen_math_ops.sparse_segment_mean_with_num_segments(
        data=data,
        indices=indices,
        segment_ids=segment_ids,
        num_segments=num_segments,
        name=name)
  else:
    return gen_math_ops.sparse_segment_mean(
        data=data, indices=indices, segment_ids=segment_ids, name=name)


@tf_export("sparse.segment_mean", v1=[])
def sparse_segment_mean_v2(data,
                           indices,
                           segment_ids,
                           num_segments=None,
                           name=None):
  r"""Computes the mean along sparse segments of a tensor.

  Read [the section on
  segmentation](https://tensorflow.org/api_guides/python/math_ops#Segmentation)
  for an explanation of segments.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.
  `segment_ids` is allowed to have missing ids, in which case the output will
  be zeros at those indices. In those cases `num_segments` is used to determine
  the size of the output.

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
  return sparse_segment_mean(
      data, indices, segment_ids, name=name, num_segments=num_segments)


@tf_export(v1=["sparse.segment_sqrt_n", "sparse_segment_sqrt_n"])
@deprecation.deprecated_endpoints("sparse_segment_sqrt_n")
def sparse_segment_sqrt_n(data,
                          indices,
                          segment_ids,
                          name=None,
                          num_segments=None):
  r"""Computes the sum along sparse segments of a tensor divided by the sqrt(N).

  `N` is the size of the segment being reduced.

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`.
      Values should be sorted and can be repeated.
    name: A name for the operation (optional).
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.

  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
  if num_segments is not None:
    return gen_math_ops.sparse_segment_sqrt_n_with_num_segments(
        data=data,
        indices=indices,
        segment_ids=segment_ids,
        num_segments=num_segments,
        name=name)
  else:
    return gen_math_ops.sparse_segment_sqrt_n(
        data=data, indices=indices, segment_ids=segment_ids, name=name)


@tf_export("sparse.segment_sqrt_n", v1=[])
def sparse_segment_sqrt_n_v2(data,
                             indices,
                             segment_ids,
                             num_segments=None,
                             name=None):
  r"""Computes the sum along sparse segments of a tensor divided by the sqrt(N).

  `N` is the size of the segment being reduced.

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
  return sparse_segment_sqrt_n(
      data, indices, segment_ids, name=name, num_segments=num_segments)


@tf_export("tensordot", "linalg.tensordot")
def tensordot(a, b, axes, name=None):
  r"""Tensor contraction of a and b along specified axes.

  Tensordot (also known as tensor contraction) sums the product of elements
  from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
  The lists `a_axes` and `b_axes` specify those pairs of axes along which to
  contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
  as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
  `a_axes` and `b_axes` must have identical length and consist of unique
  integers that specify valid axes for each of the tensors.

  This operation corresponds to `numpy.tensordot(a, b, axes)`.

  Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
  is equivalent to matrix multiplication.

  Example 2: When `a` and `b` are matrices (order 2), the case
  `axes = [[1], [0]]` is equivalent to matrix multiplication.

  Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
  tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
  \\(c_{jklm}\\) whose entry
  corresponding to the indices \\((j,k,l,m)\\) is given by:

  \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).

  In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.

  Args:
    a: `Tensor` of type `float32` or `float64`.
    b: `Tensor` with the same type as `a`.
    axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
      If axes is a scalar, sum over the last N axes of a and the first N axes of
      b in order. If axes is a list or `Tensor` the first and second row contain
      the set of unique integers specifying axes along which the contraction is
      computed, for `a` and `b`, respectively. The number of axes for `a` and
      `b` must be equal.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `a`.

  Raises:
    ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
    IndexError: If the values in axes exceed the rank of the corresponding
      tensor.
  """

  def _tensordot_reshape(a, axes, flipped=False):
    """Helper method to perform transpose and reshape for contraction op.

    This method is helpful in reducing `math_ops.tensordot` to `math_ops.matmul`
    using `array_ops.transpose` and `array_ops.reshape`. The method takes a
    tensor and performs the correct transpose and reshape operation for a given
    set of indices. It returns the reshaped tensor as well as a list of indices
    necessary to reshape the tensor again after matrix multiplication.

    Args:
      a: `Tensor`.
      axes: List or `int32` `Tensor` of unique indices specifying valid axes of
       `a`.
      flipped: An optional `bool`. Defaults to `False`. If `True`, the method
        assumes that `a` is the second argument in the contraction operation.

    Returns:
      A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
      the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
      either a list of integers or an `int32` `Tensor`, depending on whether
      the shape of a is fully specified, and free_dims_static is either a list
      of integers and None values, or None, representing the inferred
      static shape of the free dimensions
    """
    if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
      shape_a = a.get_shape().as_list()
      axes = [i if i >= 0 else i + len(shape_a) for i in axes]
      free = [i for i in xrange(len(shape_a)) if i not in axes]
      free_dims = [shape_a[i] for i in free]
      prod_free = int(np.prod([shape_a[i] for i in free]))
      prod_axes = int(np.prod([shape_a[i] for i in axes]))
      perm = list(axes) + free if flipped else free + list(axes)
      new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
      reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
      return reshaped_a, free_dims, free_dims
    else:
      if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
        shape_a = a.get_shape().as_list()
        axes = [i if i >= 0 else i + len(shape_a) for i in axes]
        free = [i for i in xrange(len(shape_a)) if i not in axes]
        axes_dims = [shape_a[i] for i in axes]
        free_dims = [shape_a[i] for i in free]
        free_dims_static = free_dims
        axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name="axes")
        free = ops.convert_to_tensor(free, dtype=dtypes.int32, name="free")
        shape_a = array_ops.shape(a)
      else:
        free_dims_static = None
        shape_a = array_ops.shape(a)
        rank_a = array_ops.rank(a)
        axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name="axes")
        axes = array_ops.where(axes >= 0, axes, axes + rank_a)
        free, _ = array_ops.setdiff1d(range(rank_a), axes)
      free_dims = array_ops.gather(shape_a, free)
      axes_dims = array_ops.gather(shape_a, axes)
      prod_free_dims = reduce_prod(free_dims)
      prod_axes_dims = reduce_prod(axes_dims)
      if flipped:
        perm = array_ops.concat([axes, free], 0)
        new_shape = array_ops.stack([prod_axes_dims, prod_free_dims])
      else:
        perm = array_ops.concat([free, axes], 0)
        new_shape = array_ops.stack([prod_free_dims, prod_axes_dims])
      reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
      return reshaped_a, free_dims, free_dims_static

  def _tensordot_axes(a, axes):
    """Generates two sets of contraction axes for the two tensor arguments."""
    a_shape = a.get_shape()
    if isinstance(axes, compat.integral_types):
      if axes < 0:
        raise ValueError("'axes' must be at least 0.")
      if a_shape.ndims is not None:
        if axes > a_shape.ndims:
          raise ValueError("'axes' must not be larger than the number of "
                           "dimensions of tensor %s." % a)
        return (list(xrange(a_shape.ndims - axes, a_shape.ndims)),
                list(xrange(axes)))
      else:
        rank = array_ops.rank(a)
        return (range(rank - axes, rank, dtype=dtypes.int32),
                range(axes, dtype=dtypes.int32))
    elif isinstance(axes, (list, tuple)):
      if len(axes) != 2:
        raise ValueError("'axes' must be an integer or have length 2.")
      a_axes = axes[0]
      b_axes = axes[1]
      if isinstance(a_axes, compat.integral_types) and \
          isinstance(b_axes, compat.integral_types):
        a_axes = [a_axes]
        b_axes = [b_axes]
      if len(a_axes) != len(b_axes):
        raise ValueError(
            "Different number of contraction axes 'a' and 'b', %s != %s." %
            (len(a_axes), len(b_axes)))
      return a_axes, b_axes
    else:
      axes = ops.convert_to_tensor(axes, name="axes", dtype=dtypes.int32)
      return axes[0], axes[1]

  with ops.name_scope(name, "Tensordot", [a, b, axes]) as name:
    a = ops.convert_to_tensor(a, name="a")
    b = ops.convert_to_tensor(b, name="b")
    a_axes, b_axes = _tensordot_axes(a, axes)
    a_reshape, a_free_dims, a_free_dims_static = _tensordot_reshape(a, a_axes)
    b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
        b, b_axes, True)
    ab_matmul = matmul(a_reshape, b_reshape)
    if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
      return array_ops.reshape(ab_matmul, a_free_dims + b_free_dims, name=name)
    else:
      a_free_dims = ops.convert_to_tensor(a_free_dims, dtype=dtypes.int32)
      b_free_dims = ops.convert_to_tensor(b_free_dims, dtype=dtypes.int32)
      product = array_ops.reshape(
          ab_matmul, array_ops.concat([a_free_dims, b_free_dims], 0), name=name)
      if a_free_dims_static is not None and b_free_dims_static is not None:
        product.set_shape(a_free_dims_static + b_free_dims_static)
      return product


@tf_export("math.polyval")
def polyval(coeffs, x, name=None):
  r"""Computes the elementwise value of a polynomial.

  If `x` is a tensor and `coeffs` is a list n + 1 tensors, this function returns
  the value of the n-th order polynomial

     p(x) = coeffs[n-1] + coeffs[n-2] * x + ...  + coeffs[0] * x**(n-1)

  evaluated using Horner's method, i.e.

     p(x) = coeffs[n-1] + x * (coeffs[n-2] + ... + x * (coeffs[1] +
            x * coeffs[0]))

  Args:
    coeffs: A list of `Tensor` representing the coefficients of the polynomial.
    x: A `Tensor` representing the variable of the polynomial.
    name: A name for the operation (optional).

  Returns:
    A `tensor` of the shape as the expression p(x) with usual broadcasting rules
    for element-wise addition and multiplication applied.

  @compatibility(numpy)
  Equivalent to numpy.polyval.
  @end_compatibility
  """

  with ops.name_scope(name, "polyval", nest.flatten(coeffs) + [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if len(coeffs) < 1:
      return array_ops.zeros_like(x, name=name)
    coeffs = [
        ops.convert_to_tensor(coeff, name=("coeff_%d" % index))
        for index, coeff in enumerate(coeffs)
    ]
    p = coeffs[0]
    for c in coeffs[1:]:
      p = c + p * x
    return p
