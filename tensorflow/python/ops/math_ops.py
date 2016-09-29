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

"""Note: Elementwise binary operations in TensorFlow follow [numpy-style
broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Arithmetic Operators

TensorFlow provides several operations that you can use to add basic arithmetic
operators to your graph.

@@add
@@sub
@@mul
@@scalar_mul
@@div
@@truediv
@@floordiv
@@mod
@@cross

## Basic Math Functions

TensorFlow provides several operations that you can use to add basic
mathematical functions to your graph.

@@add_n
@@abs
@@neg
@@sign
@@inv
@@square
@@round
@@sqrt
@@rsqrt
@@pow
@@exp
@@log
@@ceil
@@floor
@@maximum
@@minimum
@@cos
@@sin
@@lbeta
@@tan
@@acos
@@asin
@@atan
@@lgamma
@@digamma
@@erf
@@erfc
@@squared_difference
@@igamma
@@igammac
@@zeta
@@polygamma
@@betainc

## Matrix Math Functions

TensorFlow provides several operations that you can use to add linear algebra
functions on matrices to your graph.

@@diag
@@diag_part
@@trace
@@transpose

@@matrix_diag
@@matrix_diag_part
@@matrix_band_part
@@matrix_set_diag
@@matrix_transpose

@@matmul
@@batch_matmul

@@matrix_determinant
@@matrix_inverse
@@cholesky
@@cholesky_solve
@@matrix_solve
@@matrix_triangular_solve
@@matrix_solve_ls
@@self_adjoint_eig
@@self_adjoint_eigvals
@@svd

## Complex Number Functions

TensorFlow provides several operations that you can use to add complex number
functions to your graph.

@@complex
@@complex_abs
@@conj
@@imag
@@real

## Fourier Transform Functions

TensorFlow provides several operations that you can use to add discrete
Fourier transform functions to your graph.

@@fft
@@ifft
@@fft2d
@@ifft2d
@@fft3d
@@ifft3d

## Reduction

TensorFlow provides several operations that you can use to perform
common math computations that reduce various dimensions of a tensor.

@@reduce_sum
@@reduce_prod
@@reduce_min
@@reduce_max
@@reduce_mean
@@reduce_all
@@reduce_any
@@reduce_logsumexp

@@accumulate_n

@@einsum

## Scan

TensorFlow provides several operations that you can use to perform scans
(running totals) across one axis of a tensor.

@@cumsum
@@cumprod

## Segmentation

TensorFlow provides several operations that you can use to perform common
math computations on tensor segments.
Here a segmentation is a partitioning of a tensor along
the first dimension, i.e. it  defines a mapping from the first dimension onto
`segment_ids`. The `segment_ids` tensor should be the size of
the first dimension, `d0`, with consecutive IDs in the range `0` to `k`,
where `k<d0`.
In particular, a segmentation of a matrix tensor is a mapping of rows to
segments.

For example:

```python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
tf.segment_sum(c, tf.constant([0, 0, 1]))
  ==>  [[0 0 0 0]
        [5 6 7 8]]
```

@@segment_sum
@@segment_prod
@@segment_min
@@segment_max
@@segment_mean

@@unsorted_segment_sum

@@sparse_segment_sum
@@sparse_segment_mean
@@sparse_segment_sqrt_n


## Sequence Comparison and Indexing

TensorFlow provides several operations that you can use to add sequence
comparison and index extraction to your graph. You can use these operations to
determine sequence differences and determine the indexes of specific values in
a tensor.

@@argmin
@@argmax

@@listdiff
@@where
@@unique

@@edit_distance

@@invert_permutation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import state_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_math_ops import *
# pylint: enable=wildcard-import

# Aliases for some automatically-generated names.
argmax = gen_math_ops.arg_max
argmin = gen_math_ops.arg_min
linspace = gen_math_ops.lin_space


# pylint: disable=anomalous-backslash-in-string,protected-access
def abs(x, name=None):
  """Computes the absolute value of a tensor.

  Given a tensor of real numbers `x`, this operation returns a tensor
  containing the absolute value of each element in `x`. For example, if x is
  an input element and y is an output element, this operation computes
  \\\\(y = |x|\\\\).

  See [`tf.complex_abs()`](#tf_complex_abs) to compute the absolute value of a complex
  number.

  Args:
    x: A `Tensor` or `SparseTensor` of type `float32`, `float64`, `int32`, or
      `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` the same size and type as `x` with absolute
      values.
  """
  with ops.name_scope(name, "Abs", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      if x.values.dtype in (dtypes.complex64, dtypes.complex128):
        x_abs = gen_math_ops.complex_abs(x.values,
            Tout=x.values.dtype.real_dtype, name=name)
        return ops.SparseTensor(indices=x.indices, values=x_abs, shape=x.shape)
      x_abs = gen_math_ops._abs(x.values, name=name)
      return ops.SparseTensor(indices=x.indices, values=x_abs, shape=x.shape)
    else:
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype in (dtypes.complex64, dtypes.complex128):
        return gen_math_ops.complex_abs(x, Tout=x.dtype.real_dtype, name=name)
      return gen_math_ops._abs(x, name=name)


def neg(x, name=None):
  """Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  with ops.name_scope(name, "Neg", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      x_neg = gen_math_ops.neg(x.values, name=name)
      return ops.SparseTensor(indices=x.indices, values=x_neg, shape=x.shape)
    else:
      return gen_math_ops.neg(x, name=name)


def sign(x, name=None):
  """Returns an element-wise indication of the sign of a number.

  `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

  For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  with ops.name_scope(name, "Sign", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      x_sign = gen_math_ops.sign(x.values, name=name)
      return ops.SparseTensor(indices=x.indices, values=x_sign, shape=x.shape)
    else:
      return gen_math_ops.sign(x, name=name)


def square(x, name=None):
  """Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Square", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      x_square = gen_math_ops.square(x.values, name=name)
      return ops.SparseTensor(indices=x.indices, values=x_square, shape=x.shape)
    else:
      return gen_math_ops.square(x, name=name)


def sqrt(x, name=None):
  """Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  with ops.name_scope(name, "Sqrt", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      x_sqrt = gen_math_ops.sqrt(x.values, name=name)
      return ops.SparseTensor(indices=x.indices, values=x_sqrt, shape=x.shape)
    else:
      return gen_math_ops.sqrt(x, name=name)


def erf(x, name=None):
  """Computes the Gauss error function of `x` element-wise.

  Args:
    x: A `Tensor` of `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  with ops.name_scope(name, "Erf", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      x_erf = gen_math_ops.erf(x.values, name=name)
      return ops.SparseTensor(indices=x.indices, values=x_erf, shape=x.shape)
    else:
      return gen_math_ops.erf(x, name=name)


def complex_abs(x, name=None):
  r"""Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float32` or `float64` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\(a + bj\\). The
  absolute value is computed as \\( \sqrt{a^2 + b^2}\\).

  For example:

  ```
  # tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
  tf.complex_abs(x) ==> [5.25594902, 6.60492229]
  ```

  Args:
    x: A `Tensor` of type `complex64` or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  return gen_math_ops.complex_abs(x, Tout=x.dtype.real_dtype, name=name)


def scalar_mul(scalar, x):
  """Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

  Intended for use in gradient code which might deal with `IndexedSlices`
  objects, which are easy to multiply by a scalar but more expensive to
  multiply with arbitrary tensors.

  Args:
    scalar: A 0-D scalar `Tensor`. Must have known shape.
    x: A `Tensor` or `IndexedSlices` to be scaled.

  Returns:
    `scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.

  Raises:
    ValueError: if scalar is not a 0-D `scalar`.
  """
  scalar = ops.convert_to_tensor(scalar, dtype=x.dtype.base_dtype,
                                 name="scalar")
  shape = scalar.get_shape()
  if shape.ndims == 0:
    if isinstance(x, ops.IndexedSlices):
      return ops.IndexedSlices(scalar * x.values, x.indices, x.dense_shape)
    else:
      return scalar * x
  else:
    raise ValueError("Only scalar multiply works, got shape %s" % shape)


def pow(x, y, name=None):
  """Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\\\(x^y\\\\) for
  corresponding elements in `x` and `y`. For example:

  ```
  # tensor 'x' is [[2, 2], [3, 3]]
  # tensor 'y' is [[8, 16], [2, 3]]
  tf.pow(x, y) ==> [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor` of type `float32`, `float64`, `int32`, `int64`, `complex64`,
     or `complex128`.
    y: A `Tensor` of type `float32`, `float64`, `int32`, `int64`, `complex64`,
     or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.
  """
  with ops.name_scope(name, "Pow", [x]) as name:
    return gen_math_ops._pow(x, y, name=name)


def complex(real, imag, name=None):
  """Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
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


def real(input, name=None):
  """Returns the real part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float32` or `float64` that is the real part of each element in `input`.
  All elements in `input` must be complex numbers of the form \\(a + bj\\),
  where *a* is the real part returned by this operation and *b* is the
  imaginary part.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.real(input) ==> [-2.25, 3.25]
  ```

  If `input` is already real, it is returned unchanged.

  Args:
    input: A `Tensor`. Must have numeric type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Real", [input]) as name:
    real_dtype = input.dtype.real_dtype
    if input.dtype.base_dtype == real_dtype:
      return input
    return gen_math_ops.real(input, Tout=real_dtype, name=name)


def imag(input, name=None):
  """Returns the imaginary part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float32` or `float64` that is the imaginary part of each element in
  `input`. All elements in `input` must be complex numbers of the form \\(a +
  bj\\), where *a* is the real part and *b* is the imaginary part returned by
  this operation.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.imag(input) ==> [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Imag", [input]) as name:
    return gen_math_ops.imag(input, Tout=input.dtype.real_dtype, name=name)


def round(x, name=None):
  """Rounds the values of a tensor to the nearest integer, element-wise.

  For example:

  ```python
  # 'a' is [0.9, 2.5, 2.3, -4.4]
  tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ]
  ```

  Args:
    x: A `Tensor` of type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.
  """
  x = ops.convert_to_tensor(x, name="x")
  if x.dtype.is_integer:
    return x
  else:
    return gen_math_ops.floor(x + 0.5, name=name)


def cast(x, dtype, name=None):
  """Casts a tensor to a new type.

  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor`) to `dtype`.

  For example:

  ```python
  # tensor `a` is [1.8, 2.2], dtype=tf.float
  tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
  ```

  Args:
    x: A `Tensor` or `SparseTensor`.
    dtype: The destination type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x`.

  Raises:
    TypeError: If `x` cannot be cast to the `dtype`.
  """
  base_type = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "Cast", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      values_cast = cast(x.values, base_type, name=name)
      return ops.SparseTensor(x.indices, values_cast, x.shape)
    else:
      # TODO(touts): Handle what Josh said.
      #
      # Could return ops.convert_to_tensor(x, dtype=dtype, ...)  here, but that
      # allows some conversions that cast() can't do, e.g.  casting numbers to
      # strings.
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype.base_dtype == base_type:
        return x
      return gen_math_ops.cast(x, base_type, name=name)


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
      value = gen_math_ops.maximum(value, ops.convert_to_tensor(
          dtype.min, dtype=value.dtype, name="min"))
    if value.dtype.max > dtype.max:
      value = gen_math_ops.minimum(value, ops.convert_to_tensor(
          dtype.max, dtype=value.dtype, name="max"))
    return cast(value, dtype, name=name)


def to_float(x, name="ToFloat"):
  """Casts a tensor to type `float32`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `float32`.

  Raises:
    TypeError: If `x` cannot be cast to the `float32`.
  """
  return cast(x, dtypes.float32, name=name)


def to_double(x, name="ToDouble"):
  """Casts a tensor to type `float64`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `float64`.

  Raises:
    TypeError: If `x` cannot be cast to the `float64`.
  """
  return cast(x, dtypes.float64, name=name)


def to_int32(x, name="ToInt32"):
  """Casts a tensor to type `int32`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `int32`.

  Raises:
    TypeError: If `x` cannot be cast to the `int32`.
  """
  return cast(x, dtypes.int32, name=name)


def to_int64(x, name="ToInt64"):
  """Casts a tensor to type `int64`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `int64`.

  Raises:
    TypeError: If `x` cannot be cast to the `int64`.
  """
  return cast(x, dtypes.int64, name=name)


def to_bfloat16(x, name="ToBFloat16"):
  """Casts a tensor to type `bfloat16`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `bfloat16`.

  Raises:
    TypeError: If `x` cannot be cast to the `bfloat16`.
  """
  return cast(x, dtypes.bfloat16, name=name)


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
      if not isinstance(y, ops.SparseTensor):
        y = ops.convert_to_tensor(y, dtype=x.dtype.base_dtype, name="y")
      return func(x, y, name=name)

  def binary_op_wrapper_sparse(sp_x, y):
    with ops.name_scope(None, op_name, [sp_x, y]) as name:
      y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name="y")
      return ops.SparseTensor(sp_x.indices, func(sp_x.indices, sp_x.values,
                                                 sp_x.shape, y, name=name),
                              sp_x.shape)

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
    return gen_sparse_ops.sparse_dense_cwise_div(sp_indices, sp_values,
                                                 sp_shape, y, name=name)


def truediv(x, y, name=None):
  """Divides x / y elementwise, always producing floating point results.

  The same as `tf.div` for floating point arguments, but casts integer arguments
  to floating point before dividing so that the result is always floating point.
  This op is generated by normal `x / y` division in Python 3 and in Python 2.7
  with `from __future__ import division`.  If you want integer division that
  rounds down, use `x // y` or `tf.floordiv`.

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
    return gen_math_ops.div(x, y, name=name)


def floordiv(x, y, name=None):
  """Divides `x / y` elementwise, rounding down for floating point.

  The same as `tf.div(x,y)` for integers, but uses `tf.floor(tf.div(x,y))` for
  floating point arguments so that the result is always an integer (though
  possibly an integer represented as floating point).  This op is generated by
  `x // y` floor division in Python 3 and in Python 2.7 with
  `from __future__ import division`.

  Note that for efficiency, `floordiv` uses C semantics for negative numbers
  (unlike Python and Numpy).

  `x` and `y` must have the same type, and the result will have the same type
  as well.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` rounded down (except possibly towards zero for negative integers).

  Raises:
    TypeError: If the inputs are complex.
  """
  with ops.name_scope(name, "floordiv", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    dtype = x.dtype
    if dtype.is_floating:
      return gen_math_ops.floor(gen_math_ops.div(x, y), name=name)
    else:
      if not dtype.is_integer:
        raise TypeError("Expected floating point or integer, got %r" % dtype)
      return gen_math_ops.div(x, y, name=name)


def _mul_dispatch(x, y, name=None):
  """Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse"."""
  is_tensor_y = isinstance(y, ops.Tensor)
  if is_tensor_y:
    return gen_math_ops.mul(x, y, name=name)
  else:
    assert isinstance(y, ops.SparseTensor)  # Case: Dense * Sparse.
    new_vals = gen_sparse_ops.sparse_dense_cwise_mul(y.indices, y.values,
                                                     y.shape, x, name)
    return ops.SparseTensor(y.indices, new_vals, y.shape)


_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_div, "div",
                              ops.SparseTensor)
_OverrideBinaryOperatorHelper(_sparse_dense_truediv, "truediv",
                              ops.SparseTensor)
_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_mul, "mul",
                              ops.SparseTensor)


_OverrideBinaryOperatorHelper(gen_math_ops.add, "add")
_OverrideBinaryOperatorHelper(gen_math_ops.sub, "sub")
_OverrideBinaryOperatorHelper(_mul_dispatch, "mul")
_OverrideBinaryOperatorHelper(gen_math_ops.div, "div")
_OverrideBinaryOperatorHelper(truediv, "truediv")
_OverrideBinaryOperatorHelper(floordiv, "floordiv")
_OverrideBinaryOperatorHelper(gen_math_ops.mod, "mod")
_OverrideBinaryOperatorHelper(pow, "pow")


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


def range(start, limit=None, delta=1, name="range"):
  """Creates a sequence of integers.

  Creates a sequence of integers that begins at `start` and extends by
  increments of `delta` up to but not including `limit`.

  Like the Python builtin `range`, `start` defaults to 0, so that
  `range(n) = range(0, n)`.

  For example:

  ```python
  # 'start' is 3
  # 'limit' is 18
  # 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

  # 'limit' is 5
  tf.range(limit) ==> [0, 1, 2, 3, 4]
  ```

  Args:
    start: A 0-D (scalar) of type `int32`. Acts as first entry in the range if
      `limit` is not None; otherwise, acts as range limit and first entry
      defaults to 0.
    limit: A 0-D (scalar) of type `int32`. Upper limit of sequence,
      exclusive. If None, defaults to the value of `start` while the first
      entry of the range defaults to 0.
    delta: A 0-D `Tensor` (scalar) of type `int32`. Number that increments
      `start`. Defaults to 1.
    name: A name for the operation. Defaults to "range".

  Returns:
    An 1-D `int32` `Tensor`.
  """
  if limit is None:
    start, limit = 0, start
  return gen_math_ops._range(start, limit, delta, name=name)


@ops.RegisterShape("Range")
def _RangeShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[0, 1, 2])


# Reduction operations
def _ReductionDims(x, reduction_indices):
  """Returns range(0, rank(x)) if reduction_indices is None."""
  if reduction_indices is not None:
    return reduction_indices
  else:
    # Fast path: avoid creating Rank and Range ops if ndims is known.
    if isinstance(x, ops.Tensor) and x.get_shape().ndims is not None:
      return constant_op.constant(np.arange(x.get_shape().ndims),
                                  dtype=dtypes.int32)
    if (isinstance(x, ops.SparseTensor) and
        x.shape.get_shape().is_fully_defined()):
      rank = x.shape.get_shape()[0].value  # sparse.shape is an 1-D tensor.
      return constant_op.constant(np.arange(rank), dtype=dtypes.int32)

    # Otherwise, we rely on Range and Rank to do the right thing at run-time.
    return range(0, array_ops.rank(x))


def reduce_sum(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
  """Computes the sum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[1, 1, 1]
  #         [1, 1, 1]]
  tf.reduce_sum(x) ==> 6
  tf.reduce_sum(x, 0) ==> [2, 2, 2]
  tf.reduce_sum(x, 1) ==> [3, 3]
  tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
  tf.reduce_sum(x, [0, 1]) ==> 6
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return gen_math_ops._sum(input_tensor, _ReductionDims(input_tensor,
                                                        reduction_indices),
                           keep_dims, name=name)


def reduce_mean(input_tensor, reduction_indices=None, keep_dims=False,
                name=None):
  """Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[1., 1.]
  #         [2., 2.]]
  tf.reduce_mean(x) ==> 1.5
  tf.reduce_mean(x, 0) ==> [1.5, 1.5]
  tf.reduce_mean(x, 1) ==> [1.,  2.]
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return gen_math_ops._mean(input_tensor, _ReductionDims(input_tensor,
                                                         reduction_indices),
                            keep_dims, name=name)


def reduce_prod(input_tensor, reduction_indices=None, keep_dims=False,
                name=None):
  """Computes the product of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return gen_math_ops._prod(input_tensor, _ReductionDims(input_tensor,
                                                         reduction_indices),
                            keep_dims, name=name)


def reduce_min(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
  """Computes the minimum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return gen_math_ops._min(input_tensor, _ReductionDims(input_tensor,
                                                        reduction_indices),
                           keep_dims, name=name)


def reduce_max(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
  """Computes the maximum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return gen_math_ops._max(input_tensor, _ReductionDims(input_tensor,
                                                        reduction_indices),
                           keep_dims, name=name)


def reduce_all(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
  """Computes the "logical and" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[True,  True]
  #         [False, False]]
  tf.reduce_all(x) ==> False
  tf.reduce_all(x, 0) ==> [False, False]
  tf.reduce_all(x, 1) ==> [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return gen_math_ops._all(input_tensor, _ReductionDims(input_tensor,
                                                        reduction_indices),
                           keep_dims, name=name)


def reduce_any(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
  """Computes the "logical or" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[True,  True]
  #         [False, False]]
  tf.reduce_any(x) ==> True
  tf.reduce_any(x, 0) ==> [True, True]
  tf.reduce_any(x, 1) ==> [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return gen_math_ops._any(input_tensor, _ReductionDims(input_tensor,
                                                        reduction_indices),
                           keep_dims, name=name)


def reduce_logsumexp(input_tensor, reduction_indices=None, keep_dims=False,
                     name=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.

  For example:

  ```python
  # 'x' is [[0, 0, 0]]
  #         [0, 0, 0]]
  tf.reduce_logsumexp(x) ==> log(6)
  tf.reduce_logsumexp(x, 0) ==> [log(2), log(2), log(2)]
  tf.reduce_logsumexp(x, 1) ==> [log(3), log(3)]
  tf.reduce_logsumexp(x, 1, keep_dims=True) ==> [[log(3)], [log(3)]]
  tf.reduce_logsumexp(x, [0, 1]) ==> log(6)
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the default),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  with ops.name_scope(name, "ReduceLogSumExp", [input_tensor]) as name:
    my_max = array_ops.stop_gradient(
        reduce_max(input_tensor, reduction_indices, keep_dims=True))
    result = gen_math_ops.log(reduce_sum(
        gen_math_ops.exp(input_tensor - my_max),
        reduction_indices,
        keep_dims=True)) + my_max
    if not keep_dims:
      result = array_ops.squeeze(result, reduction_indices)
    return result


def trace(x, name=None):
  """ Compute the trace of a tensor `x`.

  `trace(x)` returns the sum along the main diagonal of each inner-most matrix
  in x. If x is of rank `k` with shape `[I, J, K, ..., L, M, N]`, then output
  is a tensor of rank `k-2` with dimensions `[I, J, K, ..., L]` where

  `output[i, j, k, ..., l] = trace(x[i, j, i, ..., l, :, :])`

  For example:

  ```python
  # 'x' is [[1, 2],
  #         [3, 4]]
  tf.trace(x) ==> 5

  # 'x' is [[1,2,3],
  #         [4,5,6],
  #         [7,8,9]]
  tf.trace(x) ==> 15

  # 'x' is [[[1,2,3],
  #          [4,5,6],
  #          [7,8,9]],
  #         [[-1,-2,-3],
  #          [-4,-5,-6],
  #          [-7,-8,-9]]]
  tf.trace(x) ==> [15,-15]
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


def matmul(a, b,
           transpose_a=False, transpose_b=False,
           a_is_sparse=False, b_is_sparse=False,
           name=None):
  """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

  The inputs must be two-dimensional matrices, with matching inner dimensions,
  possibly after transposition.

  Both matrices must be of the same type. The supported types are:
  `float32`, `float64`, `int32`, `complex64`.

  Either matrix can be transposed on the fly by setting the corresponding flag
  to `True`. This is `False` by default.

  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.

  For example:

  ```python
  # 2-D tensor `a`
  a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                        [4. 5. 6.]]
  # 2-D tensor `b`
  b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                           [9. 10.]
                                                           [11. 12.]]
  c = tf.matmul(a, b) => [[58 64]
                          [139 154]]
  ```

  Args:
    a: `Tensor` of type `float32`, `float64`, `int32` or `complex64`.
    b: `Tensor` with same type as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a`.
  """
  with ops.name_scope(name, "MatMul", [a, b]) as name:
    a = ops.convert_to_tensor(a, name="a")
    b = ops.convert_to_tensor(b, name="b")
    sparse_matmul_types = [dtypes.bfloat16, dtypes.float32]
    use_sparse_matmul = (a.dtype in sparse_matmul_types and
                         b.dtype in sparse_matmul_types and
                         (a_is_sparse or b_is_sparse))
    if dtypes.bfloat16 in (a.dtype, b.dtype):
      # matmul currently doesn't handle bfloat16 inputs.
      use_sparse_matmul = True
    if use_sparse_matmul:
      return sparse_matmul(a, b,
                           transpose_a=transpose_a,
                           transpose_b=transpose_b,
                           a_is_sparse=a_is_sparse,
                           b_is_sparse=b_is_sparse,
                           name=name)
    else:
      return gen_math_ops._mat_mul(a, b,
                                   transpose_a=transpose_a,
                                   transpose_b=transpose_b,
                                   name=name)

sparse_matmul = gen_math_ops._sparse_mat_mul
batch_matmul = gen_math_ops._batch_mat_mul

ops.RegisterShape("MatMul")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseMatMul")(common_shapes.call_cpp_shape_fn)


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


@ops.RegisterStatistics("MatMul", "weight_parameters")
def _calc_mat_mul_weight_parameters(graph, node):
  """Calculates the on-disk size of the weights for MatMul."""
  # We assume here that the weights are always in the second input to the op,
  # which is generally true by convention for fully-connected layers, but not
  # enforced or checked.
  weights_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                             node.input[1])
  weights_shape.assert_is_fully_defined()
  return ops.OpStats("weight_parameters",
                     (int(weights_shape[1]) * int(weights_shape[0])))


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
  with_int32_index = [o.indices for o in outputs
                      if o.indices.dtype == dtypes.int32]
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


def add_n(inputs, name=None):
  """Adds all input tensors element-wise.

  Args:
    inputs: A list of `Tensor` objects, each with same shape and type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """
  if not inputs or not isinstance(inputs, (list, tuple)):
    raise ValueError("inputs must be a list of at least one Tensor with the "
                     "same dtype and shape")
  inputs = ops.convert_n_to_tensor_or_indexed_slices(inputs)
  if not all(isinstance(x, ops.Tensor) for x in inputs):
    raise ValueError("inputs must be a list of at least one Tensor with the "
                     "same dtype and shape")

  if len(inputs) == 1:
    if name:
      return array_ops.identity(inputs[0], name=name)
    return inputs[0]
  return gen_math_ops._add_n(inputs, name=name)


def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
  """Returns the element-wise sum of a list of tensors.

  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
  otherwise, these are inferred.

  NOTE: This operation is not differentiable and cannot be used if inputs depend
  on trainable variables. Please use `tf.add_n` for such cases.

  For example:

  ```python
  # tensor 'a' is [[1, 2], [3, 4]]
  # tensor `b` is [[5, 0], [0, 6]]
  tf.accumulate_n([a, b, a]) ==> [[7, 4], [6, 14]]

  # Explicitly pass shape and type
  tf.accumulate_n([a, b, a], shape=[2, 2], tensor_dtype=tf.int32)
    ==> [[7, 4], [6, 14]]
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
  if not inputs or not isinstance(inputs, (list, tuple)):
    raise ValueError("inputs must be a list of at least one Tensor with the "
                     "same dtype and shape")
  inputs = ops.convert_n_to_tensor_or_indexed_slices(inputs)
  if not all(isinstance(x, ops.Tensor) for x in inputs):
    raise ValueError("inputs must be a list of at least one Tensor with the "
                     "same dtype and shape")
  if not all(x.dtype == inputs[0].dtype for x in inputs):
    raise ValueError("inputs must be a list of at least one Tensor with the "
                     "same dtype and shape")
  if shape is not None:
    shape = tensor_shape.as_shape(shape)
  else:
    shape = tensor_shape.unknown_shape()
  for input_tensor in inputs:
    if isinstance(input_tensor, ops.Tensor):
      shape = shape.merge_with(input_tensor.get_shape())
  if len(inputs) == 1:
    return inputs[0]
  if tensor_dtype is None:
    tensor_dtype = inputs[0].dtype
  with ops.name_scope(name, "AccumulateN", inputs) as name:
    var = gen_state_ops._temporary_variable(shape=tensor_shape.vector(0),
                                            dtype=tensor_dtype)
    with ops.colocate_with(var):
      zeros = array_ops.zeros_like(gen_control_flow_ops._merge(inputs)[0])
      zeros.set_shape(shape)
      ref = state_ops.assign(var, zeros, validate_shape=False)
      update_ops = [state_ops.assign_add(ref, input_tensor, use_locking=True)
                    for input_tensor in inputs]
      with ops.control_dependencies(update_ops):
        return gen_state_ops._destroy_temporary_variable(
            ref, var_name=var.op.name, name=name)


ops.RegisterShape("BatchMatMul")(common_shapes.call_cpp_shape_fn)


def sigmoid(x, name=None):
  """Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A Tensor with type `float32`, `float64`, `int32`, `complex64`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32`
      otherwise the return type is `quint8`.
  """
  with ops.name_scope(name, "Sigmoid", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops._sigmoid(x, name=name)


def tanh(x, name=None):
  """Computes hyperbolic tangent of `x` element-wise.

  Args:
    x: A Tensor or SparseTensor with type `float`, `double`, `int32`,
      `complex64`, `int64`, or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor or SparseTensor respectively with the same type as `x` if
    `x.dtype != qint32` otherwise the return type is `quint8`.
  """
  with ops.name_scope(name, "Tanh", [x]) as name:
    if isinstance(x, ops.SparseTensor):
      x_tanh = gen_math_ops._tanh(x.values, name=name)
      return ops.SparseTensor(indices=x.indices, values=x_tanh, shape=x.shape)
    else:
      return gen_math_ops._tanh(x, name=name)


def cumsum(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:
  ```prettyprint
  tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is performed
  instead:
  ```prettyprint
  tf.cumsum([a, b, c], exclusive=True) ==> [0, a, a + b]
  ```

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:
  ```prettyprint
  tf.cumsum([a, b, c], reverse=True) ==> [a + b + c, b + c, c]
  ```
  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:
  ```prettyprint
  tf.cumsum([a, b, c], exclusive=True, reverse=True) ==> [b + c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
       axis: A `Tensor` of type `int32` (default: 0).
       reverse: A `bool` (default: False).
       name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Cumsum", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.cumsum(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)


def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumprod, which means that the
  first
  element of the input is identical to the first element of the output:
  ```prettyprint
  tf.cumprod([a, b, c]) ==> [a, a * b, a * b * c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  performed
  instead:
  ```prettyprint
  tf.cumprod([a, b, c], exclusive=True) ==> [0, a, a * b]
  ```

  By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  opposite direction:
  ```prettyprint
  tf.cumprod([a, b, c], reverse=True) ==> [a * b * c, b * c, c]
  ```
  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:
  ```prettyprint
  tf.cumprod([a, b, c], exclusive=True, reverse=True) ==> [b * c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor` of type `int32` (default: 0).
    reverse: A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Cumprod", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.cumprod(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)


def conj(x, name=None):
  r"""Returns the complex conjugate of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `input`. The
  complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
  real part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

      # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
      tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]

  If `x` is real, it is returned unchanged.

  Args:
    x: `Tensor` to conjugate.  Must have numeric type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that is the conjugate of `x` (with the same type).

  Raises:
    TypeError: If `x` is not a numeric tensor.
  """
  with ops.name_scope(name, "Conj", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.is_complex:
      return gen_math_ops._conj(x, name=name)
    elif x.dtype.is_floating or x.dtype.is_integer:
      return x
    else:
      raise TypeError("Expected numeric tensor, got dtype %r" % x.dtype)


ops.RegisterShape("Abs")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Acos")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Asin")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Atan")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Ceil")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Conj")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Cos")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Cross")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Exp")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Floor")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Imag")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Inv")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("IsFinite")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("IsInf")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("IsNan")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Log")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("LogicalNot")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Neg")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Real")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Rsqrt")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Sign")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Sin")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Sqrt")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Square")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Sigmoid")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Tanh")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Tan")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Lgamma")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Digamma")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Erf")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Erfc")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Cast")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ComplexAbs")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("FFT")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("IFFT")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("FFT2D")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("IFFT2D")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("FFT3D")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("IFFT3D")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("TanhGrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SigmoidGrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("InvGrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SqrtGrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("RsqrtGrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Cumsum")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Cumprod")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Add")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Complex")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Div")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Equal")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Greater")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("GreaterEqual")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Igamma")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Igammac")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Zeta")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Polygamma")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Less")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("LessEqual")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("LogicalAnd")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("LogicalOr")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Maximum")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Minimum")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Mod")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Mul")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("NotEqual")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Pow")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Sub")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SquaredDifference")(common_shapes.call_cpp_shape_fn)


def _BroadcastShape(op):
  """Common shape function for binary operators that broadcast their inputs."""
  return [common_shapes.broadcast_shape(
      op.inputs[0].get_shape(),
      op.inputs[1].get_shape())]


ops.RegisterShape("Betainc")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseDenseCwiseMul")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseDenseCwiseDiv")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseDenseCwiseAdd")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("AddN")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Select")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("ArgMax")
@ops.RegisterShape("ArgMin")
def _ArgOpShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


@ops.RegisterShape("All")
@ops.RegisterShape("Any")
@ops.RegisterShape("Max")
@ops.RegisterShape("Mean")
@ops.RegisterShape("Min")
@ops.RegisterShape("Prod")
@ops.RegisterShape("Sum")
def _ReductionShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


ops.RegisterShape("SegmentMax")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SegmentMean")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SegmentMin")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SegmentProd")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SegmentSum")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseSegmentMean")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseSegmentSqrtN")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseSegmentSum")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("SparseSegmentMeanGrad")
@ops.RegisterShape("SparseSegmentSqrtNGrad")
# pylint: disable=invalid-name
def _SparseSegmentReductionGradShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[3])
# pylint: enable=invalid-name


@ops.RegisterShape("UnsortedSegmentSum")
def _UnsortedSegmentSumShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[2])


@ops.RegisterShape("LinSpace")
def _LinspaceShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[2])


def reduced_shape(input_shape, axes):
  """Helper function for reduction ops.

  Args:
    input_shape: 1-D Tensor, the shape of the Tensor being reduced.
    axes: 1-D Tensor, the reduction axes.
  Returns:
    A 1-D Tensor, the output shape as if keep_dims were set to True.
  """
  # Example:
  # cast needed for SparseTensor reductions
  input_shape = to_int32(input_shape)       # [2, 3, 5, 7]
  axes = to_int32(axes)                     # [1, 2]

  input_rank = array_ops.size(input_shape)  # 4
  axes = (axes + input_rank) % input_rank
  axes_shape = array_ops.shape(axes)        # [2]
  return gen_data_flow_ops.dynamic_stitch(  # [2, 1, 1, 7]
      [range(input_rank),                   # [0, 1, 2, 3]
       axes],                               # [1, 2]
      [input_shape,                         # [2, 3, 5, 7]
       array_ops.fill(axes_shape, 1)])      # [1, 1]
