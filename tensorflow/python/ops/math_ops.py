# Copyright 2015 Google Inc. All Rights Reserved.
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

"""## Arithmetic Operators

TensorFlow provides several operations that you can use to add basic arithmetic
operators to your graph.

@@add
@@sub
@@mul
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
@@lgamma
@@erf
@@erfc

## Matrix Math Functions

TensorFlow provides several operations that you can use to add basic
mathematical functions for matrices to your graph.

@@diag
@@transpose

@@matmul
@@batch_matmul

@@matrix_determinant
@@batch_matrix_determinant

@@matrix_inverse
@@batch_matrix_inverse

@@cholesky
@@batch_cholesky

@@self_adjoint_eig
@@batch_self_adjoint_eig

@@matrix_solve
@@batch_matrix_solve

@@matrix_triangular_solve
@@batch_matrix_triangular_solve

@@matrix_solve_ls
@@batch_matrix_solve_ls

## Complex Number Functions

TensorFlow provides several operations that you can use to add complex number
functions to your graph.

@@complex
@@complex_abs
@@conj
@@imag
@@real
@@fft2d
@@ifft2d

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

@@accumulate_n

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
import six.moves

from tensorflow.python.client import graph_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import gen_state_ops
# pylint: disable=wildcard-import,undefined-variable
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
    x: A `Tensor` of type `float`, `double`, `int32`, or `int64`.
    name: A name for the operation (optional).

  Returns:
     A `Tensor` the same size and type as `x` with absolute values.
  """
  with ops.op_scope([x], name, "Abs") as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype == dtypes.complex64:
      return gen_math_ops.complex_abs(x, name=name)
    return gen_math_ops._abs(x, name=name)


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
  scalar = ops.convert_to_tensor(scalar, dtype=x.dtype, name="scalar")
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
  # tensor 'x' is [[2, 2]], [3, 3]]
  # tensor 'y' is [[8, 16], [2, 3]]
  tf.pow(x, y) ==> [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor` of type `float`, `double`, `int32`, `complex64`, or `int64`.
    y: A `Tensor` of type `float`, `double`, `int32`, `complex64`, or `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.
  """
  with ops.op_scope([x], name, "Pow") as name:
    return gen_math_ops._pow(x, y, name=name)


def complex(real, imag, name=None):
  """Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation computes complex numbers elementwise of the form \\\\(a + bj\\\\),
  where *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must be the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.74j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor` of type `float`.
    imag: A `Tensor` of type `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  with ops.op_scope([real, imag], name, "Complex") as name:
    return gen_math_ops._complex(real, imag, name=name)


def round(x, name=None):
  """Rounds the values of a tensor to the nearest integer, element-wise.

  For example:

  ```python
  # 'a' is [0.9, 2.5, 2.3, -4.4]
  tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ]
  ```

  Args:
    x: A `Tensor` of type `float` or `double`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.
  """
  x = ops.convert_to_tensor(x, name="x")
  if x.dtype.is_integer:
    return x
  else:
    return floor(x + 0.5, name=name)


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
  with ops.op_scope([x], name, "Cast") as name:
    if isinstance(x, ops.SparseTensor):
      values_cast = cast(x.values, dtype, name=name)
      return ops.SparseTensor(x.indices, values_cast, x.shape)
    else:
      # TODO(touts): Handle what Josh said.
      #
      # Could return ops.convert_to_tensor(x, dtype=dtype, ...)  here, but that
      # allows some conversions that cast() can't do, e.g.  casting numbers to
      # strings.
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype.base_dtype == dtype:
        return x
      return gen_math_ops.cast(x, dtype, name=name)


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


ops.Tensor._override_operator("__neg__", neg)
ops.Tensor._override_operator("__abs__", abs)
# __invert__ corresponds to the ~ operator.  Here we follow the numpy convention
# ~ marks an elementwise bit-wise inverse.  This is only implemented for boolean
# tensors and will throw a TypeError if used on nonboolean arrays
ops.Tensor._override_operator("__invert__", logical_not)


def _OverrideBinaryOperatorHelper(func, op_name):
  """Register operators with different tensor and scalar versions.

  Args:
    func: the operator
    op_name: name of the operator being overridden
  """

  def binary_op_wrapper(x, y):
    with ops.op_scope([x, y], None, op_name) as name:
      assert isinstance(x, ops.Tensor)
      y = ops.convert_to_tensor(y, dtype=x.dtype.base_dtype, name="y")
      return func(x, y, name=name)

  ops.Tensor._override_operator("__%s__" % op_name, binary_op_wrapper)
  del binary_op_wrapper

  def r_binary_op_wrapper(y, x):
    with ops.op_scope([x, y], None, op_name) as name:
      assert isinstance(y, ops.Tensor)
      x = ops.convert_to_tensor(x, dtype=y.dtype.base_dtype, name="x")
      return func(x, y, name=name)

  ops.Tensor._override_operator("__r%s__" % op_name, r_binary_op_wrapper)
  del r_binary_op_wrapper


# Conversion table for __truediv__.  None entries mean no conversion required.
_TRUEDIV_TABLE = {
    dtypes.uint8: dtypes.float32,
    dtypes.int8: dtypes.float32,
    dtypes.int16: dtypes.float32,
    dtypes.int32: dtypes.float64,
    dtypes.int64: dtypes.float64,
    dtypes.float32: None,
    dtypes.float64: None,
    dtypes.complex64: None,
}


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
  with ops.op_scope([x, y], name, "truediv") as name:
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
    return div(x, y, name=name)


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
  with ops.op_scope([x, y], name, "floordiv") as name:
    x = ops.convert_to_tensor(x, name="x")
    dtype = x.dtype
    if dtype.is_floating:
      return floor(div(x, y), name=name)
    else:
      if not dtype.is_integer:
        raise TypeError("Expected floating point or integer, got %r" % dtype)
      return div(x, y, name=name)


_OverrideBinaryOperatorHelper(add, "add")
_OverrideBinaryOperatorHelper(sub, "sub")
_OverrideBinaryOperatorHelper(mul, "mul")
_OverrideBinaryOperatorHelper(div, "div")
_OverrideBinaryOperatorHelper(truediv, "truediv")
_OverrideBinaryOperatorHelper(floordiv, "floordiv")
_OverrideBinaryOperatorHelper(mod, "mod")
_OverrideBinaryOperatorHelper(pow, "pow")


def logical_xor(x, y, name="LogicalXor"):
  """x ^ y = (x | y) & ~(x & y)."""
  # TODO(alemi) Make this a cwise op if people end up relying on it.
  return logical_and(logical_or(x, y), logical_not(logical_and(x, y)),
                     name=name)

_OverrideBinaryOperatorHelper(logical_and, "and")
_OverrideBinaryOperatorHelper(logical_or, "or")
_OverrideBinaryOperatorHelper(logical_xor, "xor")

ops.Tensor._override_operator("__lt__", less)
ops.Tensor._override_operator("__le__", less_equal)
ops.Tensor._override_operator("__gt__", greater)
ops.Tensor._override_operator("__ge__", greater_equal)


def range(start, limit=None, delta=1, name="range"):
  """Creates a sequence of integers.

  Creates a sequence of integers that begins at `start` and extends by
  increments of `delta` up to but not including `limit`.

  Like the Python builtin `range`, `start` defaults to 0, so that
  `range(n) = range(0, n)`.

  For example:

  ```
  # 'start' is 3
  # 'limit' is 18
  # 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

  # 'limit' is 5
  tf.range(limit) ==> [0, 1, 2, 3, 4]
  ```

  Args:
    start: A 0-D (scalar) of type `int32`. First entry in sequence.
      Defaults to 0.
    limit: A 0-D (scalar) of type `int32`. Upper limit of sequence,
      exclusive.
    delta: A 0-D `Tensor` (scalar) of type `int32`. Optional. Default is 1.
      Number that increments `start`.
    name: A name for the operation (optional).

  Returns:
    An 1-D `int32` `Tensor`.
  """
  if limit is None:
    start, limit = 0, start
  return gen_math_ops._range(start, limit, delta, name=name)


@ops.RegisterShape("Range")
def _RangeShape(op):
  start_value = tensor_util.constant_value(op.inputs[0])
  limit_value = tensor_util.constant_value(op.inputs[1])
  delta_value = tensor_util.constant_value(op.inputs[2])
  if start_value is None or limit_value is None or delta_value is None:
    return [tensor_shape.vector(None)]
  else:
    return [tensor_shape.vector((limit_value - start_value + delta_value - 1) //
                                delta_value)]


# Reduction operations
def _ReductionDims(x, reduction_indices):
  """Returns range(0, rank(x)) if reduction_indices is None."""
  if reduction_indices is not None:
    return reduction_indices
  else:
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


def matmul(a, b,
           transpose_a=False, transpose_b=False,
           a_is_sparse=False, b_is_sparse=False,
           name=None):
  """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

  The inputs must be two-dimensional matrices, with matching inner dimensions,
  possibly after transposition.

  Both matrices must be of the same type. The supported types are:
  `float`, `double`, `int32`, `complex64`.

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
    a: `Tensor` of type `float`, `double`, `int32` or `complex64`.
    b: `Tensor` with same type as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a`.
  """
  with ops.op_scope([a, b], name, "MatMul") as name:
    a = ops.convert_to_tensor(a, name="a")
    b = ops.convert_to_tensor(b, name="b")
    if a.dtype == dtypes.float32 and (a_is_sparse or b_is_sparse):
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

ops.RegisterShape("MatMul")(common_shapes.matmul_shape)
ops.RegisterShape("SparseMatMul")(common_shapes.matmul_shape)


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


def _as_indexed_slices(x):
  """Convert 'x' to IndexedSlices.

  Convert a dense Tensor to a block-sparse IndexedSlices.

  Args:
    x: Either a Tensor object, or an IndexedSlices object.

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
  x_shape = array_ops.shape(x)
  return ops.IndexedSlices(x, range(0, x_shape[0]), x_shape)


def _as_indexed_slices_list(inputs):
  """Convert all elements of 'inputs' to IndexedSlices.

  Additionally, homogenize the types of all the indices to
  either int32 or int64.

  Args:
    inputs: List containing either Tensor or IndexedSlices objects.

  Returns:
    A list of IndexedSlices objects.

  Raises:
    TypeError: If 'inputs' is not a list or a tuple.
  """
  if not isinstance(inputs, (list, tuple)):
    raise TypeError("Expected a list or tuple, not a %s" % type(inputs))
  outputs = [_as_indexed_slices(i) for i in inputs]
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


def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
  """Returns the element-wise sum of a list of tensors.

  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
  otherwise, these are inferred.

  For example:

  ```python
  # tensor 'a' is [[1, 2], [3, 4]
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
  if tensor_dtype is None:
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
    tensor_dtype = inputs[0].dtype
  if shape is not None:
    shape = tensor_shape.as_shape(shape)
  else:
    shape = tensor_shape.unknown_shape()
    for input_tensor in inputs:
      if isinstance(input_tensor, ops.Tensor):
        shape = shape.merge_with(input_tensor.get_shape())
  if not shape.is_fully_defined():
    # TODO(pbar): Make a version of assign_add that accepts an uninitialized
    # lvalue, and takes its shape from that? This would allow accumulate_n to
    # work in all situations that add_n currently works.
    raise ValueError("Cannot infer the shape of the accumulator for "
                     "accumulate_n. Pass the shape argument, or set the shape "
                     "of at least one of the inputs.")
  with ops.op_scope(inputs, name, "AccumulateN") as name:
    var = gen_state_ops._temporary_variable(shape=shape, dtype=tensor_dtype)
    var_name = var.op.name
    var = state_ops.assign(var, array_ops.zeros_like(inputs[0]))
    update_ops = []
    for input_tensor in inputs:
      op = state_ops.assign_add(var, input_tensor, use_locking=True)
      update_ops.append(op)
    with ops.control_dependencies(update_ops):
      return gen_state_ops._destroy_temporary_variable(var,
                                                       var_name=var_name,
                                                       name=name)


@ops.RegisterShape("BatchMatMul")
def _BatchMatMulShape(op):
  """Shape function for BatchMatMul op."""
  a_shape = op.inputs[0].get_shape()
  adj_a = op.get_attr("adj_x")
  b_shape = op.inputs[1].get_shape()
  adj_b = op.get_attr("adj_y")
  if a_shape.dims is None and b_shape.dims is None:
    return [tensor_shape.unknown_shape()]
  batch_dims = a_shape[:-2].merge_with(b_shape[:-2])
  output_rows = a_shape[-1] if adj_a else a_shape[-2]
  output_cols = b_shape[-2] if adj_b else b_shape[-1]
  inner_a = a_shape[-2] if adj_a else a_shape[-1]
  inner_b = b_shape[-1] if adj_b else b_shape[-2]
  inner_a.assert_is_compatible_with(inner_b)
  return [batch_dims.concatenate([output_rows, output_cols])]


def sigmoid(x, name=None):
  """Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32`
      otherwise the return type is `quint8`.
  """
  with ops.op_scope([x], name, "Sigmoid") as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops._sigmoid(x, name=name)


def tanh(x, name=None):
  """Computes hyperbolic tangent of `x` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
  with ops.op_scope([x], name, "Tanh") as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops._tanh(x, name=name)


def lgamma(x, name=None):
  """Computes `ln(|gamma(x)|)` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
  with ops.op_scope([x], name, "Lgamma") as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops._lgamma(x, name=name)


def erf(x, name=None):
  """Computes Gauss error function of `x` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
  with ops.op_scope([x], name, "Erf") as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops._erf(x, name=name)


def erfc(x, name=None):
  """Computes complementary error function of `x` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
  with ops.op_scope([x], name, "Erfc") as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops._erfc(x, name=name)


ops.RegisterShape("Abs")(common_shapes.unchanged_shape)
ops.RegisterShape("Ceil")(common_shapes.unchanged_shape)
ops.RegisterShape("Conj")(common_shapes.unchanged_shape)
ops.RegisterShape("Cos")(common_shapes.unchanged_shape)
ops.RegisterShape("Cross")(common_shapes.unchanged_shape)
ops.RegisterShape("Exp")(common_shapes.unchanged_shape)
ops.RegisterShape("Floor")(common_shapes.unchanged_shape)
ops.RegisterShape("Imag")(common_shapes.unchanged_shape)
ops.RegisterShape("Inv")(common_shapes.unchanged_shape)
ops.RegisterShape("IsFinite")(common_shapes.unchanged_shape)
ops.RegisterShape("IsInf")(common_shapes.unchanged_shape)
ops.RegisterShape("IsNan")(common_shapes.unchanged_shape)
ops.RegisterShape("Log")(common_shapes.unchanged_shape)
ops.RegisterShape("LogicalNot")(common_shapes.unchanged_shape)
ops.RegisterShape("Neg")(common_shapes.unchanged_shape)
ops.RegisterShape("Real")(common_shapes.unchanged_shape)
ops.RegisterShape("Rsqrt")(common_shapes.unchanged_shape)
ops.RegisterShape("Sign")(common_shapes.unchanged_shape)
ops.RegisterShape("Sin")(common_shapes.unchanged_shape)
ops.RegisterShape("Sqrt")(common_shapes.unchanged_shape)
ops.RegisterShape("Square")(common_shapes.unchanged_shape)
ops.RegisterShape("Sigmoid")(common_shapes.unchanged_shape)
ops.RegisterShape("Tanh")(common_shapes.unchanged_shape)
ops.RegisterShape("Lgamma")(common_shapes.unchanged_shape)
ops.RegisterShape("Erf")(common_shapes.unchanged_shape)
ops.RegisterShape("Erfc")(common_shapes.unchanged_shape)
ops.RegisterShape("Cast")(common_shapes.unchanged_shape)
ops.RegisterShape("ComplexAbs")(common_shapes.unchanged_shape)
ops.RegisterShape("FFT2D")(common_shapes.unchanged_shape)
ops.RegisterShape("IFFT2D")(common_shapes.unchanged_shape)


@ops.RegisterShape("Add")
@ops.RegisterShape("Complex")
@ops.RegisterShape("Div")
@ops.RegisterShape("Equal")
@ops.RegisterShape("Greater")
@ops.RegisterShape("GreaterEqual")
@ops.RegisterShape("Less")
@ops.RegisterShape("LessEqual")
@ops.RegisterShape("LogicalAnd")
@ops.RegisterShape("LogicalOr")
@ops.RegisterShape("Maximum")
@ops.RegisterShape("Minimum")
@ops.RegisterShape("Mod")
@ops.RegisterShape("Mul")
@ops.RegisterShape("NotEqual")
@ops.RegisterShape("Pow")
@ops.RegisterShape("Sub")
def _BroadcastShape(op):
  """Common shape function for binary operators that broadcast their inputs."""
  shape_x = op.inputs[0].get_shape()
  shape_y = op.inputs[1].get_shape()
  if shape_x.ndims is None or shape_y.ndims is None:
    return [tensor_shape.unknown_shape()]

  # To compute the broadcasted dimensions, we zip together shape_x and shape_y,
  # and pad with 1 to make them the same length.
  broadcasted_dims = reversed(list(six.moves.zip_longest(
      reversed(shape_x.dims),
      reversed(shape_y.dims),
      fillvalue=tensor_shape.Dimension(1))))
  # Next we combine the dimensions according to the numpy broadcasting rules.
  # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  return_dims = []
  for (dim_x, dim_y) in broadcasted_dims:
    if dim_x.value is None or dim_y.value is None:
      # One or both dimensions is unknown. If either dimension is greater than
      # 1, we assume that the program is correct, and the other dimension will
      # be broadcast to match it.
      # TODO(mrry): If we eliminate the shape checks in C++, we must still
      # assert that the unknown dim is either 1 or the same as the known dim.
      if dim_x.value is not None and dim_x.value > 1:
        return_dims.append(dim_x)
      elif dim_y.value is not None and dim_y.value > 1:
        return_dims.append(dim_y)
      else:
        return_dims.append(None)
    elif dim_x.value == 1:
      # We will broadcast dim_x to dim_y.
      return_dims.append(dim_y)
    elif dim_y.value == 1:
      # We will broadcast dim_y to dim_x.
      return_dims.append(dim_x)
    elif dim_x.value == dim_y.value:
      # The dimensions are compatible, so output is the same size in that
      # dimension.
      return_dims.append(dim_x.merge_with(dim_y))
    else:
      raise ValueError("Incompatible shapes for broadcasting: %s and %s"
                       % (shape_x, shape_y))
  return [tensor_shape.TensorShape(return_dims)]


@ops.RegisterShape("AddN")
def _AddNShape(op):
  merged_shape = tensor_shape.unknown_shape()
  for input_ in op.inputs:
    merged_shape = merged_shape.merge_with(input_.get_shape())
  return [merged_shape]


@ops.RegisterShape("Select")
def _SelectShape(op):
  """Shape function for SelectOp."""
  # The inputs 'then' and 'else' must have the same shape.
  # The input 'cond' must either have the same shape as 'then' and
  # 'else', or be a vector if 'then' and 'else' are at least vectors.
  c_shape = op.inputs[0].get_shape()
  t_shape = op.inputs[1].get_shape()
  e_shape = op.inputs[2].get_shape()
  t_e_shape = t_shape.merge_with(e_shape)
  c_shape_list = c_shape.as_list() if c_shape.ndims is not None else None
  t_e_shape_list = t_e_shape.as_list() if t_e_shape.ndims is not None else None
  if c_shape_list is not None and t_e_shape_list is not None:
    if len(c_shape_list) != 1:
      # If the rank of 'cond' is != 1, the shape must match 'then' and 'else'
      t_e_shape = t_e_shape.merge_with(c_shape)
    if t_e_shape_list:
      # If then and else are not scalars, then cond must be at least
      # a vector, and its first value must match that of 'else'
      c_shape = c_shape.with_rank_at_least(1)
      if len(c_shape.as_list()) == 1:
        c_shape.merge_with(tensor_shape.vector(t_e_shape_list[0]))
  return [t_e_shape]


@ops.RegisterShape("ArgMax")
@ops.RegisterShape("ArgMin")
def _ArgOpShape(op):
  """Common shape function for arg-reduction ops."""
  dimension_shape = op.inputs[1].get_shape()
  dimension_shape.assert_is_compatible_with(tensor_shape.scalar())
  input_shape = op.inputs[0].get_shape()
  if input_shape.ndims is None:
    return [tensor_shape.unknown_shape()]
  elif input_shape.ndims <= 1:
    return [tensor_shape.scalar()]

  dimension = tensor_util.constant_value(op.inputs[1])
  if dimension is None:
    return [tensor_shape.unknown_shape(ndims=input_shape.ndims - 1)]
  elif 0 <= dimension and dimension < input_shape.ndims:
    returned_shape = []
    for i, dim in enumerate(input_shape.dims):
      if i != dimension:
        returned_shape.append(dim)
    return [tensor_shape.TensorShape(returned_shape)]
  else:
    raise ValueError(
        "dimension (%d) must be in the range [0, %d), where %d is the number "
        "of dimensions in the input"
        % (dimension, input_shape.ndims, input_shape.ndims))


@ops.RegisterShape("All")
@ops.RegisterShape("Any")
@ops.RegisterShape("Max")
@ops.RegisterShape("Mean")
@ops.RegisterShape("Min")
@ops.RegisterShape("Prod")
@ops.RegisterShape("Sum")
def _ReductionShape(op):
  """Common shape function for reduction ops."""
  input_shape = op.inputs[0].get_shape()
  reduction_indices = tensor_util.constant_value(op.inputs[1])
  keep_dims = op.get_attr("keep_dims")
  if reduction_indices is None or input_shape.ndims is None:
    if keep_dims:
      return [tensor_shape.unknown_shape(ndims=input_shape.ndims)]
    else:
      return [tensor_shape.unknown_shape()]

  # Turn reduction_indices from scalar to vector if necessary
  reduction_indices = np.ravel(reduction_indices)

  for reduction_index in reduction_indices:
    if reduction_index < 0 or reduction_index >= input_shape.ndims:
      raise ValueError("Invalid reduction dimension %d for input with %d "
                       "dimensions" % (reduction_index, input_shape.ndims))

  returned_dims = []
  if keep_dims:
    for i, dim in enumerate(input_shape.dims):
      if i in reduction_indices:
        returned_dims.append(1)
      else:
        returned_dims.append(dim)
  else:
    for i, dim in enumerate(input_shape.dims):
      if i not in reduction_indices:
        returned_dims.append(dim)
  return [tensor_shape.TensorShape(returned_dims)]


@ops.RegisterShape("SegmentMax")
@ops.RegisterShape("SegmentMean")
@ops.RegisterShape("SegmentMin")
@ops.RegisterShape("SegmentProd")
@ops.RegisterShape("SegmentSum")
def _SegmentReductionShape(op):
  """Common shape function for segment reduction ops."""
  data_shape = op.inputs[0].get_shape()
  segment_ids_shape = op.inputs[1].get_shape()
  segment_ids_shape.assert_has_rank(1)
  return [tensor_shape.TensorShape([None]).concatenate(data_shape[1:])]


@ops.RegisterShape("SparseSegmentMean")
@ops.RegisterShape("SparseSegmentSqrtN")
@ops.RegisterShape("SparseSegmentSum")
def _SparseSegmentReductionShape(op):
  """Common shape function for sparse segment reduction ops."""
  data_shape = op.inputs[0].get_shape()
  indices_shape = op.inputs[1].get_shape()
  indices_shape.assert_has_rank(1)
  segment_ids_shape = op.inputs[2].get_shape()
  segment_ids_shape.assert_has_rank(1)
  indices_shape.assert_is_compatible_with(segment_ids_shape)
  return [tensor_shape.TensorShape([None]).concatenate(data_shape[1:])]


@ops.RegisterShape("SparseSegmentMeanGrad")
@ops.RegisterShape("SparseSegmentSqrtNGrad")


# pylint: disable=invalid-name
def _SparseSegmentReductionGradShape(op):
  """Shape function for the SparseSegment[Mean|SqrtN]Grad ops."""
  input_shape = op.inputs[0].get_shape()
  indices_shape = op.inputs[1].get_shape().with_rank(1)
  unused_segment_ids_shape = op.inputs[2].get_shape().merge_with(indices_shape)
  unused_output_dim0_shape = op.inputs[3].get_shape().merge_with(
      tensor_shape.scalar())
  output_dim0 = tensor_util.constant_value(op.inputs[3])
  if output_dim0 is not None:
    dim0 = output_dim0[0]
  else:
    dim0 = None
  return [tensor_shape.TensorShape([dim0]).concatenate(input_shape[1:])]
# pylint: enable=invalid-name


@ops.RegisterShape("UnsortedSegmentSum")
def _UnsortedSegmentSumShape(op):
  """Shape function for UnsortedSegmentSum."""
  data_shape = op.inputs[0].get_shape()
  segment_ids_shape = op.inputs[1].get_shape()
  mid = segment_ids_shape.ndims
  if mid is None:
    return [tensor_shape.unknown_shape()]
  else:
    num_segments = tensor_util.constant_value(op.inputs[2])
    return [tensor_shape.TensorShape([num_segments]).concatenate(
        data_shape[mid:])]


@ops.RegisterShape("LinSpace")
def _LinspaceShape(op):
  num = tensor_util.constant_value(op.inputs[2])
  return [tensor_shape.vector(num)]
