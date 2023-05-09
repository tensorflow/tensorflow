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
"""Math Operations.

Note: Functions taking `Tensor` arguments can also take anything accepted by
`tf.convert_to_tensor`.

Note: Elementwise binary operations in TensorFlow follow [numpy-style
broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

TensorFlow provides a variety of math functions including:

* Basic arithmetic operators and trigonometric functions.
* Special math functions (like: `tf.math.igamma` and `tf.math.zeta`)
* Complex number functions (like: `tf.math.imag` and `tf.math.angle`)
* Reductions and scans (like: `tf.math.reduce_mean` and `tf.math.cumsum`)
* Segment functions (like: `tf.math.segment_sum`)

See: `tf.linalg` for matrix and tensor functions.

<a id=Segmentation></a>

## About Segmentation

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
tf.math.segment_sum(c, tf.constant([0, 0, 1]))
#  ==>  [[0 0 0 0]
#        [5 6 7 8]]
```

The standard `segment_*` functions assert that the segment indices are sorted.
If you have unsorted indices use the equivalent `unsorted_segment_` function.
These functions take an additional argument `num_segments` so that the output
tensor can be efficiently allocated.

``` python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
tf.math.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 6,  8, 10, 12],
#       [-1, -2, -3, -4]]
```

"""
import builtins
import numbers
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
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
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export


np_dtypes = LazyLoader(
    "np_dtypes", globals(),
    "tensorflow.python.ops.numpy_ops.np_dtypes")


# Aliases for some automatically-generated names.
nextafter = gen_math_ops.next_after


@tf_export("linspace", v1=["lin_space", "linspace"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("lin_space")
def linspace_nd(start, stop, num, name=None, axis=0):
  r"""Generates evenly-spaced values in an interval along a given axis.

  A sequence of `num` evenly-spaced values are generated beginning at `start`
  along a given `axis`.
  If `num > 1`, the values in the sequence increase by
  `(stop - start) / (num - 1)`, so that the last one is exactly `stop`.
  If `num <= 0`, `ValueError` is raised.

  Matches
  [np.linspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)'s
  behaviour
  except when `num == 0`.

  For example:

  ```
  tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  ```

  `Start` and `stop` can be tensors of arbitrary size:

  >>> tf.linspace([0., 5.], [10., 40.], 5, axis=0)
  <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
  array([[ 0.  ,  5.  ],
         [ 2.5 , 13.75],
         [ 5.  , 22.5 ],
         [ 7.5 , 31.25],
         [10.  , 40.  ]], dtype=float32)>

  `Axis` is where the values will be generated (the dimension in the
  returned tensor which corresponds to the axis will be equal to `num`)

  >>> tf.linspace([0., 5.], [10., 40.], 5, axis=-1)
  <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
  array([[ 0.  ,  2.5 ,  5.  ,  7.5 , 10.  ],
         [ 5.  , 13.75, 22.5 , 31.25, 40.  ]], dtype=float32)>



  Args:
    start: A `Tensor`. Must be one of the following types: `bfloat16`,
      `float32`, `float64`. N-D tensor. First entry in the range.
    stop: A `Tensor`. Must have the same type and shape as `start`. N-D tensor.
      Last entry in the range.
    num: A `Tensor`. Must be one of the following types: `int32`, `int64`. 0-D
      tensor. Number of values to generate.
    name: A name for the operation (optional).
    axis: Axis along which the operation is performed (used only when N-D
      tensors are provided).

  Returns:
    A `Tensor`. Has the same type as `start`.
  """

  with ops.name_scope(name, "linspace", [start, stop]):
    start = ops.convert_to_tensor(start, name="start")
    # stop must be convertible to the same dtype as start
    stop = ops.convert_to_tensor(stop, name="stop", dtype=start.dtype)
    num_int = array_ops.convert_to_int_tensor(num, name="num")
    num = cast(num_int, dtype=start.dtype)

    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(start), array_ops.shape(stop))
    start = array_ops.broadcast_to(start, broadcast_shape)
    stop = array_ops.broadcast_to(stop, broadcast_shape)

    expanded_start = array_ops.expand_dims(start, axis=axis)
    expanded_stop = array_ops.expand_dims(stop, axis=axis)

    shape = array_ops.shape(expanded_start)
    ndims = array_ops.shape(shape)[0]

    axis = array_ops.where_v2(axis >= 0, axis, ndims + axis)

    # The purpose is to avoid having negative values when repeating.
    num_fill = gen_math_ops.maximum(num_int - 2, 0)
    # To avoid having negative values in the range or zero division
    # the result is sliced in the end so a correct result is returned for
    # num == 1, and num == 0.
    n_steps = gen_math_ops.maximum(num_int - 1, 1)
    delta = (expanded_stop - expanded_start) / cast(n_steps,
                                                    expanded_stop.dtype)
    # Re-cast tensors as delta.
    expanded_start = cast(expanded_start, delta.dtype)
    expanded_stop = cast(expanded_stop, delta.dtype)
    # If num < 0, we will throw exception in the range
    # otherwise use the same div for delta
    range_end = array_ops.where_v2(num_int >= 0, n_steps, -1)
    # Even though range supports an output dtype, its limited
    # (e.g. doesn't support half at the moment).
    desired_range = cast(range(1, range_end, dtype=dtypes.int64), delta.dtype)
    mask = gen_math_ops.equal(axis, range(ndims))
    # desired_range_shape is [1. 1. 1. ... 1. num_fill 1. 1. ... 1.], where the
    # index of num_fill is equal to axis.
    desired_range_shape = array_ops.where_v2(mask, num_fill, 1)
    desired_range = array_ops.reshape(desired_range, desired_range_shape)

    res = expanded_start + delta * desired_range

    # Add the start and endpoints to the result, and slice out the desired
    # portion.
    all_tensors = (expanded_start, res, expanded_stop)
    concatenated = array_ops.concat(all_tensors, axis=axis)
    begin = array_ops.zeros_like(shape)
    size = array_ops.where_v2(mask, num_int, shape)

    return array_ops.slice(concatenated, begin, size)


linspace = linspace_nd

arg_max = deprecation.deprecated(None, "Use `tf.math.argmax` instead")(arg_max)  # pylint: disable=used-before-assignment
arg_min = deprecation.deprecated(None, "Use `tf.math.argmin` instead")(arg_min)  # pylint: disable=used-before-assignment
tf_export(v1=["arg_max"])(dispatch.add_dispatch_support(arg_max))
tf_export(v1=["arg_min"])(dispatch.add_dispatch_support(arg_min))


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
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, "Use the `axis` argument instead",
                             "dimension")
@_set_doc(
    gen_math_ops.arg_max.__doc__.replace("dimensions",
                                         "axes").replace("dimension", "axis"))
def argmax(input,
           axis=None,
           name=None,
           dimension=None,
           output_type=dtypes.int64):
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dimension",
                                                dimension)
  return argmax_v2(input, axis, output_type, name)


@tf_export("math.argmax", "argmax", v1=[])
@dispatch.add_dispatch_support
def argmax_v2(input, axis=None, output_type=dtypes.int64, name=None):
  """Returns the index with the largest value across axes of a tensor.

  In case of identity returns the smallest index.

  For example:

  >>> A = tf.constant([2, 20, 30, 3, 6])
  >>> tf.math.argmax(A)  # A[2] is maximum in tensor A
  <tf.Tensor: shape=(), dtype=int64, numpy=2>
  >>> B = tf.constant([[2, 20, 30, 3, 6], [3, 11, 16, 1, 8],
  ...                  [14, 45, 23, 5, 27]])
  >>> tf.math.argmax(B, 0)
  <tf.Tensor: shape=(5,), dtype=int64, numpy=array([2, 2, 0, 2, 2])>
  >>> tf.math.argmax(B, 1)
  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 2, 1])>
  >>> C = tf.constant([0, 0, 0, 0])
  >>> tf.math.argmax(C) # Returns smallest index in case of ties
  <tf.Tensor: shape=(), dtype=int64, numpy=0>

  Args:
    input: A `Tensor`.
    axis: An integer, the axis to reduce across. Default to 0.
    output_type: An optional output dtype (`tf.int32` or `tf.int64`). Defaults
      to `tf.int64`.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of type `output_type`.
  """
  if axis is None:
    axis = 0
  return gen_math_ops.arg_max(input, axis, name=name, output_type=output_type)


@tf_export(v1=["math.argmin", "argmin"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, "Use the `axis` argument instead",
                             "dimension")
@_set_doc(
    gen_math_ops.arg_min.__doc__.replace("dimensions",
                                         "axes").replace("dimension", "axis"))
def argmin(input,
           axis=None,
           name=None,
           dimension=None,
           output_type=dtypes.int64):
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dimension",
                                                dimension)
  return argmin_v2(input, axis, output_type, name)


@tf_export("math.argmin", "argmin", v1=[])
@dispatch.add_dispatch_support
def argmin_v2(input, axis=None, output_type=dtypes.int64, name=None):
  """Returns the index with the smallest value across axes of a tensor.

  Returns the smallest index in case of ties.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,
      `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`,
      `uint64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `-rank(input), rank(input))`.
      Describes which axis of the input Tensor to reduce across. For vectors,
      use axis = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to
      `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.

  Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.math.argmin(input = a)
  c = tf.keras.backend.eval(b)
  # c = 0
  # here a[0] = 1 which is the smallest element of a across axis 0
  ```
  """
  if axis is None:
    axis = 0
  return gen_math_ops.arg_min(input, axis, name=name, output_type=output_type)


# pylint: enable=redefined-builtin


# pylint: disable=anomalous-backslash-in-string,protected-access
# pylint: disable=g-docstring-has-escape
@tf_export("math.abs", "abs")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def abs(x, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the absolute value of a tensor.

  Given a tensor of integer or floating-point values, this operation returns a
  tensor of the same type, where each element contains the absolute value of the
  corresponding element in the input.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float32` or `float64` that is the absolute value of each element in `x`. For
  a complex number \\(a + bj\\), its absolute value is computed as
  \\(\sqrt{a^2 + b^2}\\).

  For example:

  >>> # real number
  >>> x = tf.constant([-2.25, 3.25])
  >>> tf.abs(x)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([2.25, 3.25], dtype=float32)>

  >>> # complex number
  >>> x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
  >>> tf.abs(x)
  <tf.Tensor: shape=(2, 1), dtype=float64, numpy=
  array([[5.25594901],
         [6.60492241]])>

  Args:
    x: A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
      `int32`, `int64`, `complex64` or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` of the same size, type and sparsity as `x`,
      with absolute values. Note, for `complex64` or `complex128` input, the
      returned `Tensor` will be of type `float32` or `float64`, respectively.
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


class DivideDelegateWithName:
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
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def divide(x, y, name=None):
  """Computes Python style division of `x` by `y`.

  For example:

  >>> x = tf.constant([16, 12, 11])
  >>> y = tf.constant([4, 6, 2])
  >>> tf.divide(x,y)
  <tf.Tensor: shape=(3,), dtype=float64,
  numpy=array([4. , 2. , 5.5])>

  Args:
    x: A `Tensor`
    y: A `Tensor`
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with same shape as input
  """

  if name is not None:
    # Cannot use tensors operator overload, because it has no way to track
    # override names. Use a dummy class to track the runtime division behavior
    return DivideDelegateWithName(x, name) / y
  else:
    # We do conversion here to make sure at least x is a tensor.
    if not tensor_util.is_tf_type(x):
      dtype = y.dtype.base_dtype if tensor_util.is_tf_type(y) else None
      x = ops.convert_to_tensor(x, dtype=dtype)
    return x / y


@tf_export("math.multiply", "multiply")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def multiply(x, y, name=None):
  """Returns an element-wise x * y.

  For example:

  >>> x = tf.constant(([1, 2, 3, 4]))
  >>> tf.math.multiply(x, x)
  <tf.Tensor: shape=(4,), dtype=..., numpy=array([ 1,  4,  9, 16], dtype=int32)>

  Since `tf.math.multiply` will convert its arguments to `Tensor`s, you can also
  pass in non-`Tensor` arguments:

  >>> tf.math.multiply(7,6)
  <tf.Tensor: shape=(), dtype=int32, numpy=42>

  If `x.shape` is not the same as `y.shape`, they will be broadcast to a
  compatible shape. (More about broadcasting
  [here](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).)

  For example:

  >>> x = tf.ones([1, 2]);
  >>> y = tf.ones([2, 1]);
  >>> x * y  # Taking advantage of operator overriding
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[1., 1.],
       [1., 1.]], dtype=float32)>

  The reduction version of this elementwise operation is `tf.math.reduce_prod`

  Args:
    x: A Tensor. Must be one of the following types: `bfloat16`,
      `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`,
      `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:

  A `Tensor`.  Has the same type as `x`.

  Raises:

   * InvalidArgumentError: When `x` and `y` have incompatible shapes or types.
  """

  return gen_math_ops.mul(x, y, name)


# TODO(aselle): put deprecation in after another round of global code changes
@deprecation.deprecated(
    "2016-12-30",
    "`tf.mul(x, y)` is deprecated; use `tf.math.multiply(x, y)` or `x * y`")
def _mul(x, y, name=None):
  return gen_math_ops.mul(x, y, name)


_mul.__doc__ = (
    gen_math_ops.mul.__doc__ + ("" if _mul.__doc__ is None else _mul.__doc__))


@tf_export("math.subtract", "subtract")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def subtract(x, y, name=None):
  return gen_math_ops.sub(x, y, name)


subtract.__doc__ = gen_math_ops.sub.__doc__


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
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def scalar_mul(scalar, x, name=None):
  """Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

  This is a special case of `tf.math.multiply`, where the first value must be a
  `scalar`. Unlike the general form of `tf.math.multiply`, this is operation is
  guaranteed to be efficient for `tf.IndexedSlices`.

  >>> x = tf.reshape(tf.range(30, dtype=tf.float32), [10, 3])
  >>> with tf.GradientTape() as g:
  ...   g.watch(x)
  ...   y = tf.gather(x, [1, 2])  # IndexedSlices
  ...   z = tf.math.scalar_mul(10.0, y)

  Args:
    scalar: A 0-D scalar `Tensor`. Must have known shape.
    x: A `Tensor` or `IndexedSlices` to be scaled.
    name: A name for the operation (optional).

  Returns:
    `scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.

  Raises:
    ValueError: if scalar is not a 0-D `scalar`.
  """
  base_dtype = dtypes.as_dtype(x.dtype).base_dtype
  scalar = ops.convert_to_tensor(
      scalar, dtype=base_dtype, name="scalar")
  shape = scalar.get_shape()
  if shape.ndims == 0:
    if isinstance(x, indexed_slices.IndexedSlices):
      return indexed_slices.IndexedSlices(
          gen_math_ops.mul(scalar, x.values, name), x.indices, x.dense_shape)
    else:
      return gen_math_ops.mul(scalar, x, name)
  else:
    raise ValueError(
        f"The input scalar must be a 0-D value. Received shape {shape}.")


@tf_export("math.softplus", "nn.softplus", v1=["math.softplus", "nn.softplus"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def softplus(features, name=None):
  """Computes elementwise softplus: `softplus(x) = log(exp(x) + 1)`.

  `softplus` is a smooth approximation of `relu`. Like `relu`, `softplus` always
  takes on positive values.

  <img style="width:100%" src="https://www.tensorflow.org/images/softplus.png">

  Example:

  >>> import tensorflow as tf
  >>> tf.math.softplus(tf.range(0, 2, dtype=tf.float32)).numpy()
  array([0.6931472, 1.3132616], dtype=float32)

  Args:
    features: `Tensor`
    name: Optional: name to associate with this operation.
  Returns:
    `Tensor`
  """
  return gen_nn_ops.softplus(features, name)


@tf_export("math.scalar_mul", "scalar_mul", v1=[])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@_set_doc(scalar_mul.__doc__)
def scalar_mul_v2(scalar, x, name=None):
  with ops.name_scope(name, "scalar_mul", [x]) as name:
    return scalar_mul(scalar, x, name)


@tf_export("math.pow", "pow")
@dispatch.register_binary_elementwise_api
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
@dispatch.register_binary_elementwise_api
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
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64` or `complex128`.

  Raises:
    TypeError: Real and imag must be correct types
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
      raise TypeError(
          f"The `real` and `imag` components have incorrect types: "
          f"{real.dtype.name} {imag.dtype.name}. They must be consistent, and "
          f"one of {[dtypes.float32, dtypes.float64]}")
    return gen_math_ops._complex(real, imag, Tout=Tout, name=name)


@tf_export("math.sign", "sign")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def sign(x, name=None):
  r"""Returns an element-wise indication of the sign of a number.

  `y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0`.

  For complex numbers, `y = sign(x) = x / |x| if x != 0, otherwise y = 0`.

  Example usage:

  >>> # real number
  >>> tf.math.sign([0., 2., -3.])
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([ 0.,  1., -1.], dtype=float32)>

  >>> # complex number
  >>> tf.math.sign([1 + 1j, 0 + 0j])
  <tf.Tensor: shape=(2,), dtype=complex128,
  numpy=array([0.70710678+0.70710678j, 0.        +0.j        ])>

  Args:
   x: A Tensor. Must be one of the following types: bfloat16, half, float32,
     float64, int32, int64, complex64, complex128.
   name: A name for the operation (optional).

  Returns:
   A Tensor. Has the same type as x.

   If x is a SparseTensor, returns SparseTensor(x.indices,
     tf.math.sign(x.values, ...), x.dense_shape).
  """
  x = ops.convert_to_tensor(x)
  if x.dtype.is_complex:
    return gen_math_ops.div_no_nan(
        x,
        cast(
            gen_math_ops.complex_abs(
                x,
                Tout=dtypes.float32
                if x.dtype == dtypes.complex64 else dtypes.float64),
            dtype=x.dtype),
        name=name)
  return gen_math_ops.sign(x, name=name)


@tf_export("math.real", v1=["math.real", "real"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("real")
def real(input, name=None):
  r"""Returns the real part of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the real part of each element in `input` considered as a complex number.

  For example:

  ```python
  x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  tf.math.real(x)  # [-2.25, 3.25]
  ```

  If `input` is already real, it is returned unchanged.

  Args:
    input: A `Tensor`. Must have numeric type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Real", [input]) as name:
    input = ops.convert_to_tensor(input, name="input")
    if input.dtype.is_complex:
      real_dtype = input.dtype.real_dtype
      return gen_math_ops.real(input, Tout=real_dtype, name=name)
    elif input.dtype.is_numeric:
      return input
    else:
      raise TypeError("input must be a numeric tensor, but got tensor with dtype {}".format(input.dtype))


@tf_export("math.imag", v1=["math.imag", "imag"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("imag")
def imag(input, name=None):
  r"""Returns the imaginary part of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the imaginary part of each element in `input` considered as a complex
  number. If `input` is real, a tensor of all zeros is returned.

  For example:

  ```python
  x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  tf.math.imag(x)  # [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float`, `double`,
      `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Imag", [input]) as name:
    input = ops.convert_to_tensor(input, name="input")
    if input.dtype.is_complex:
      return gen_math_ops.imag(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.zeros_like(input)


@tf_export("math.angle", v1=["math.angle", "angle"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("angle")
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
  input = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j], dtype=tf.complex64)
  tf.math.angle(input).numpy()
  # ==> array([2.0131705, 1.056345 ], dtype=float32)
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float`, `double`,
      `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Angle", [input]) as name:
    input = ops.convert_to_tensor(input, name="input")
    if input.dtype.is_complex:
      return gen_math_ops.angle(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.where(input < 0, np.pi * array_ops.ones_like(input),
                             array_ops.zeros_like(input))


# pylint: enable=redefined-outer-name,redefined-builtin


@tf_export("math.round", "round")
@dispatch.register_unary_elementwise_api
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


# TODO(mdan): Include a full_type argument to replace dtype.
@tf_export("cast", "dtypes.cast")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def cast(x, dtype, name=None):
  """Casts a tensor to a new type.

  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor` or `IndexedSlices`) to `dtype`.

  For example:

  >>> x = tf.constant([1.8, 2.2], dtype=tf.float32)
  >>> tf.cast(x, tf.int32)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>

  Notice `tf.cast` has an alias `tf.dtypes.cast`:

  >>> x = tf.constant([1.8, 2.2], dtype=tf.float32)
  >>> tf.dtypes.cast(x, tf.int32)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>

  The operation supports data types (for `x` and `dtype`) of
  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `complex64`, `complex128`, `bfloat16`.
  In case of casting from complex types (`complex64`, `complex128`) to real
  types, only the real part of `x` is returned. In case of casting from real
  types to complex types (`complex64`, `complex128`), the imaginary part of the
  returned value is set to `0`. The handling of complex types here matches the
  behavior of numpy.

  Note casting nan and inf values to integral types has undefined behavior.

  Note this operation can lead to a loss of precision when converting native
  Python `float` and `complex` variables to `tf.float64` or `tf.complex128`
  tensors, since the input is first converted to the `float32` data type and
  then widened. It is recommended to use `tf.convert_to_tensor` instead of
  `tf.cast` for any non-tensor inputs.

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
    elif isinstance(x, indexed_slices.IndexedSlices):
      values_cast = cast(x.values, base_type, name=name)
      x = indexed_slices.IndexedSlices(values_cast, x.indices, x.dense_shape)
    else:
      # TODO(josh11b): If x is not already a Tensor, we could return
      # ops.convert_to_tensor(x, dtype=dtype, ...)  here, but that
      # allows some conversions that cast() can't do, e.g. casting numbers to
      # strings.
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype.is_complex and base_type.is_floating:
        logging.warn(
            f"You are casting an input of type {x.dtype.name} to an "
            f"incompatible dtype {base_type.name}.  This will "
            "discard the imaginary part and may not be what you "
            "intended."
        )
      if x.dtype != base_type:
        x = gen_math_ops.cast(x, base_type, name=name)
    return x


@tf_export("dtypes.saturate_cast", "saturate_cast")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def saturate_cast(value, dtype, name=None):
  """Performs a safe saturating cast of `value` to `dtype`.

  This function casts the input to `dtype` without overflow.  If
  there is a danger that values would over or underflow in the cast, this op
  applies the appropriate clamping before the cast.  See `tf.cast` for more
  details.

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

    in_dtype = value.dtype
    if in_dtype.is_complex:
      if dtype.is_complex:
        # Clamp real and imag components separately, if required.
        real_in_dtype = in_dtype.real_dtype
        real_out_dtype = dtype.real_dtype
        if real_in_dtype.min < real_out_dtype.min or real_in_dtype.max > real_out_dtype.max:
          value = gen_math_ops._clip_by_value(
              value,
              ops.convert_to_tensor(
                  builtins.complex(real_out_dtype.min, real_out_dtype.min),
                  dtype=in_dtype),
              ops.convert_to_tensor(
                  builtins.complex(real_out_dtype.max, real_out_dtype.max),
                  dtype=in_dtype),
              name="clamp")
        return cast(value, dtype, name=name)
      else:
        # Extract real component and fall through to clamp+cast.
        value = real(value)
        logging.warn("Casting complex to real discards imaginary part.")
        in_dtype = in_dtype.real_dtype

    # in_dtype is real, but out_dtype could be complex.
    out_real_dtype = dtype.real_dtype
    if in_dtype.min < out_real_dtype.min or in_dtype.max > out_real_dtype.max:
      # The output min/max may not actually be representable in the
      # in_dtype (e.g. casting float32 to uint32).  This can lead to undefined
      # behavior when trying to cast a value outside the valid range of the
      # target type. We work around this by nudging the min/max to fall within
      # the valid output range.  The catch is that we may actually saturate
      # to a value less than the true saturation limit, but this is the best we
      # can do in order to avoid UB without introducing a separate SaturateCast
      # op.
      min_limit = in_dtype.as_numpy_dtype(out_real_dtype.min)
      if min_limit < out_real_dtype.min:
        min_limit = np.nextafter(
            out_real_dtype.min, 0, dtype=in_dtype.as_numpy_dtype
        )

      max_limit = in_dtype.as_numpy_dtype(out_real_dtype.max)
      if max_limit > out_real_dtype.max:
        max_limit = np.nextafter(
            out_real_dtype.max, 0, dtype=in_dtype.as_numpy_dtype
        )

      value = gen_math_ops._clip_by_value(
          value,
          ops.convert_to_tensor(min_limit, dtype=in_dtype),
          ops.convert_to_tensor(max_limit, dtype=in_dtype),
          name="clamp",
      )
    return cast(value, dtype, name=name)


@tf_export(v1=["to_float"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions="Use `tf.cast` instead.")
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

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.float32)`. There are no further issues with eager execution
  or tf.function.

  Before:

  >>> tf.compat.v1.to_float(tf.constant(3.14, dtype=tf.double))
  <tf.Tensor: shape=(), dtype=float32, numpy=3.14>

  After:

  >>> tf.cast(tf.constant(3.14, dtype=tf.double), tf.float32)
  <tf.Tensor: shape=(), dtype=float32, numpy=3.14>

  @end_compatibility

  """
  return cast(x, dtypes.float32, name=name)


@tf_export(v1=["to_double"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions="Use `tf.cast` instead.")
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

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.double)`. There are no further issues with eager execution or
  tf.function.

  Before:

  >>> tf.compat.v1.to_double(tf.constant(3.14, dtype=tf.float32))
  <tf.Tensor: shape=(), dtype=float64, numpy=3.14>

  After:

  >>> tf.cast(tf.constant(3.14, dtype=tf.float32), tf.double)
  <tf.Tensor: shape=(), dtype=float64, numpy=3.14>

  @end_compatibility

  """
  return cast(x, dtypes.float64, name=name)


@tf_export(v1=["to_int32"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions="Use `tf.cast` instead.")
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

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.int32)`. There are no further issues with eager execution or
  tf.function.

  Before:

  >>> tf.compat.v1.to_int32(tf.constant(1, dtype=tf.int64))
  <tf.Tensor: shape=(), dtype=int32, numpy=1>

  After:

  >>> tf.cast(tf.constant(1, dtype=tf.int64), tf.int32)
  <tf.Tensor: shape=(), dtype=int32, numpy=1>

  @end_compatibility

  """
  return cast(x, dtypes.int32, name=name)


@tf_export(v1=["to_int64"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions="Use `tf.cast` instead.")
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

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.int64)`. There are no further issues with eager execution or
  tf.function.

  Before:

  >>> tf.compat.v1.to_int64(tf.constant(1, dtype=tf.int32))
  <tf.Tensor: shape=(), dtype=int64, numpy=1>

  After:

  >>> tf.cast(tf.constant(1, dtype=tf.int32), tf.int64)
  <tf.Tensor: shape=(), dtype=int64, numpy=1>

  @end_compatibility

  """
  return cast(x, dtypes.int64, name=name)


@tf_export(v1=["to_bfloat16"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions="Use `tf.cast` instead.")
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

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.bfloat16)`. There are no further issues with eager execution
  or tf.function.

  Before:

  >>> tf.compat.v1.to_bfloat16(tf.constant(3.14, dtype=tf.float32))
  <tf.Tensor: shape=(), dtype=bfloat16, numpy=3.14>

  After:

  >>> tf.cast(tf.constant(3.14, dtype=tf.float32), tf.bfloat16)
  <tf.Tensor: shape=(), dtype=bfloat16, numpy=3.14>

  @end_compatibility

  """
  return cast(x, dtypes.bfloat16, name=name)


@tf_export(v1=["to_complex64"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions="Use `tf.cast` instead.")
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

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.complex64)`. There are no further issues with eager execution
  or tf.function.

  Before:

  >>> tf.compat.v1.to_complex64(tf.constant(1. + 2.j, dtype=tf.complex128))
  <tf.Tensor: shape=(), dtype=complex64, numpy=(1+2j)>

  After:

  >>> tf.cast(tf.constant(1. + 2.j, dtype=tf.complex128), tf.complex64)
  <tf.Tensor: shape=(), dtype=complex64, numpy=(1+2j)>

  @end_compatibility

  """
  return cast(x, dtypes.complex64, name=name)


@tf_export(v1=["to_complex128"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions="Use `tf.cast` instead.")
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

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.complex128)`. There are no further issues with eager
  execution or tf.function.

  Before:

  >>> tf.compat.v1.to_complex128(tf.constant(1. + 2.j, dtype=tf.complex64))
  <tf.Tensor: shape=(), dtype=complex128, numpy=(1+2j)>

  After:

  >>> tf.cast(tf.constant(1. + 2.j, dtype=tf.complex64), tf.complex128)
  <tf.Tensor: shape=(), dtype=complex128, numpy=(1+2j)>

  @end_compatibility

  """
  return cast(x, dtypes.complex128, name=name)


ops.Tensor._override_operator("__neg__", gen_math_ops.neg)
ops.Tensor._override_operator("__abs__", abs)


def _maybe_get_dtype(x):
  """Returns a numpy type if available from x. Skips if x is numpy.ndarray."""
  # Don't put np.ndarray in this list, because np.result_type looks at the
  # value (not just dtype) of np.ndarray to decide the result type.
  if isinstance(x, numbers.Real):
    return x
  if isinstance(x, ops.Tensor):
    return x.dtype.as_numpy_dtype
  if isinstance(x, dtypes.DType):
    return x.as_numpy_dtype
  if isinstance(x, tensor_shape.TensorShape):
    return np.int32
  if isinstance(x, (list, tuple)):
    raise ValueError(f"Cannot determine dtype.  Got sequence {x}.")
  return x


def maybe_promote_tensors(*tensors, force_same_dtype=False):
  """Promotes tensors if numpy style promotion is enabled.

  This function promotes `tensors` according to numpy promotion rules
  if numpy style promotion is enabled.  Otherwise, if
  `force_same_dtype` is `True`, it force-casts `tensors[1:]` to
  `tensor[0]`'s dtype. Note that this force-cast can be problematic.
  For example, when some `tensors[1:]` elements can be silently
  downcasted.

  Args:
    *tensors: the list of tensors to promote.
    force_same_dtype: bool (optional, default to `False`). When numpy
      style promotion is disabled and `force_same_dtype` is `True`,
      this function will force-casts `tensors[1:]` to `tensor[0]`'s
      dtype (which could be problematic).

  Returns:
    The promoted list of tensors.
  """
  if not tensors:
    return tensors
  if not ops._numpy_style_type_promotion:
    if not force_same_dtype:
      return tensors
    promoted_tensors = []
    promoted_tensors.append(tensors[0])
    dtype = tensors[0].dtype.base_dtype
    for tensor in tensors[1:]:
      promoted_tensors.append(
          ops.convert_to_tensor(tensor, dtype, name="x"))
    return promoted_tensors
  result_type = np_dtypes._result_type(
      *[_maybe_get_dtype(x) for x in nest.flatten(tensors)])
  def _promote_or_cast(x):
    if isinstance(x, ops.Tensor):
      x = cast(x, result_type)
    else:
      x = ops.convert_to_tensor(x, result_type)
    return x
  return [_promote_or_cast(x) for x in tensors]


def _OverrideBinaryOperatorHelper(func, op_name, clazz_object=ops.Tensor):
  """Register operators with different tensor and scalar versions.

  If `clazz_object` is `SparseTensor`, assumes `func` takes `(sp_indices,
  sp_values, sp_shape, dense)` and outputs `(new_sp_values)`.

  Args:
    func: the operator
    op_name: name of the operator being overridden
    clazz_object: class to override for.  Either `Tensor` or `SparseTensor`.
  """

  @traceback_utils.filter_traceback
  def binary_op_wrapper(x, y):
    with ops.name_scope(None, op_name, [x, y]) as name:
      try:
        # force_same_dtype=False to preserve existing TF behavior
        # TODO(b/178860388): Figure out why binary_op_wrapper and
        #   r_binary_op_wrapper use different force_same_dtype values.
        x, y = maybe_promote_tensors(x, y)
        return func(x, y, name=name)
      except (TypeError, ValueError) as e:
        # Even if dispatching the op failed, the RHS may be a tensor aware
        # object that can implement the operator with knowledge of itself
        # and the tensor.
        # If the RHS is not tensor aware we still want to raise the
        # original error from the LHS, because it may be more
        # informative.
        if hasattr(type(y), "__r%s__" % op_name):
          try:
            r_op = getattr(y, "__r%s__" % op_name)
            out = r_op(x)
            if out is NotImplemented:
              raise
            return out
          except (TypeError, ValueError):
            raise e
        else:
          raise

  @traceback_utils.filter_traceback
  def binary_op_wrapper_sparse(sp_x, y):
    with ops.name_scope(None, op_name, [sp_x, y]) as name:
      y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name="y")
      return sparse_tensor.SparseTensor(
          sp_x.indices,
          func(sp_x.indices, sp_x.values, sp_x.dense_shape, y, name=name),
          sp_x.dense_shape)

  @traceback_utils.filter_traceback
  def r_binary_op_wrapper(y, x):
    with ops.name_scope(None, op_name, [x, y]) as name:
      # TODO(b/178860388): Figure out why binary_op_wrapper and
      #   r_binary_op_wrapper use different force_same_dtype values.
      y, x = maybe_promote_tensors(y, x, force_same_dtype=True)
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
    dtypes.uint32: dtypes.float64,
    dtypes.int32: dtypes.float64,
    dtypes.uint64: dtypes.float64,
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
      raise TypeError(f"`x` and `y` must have the same dtype, "
                      f"got {x_dtype!r} != {y_dtype!r}.")
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError:
      raise TypeError(
          f"Invalid dtype {x_dtype!r} in __truediv__. Expected one "
          f"of {{{', '.join([repr(x) for x in _TRUEDIV_TABLE.keys()])}}}.")
    if dtype is not None:
      sp_values = cast(sp_values, dtype)
      y = cast(y, dtype)
    return gen_sparse_ops.sparse_dense_cwise_div(
        sp_indices, sp_values, sp_shape, y, name=name)


def _truediv_python3(x, y, name=None):
  with ops.name_scope(name, "truediv", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, dtype_hint=x.dtype.base_dtype, name="y")
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError(f"`x` and `y` must have the same dtype, "
                      f"got {x_dtype!r} != {y_dtype!r}.")
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError:
      raise TypeError(
          f"Invalid dtype {x_dtype!r} in __truediv__. Expected one "
          f"of {{{', '.join([repr(x) for x in _TRUEDIV_TABLE.keys()])}}}.")
    if dtype is not None:
      x = cast(x, dtype)
      y = cast(y, dtype)
    return gen_math_ops.real_div(x, y, name=name)


def _div_python2(x, y, name=None):
  """Divide two values using Python 2 semantics.

  Used for Tensor.__div__.

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
      raise TypeError(f"`x` and `y` must have the same dtype, "
                      f"got {x_dtype!r} != {y_dtype!r}.")
    if x_dtype.is_floating or x_dtype.is_complex:
      return gen_math_ops.real_div(x, y, name=name)
    else:
      return gen_math_ops.floor_div(x, y, name=name)


@tf_export("math.truediv", "truediv")
@dispatch.register_binary_elementwise_api
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


@tf_export(v1=["div"])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(
    date=None,
    instructions="Deprecated in favor of operator or tf.math.divide.")
def div(x, y, name=None):
  """Divides x / y elementwise (using Python 2 division operator semantics).

  @compatibility(TF2)
  This function is deprecated in TF2. Prefer using the Tensor division operator,
  `tf.divide`, or `tf.math.divide`, which obey the Python 3 division operator
  semantics.
  @end_compatibility


  This function divides `x` and `y`, forcing Python 2 semantics. That is, if `x`
  and `y` are both integers then the result will be an integer. This is in
  contrast to Python 3, where division with `/` is always a float while division
  with `//` is always an integer.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` returns the quotient of x and y.
  """
  return _div_python2(x, y, name)


@tf_export("math.divide_no_nan", v1=["math.divide_no_nan", "div_no_nan"])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("div_no_nan")
def div_no_nan(x, y, name=None):
  """Computes a safe divide which returns 0 if `y` (denominator) is zero.

  For example:

  >>> tf.constant(3.0) / 0.0
  <tf.Tensor: shape=(), dtype=float32, numpy=inf>
  >>> tf.math.divide_no_nan(3.0, 0.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>

  Note that 0 is returned if `y` is 0 even if `x` is nonfinite:

  >>> tf.math.divide_no_nan(np.nan, 0.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>

  Args:
    x: A `Tensor` of a floating or integer dtype.
    y: A `Tensor` with the same dtype as `x` and a compatible shape.
    name: A name for the operation (optional).

  Returns:
    The element-wise quotient as in `tf.math.divide(x, y)`,
    except that division by zero produces `0.0`, not `nan`.
  """

  with ops.name_scope(name, "div_no_nan", [x, y]) as name:
    if not tensor_util.is_tf_type(x) and tensor_util.is_tf_type(y):
      # Treat this case specially like divide() does above.
      y = ops.convert_to_tensor(y, name="y")
      x = ops.convert_to_tensor(x, dtype=y.dtype.base_dtype, name="x")
    else:
      x = ops.convert_to_tensor(x, name="x")
      y = ops.convert_to_tensor(y, dtype_hint=x.dtype.base_dtype, name="y")
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError(f"`x` and `y` must have the same dtype, "
                      f"got {x_dtype!r} != {y_dtype!r}.")
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError as e:
      raise TypeError(
          f"Invalid dtype {x_dtype!r} in tf.math.divide_no_nan. Expected one "
          f"of {{{', '.join([repr(x) for x in _TRUEDIV_TABLE.keys()])}}}."
      ) from e
    if dtype is not None:
      x = cast(x, dtype)
      y = cast(y, dtype)
    return gen_math_ops.div_no_nan(x, y, name=name)


@tf_export("math.multiply_no_nan")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def multiply_no_nan(x, y, name=None):
  """Computes the product of x and y and returns 0 if the y is zero, even if x is NaN or infinite.

  Note this is noncommutative: if y is NaN or infinite and x is 0, the result
  will be NaN.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    y: A `Tensor` whose dtype is compatible with `x`.
    name: A name for the operation (optional).

  Returns:
    The element-wise value of the x times y.
  """

  with ops.name_scope(name, "multiply_no_nan", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError(f"`x` and `y` must have the same dtype, "
                      f"got {x_dtype!r} != {y_dtype!r}")
    return gen_math_ops.mul_no_nan(x, y, name=name)


# TODO(aselle): This should be removed
mod = gen_math_ops.floor_mod


@tf_export("math.floordiv", v1=["math.floordiv", "floordiv"])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("floordiv")
def floordiv(x, y, name=None):
  """Divides `x / y` elementwise, rounding toward the most negative integer.

  Mathematically, this is equivalent to floor(x / y). For example:
    floor(8.4 / 4.0) = floor(2.1) = 2.0
    floor(-8.4 / 4.0) = floor(-2.1) = -3.0
  This is equivalent to the '//' operator in Python 3.0 and above.

  Note: `x` and `y` must have the same type, and the result will have the same
  type as well.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` rounded toward -infinity.

  Raises:
    TypeError: If the inputs are complex.
  """
  with ops.name_scope(name, "floordiv", [x, y]) as name:
    return gen_math_ops.floor_div(x, y, name=name)


realdiv = gen_math_ops.real_div
truncatediv = gen_math_ops.truncate_div
floor_div = gen_math_ops.floor_div
truncatemod = gen_math_ops.truncate_mod
floormod = gen_math_ops.floor_mod


@tf_export("__operators__.add", v1=[])
@dispatch.add_dispatch_support
def _add_dispatch(x, y, name=None):
  """The operation invoked by the `Tensor.__add__` operator.

  Purpose in the API:

    This method is exposed in TensorFlow's API so that library developers
    can register dispatching for `Tensor.__add__` to allow it to handle
    custom composite tensors & other custom objects.

    The API symbol is not intended to be called by users directly and does
    appear in TensorFlow's generated documentation.

  Args:
    x: The left-hand side of the `+` operator.
    y: The right-hand side of the `+` operator.
    name: an optional name for the operation.

  Returns:
    The result of the elementwise `+` operation.
  """
  if not isinstance(y, ops.Tensor) and not isinstance(
      y, sparse_tensor.SparseTensor):
    y = ops.convert_to_tensor(y, dtype_hint=x.dtype.base_dtype, name="y")
  if x.dtype == dtypes.string:
    return gen_math_ops.add(x, y, name=name)
  else:
    return gen_math_ops.add_v2(x, y, name=name)


def _mul_dispatch(x, y, name=None):
  """Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse"."""
  if isinstance(y, sparse_tensor.SparseTensor):  # Case: Dense * Sparse.
    new_vals = gen_sparse_ops.sparse_dense_cwise_mul(y.indices, y.values,
                                                     y.dense_shape, x, name)
    return sparse_tensor.SparseTensor(y.indices, new_vals, y.dense_shape)
  else:
    return multiply(x, y, name=name)


# NOTE(aselle): When integer division is added for sparse_dense_cwise,
# div, truediv, and floordiv should be delegated appropriately for
# Python semantics, analogous to dense cwise tensor operations.
_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_div, "div",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(_sparse_dense_truediv, "truediv",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_mul, "mul",
                              sparse_tensor.SparseTensor)

_OverrideBinaryOperatorHelper(_add_dispatch, "add")
_OverrideBinaryOperatorHelper(subtract, "sub")
_OverrideBinaryOperatorHelper(_mul_dispatch, "mul")
_OverrideBinaryOperatorHelper(div, "div")
_OverrideBinaryOperatorHelper(truediv, "truediv")
_OverrideBinaryOperatorHelper(floordiv, "floordiv")
_OverrideBinaryOperatorHelper(gen_math_ops.floor_mod, "mod")
_OverrideBinaryOperatorHelper(pow, "pow")


@tf_export("math.logical_xor", v1=["math.logical_xor", "logical_xor"])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("logical_xor")
def logical_xor(x, y, name="LogicalXor"):
  """Logical XOR function.

  x ^ y = (x | y) & ~(x & y)

  Requires that `x` and `y` have the same shape or have
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  shapes. For example, `x` and `y` can be:

  - Two single elements of type `bool`
  - One `tf.Tensor` of type `bool` and one single `bool`, where the result will
    be calculated by applying logical XOR with the single element to each
    element in the larger Tensor.
  - Two `tf.Tensor` objects of type `bool` of the same shape. In this case,
    the result will be the element-wise logical XOR of the two input tensors.

  Usage:

  >>> a = tf.constant([True])
  >>> b = tf.constant([False])
  >>> tf.math.logical_xor(a, b)
  <tf.Tensor: shape=(1,), dtype=bool, numpy=array([ True])>

  >>> c = tf.constant([True])
  >>> x = tf.constant([False, True, True, False])
  >>> tf.math.logical_xor(c, x)
  <tf.Tensor: shape=(4,), dtype=bool, numpy=array([ True, False, False,  True])>

  >>> y = tf.constant([False, False, True, True])
  >>> z = tf.constant([False, True, False, True])
  >>> tf.math.logical_xor(y, z)
  <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False,  True,  True, False])>

  Args:
      x: A `tf.Tensor` type bool.
      y: A `tf.Tensor` of type bool.
      name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of type bool with the same size as that of x or y.
  """
  # TODO(alemi) Make this a cwise op if people end up relying on it.
  return gen_math_ops.logical_and(
      gen_math_ops.logical_or(x, y),
      gen_math_ops.logical_not(gen_math_ops.logical_and(x, y)),
      name=name)


def and_(x, y, name=None):
  if x.dtype == dtypes.bool:
    return gen_math_ops.logical_and(x, y, name)
  return gen_bitwise_ops.bitwise_and(x, y)


def or_(x, y, name=None):
  if x.dtype == dtypes.bool:
    return gen_math_ops.logical_or(x, y, name)
  return gen_bitwise_ops.bitwise_or(x, y)


def xor_(x, y, name=None):
  if x.dtype == dtypes.bool:
    return logical_xor(x, y, name)
  return gen_bitwise_ops.bitwise_xor(x, y)


def invert_(x, name=None):
  if x.dtype == dtypes.bool:
    return gen_math_ops.logical_not(x, name=name)
  return gen_bitwise_ops.invert(x, name=name)


_OverrideBinaryOperatorHelper(and_, "and")
_OverrideBinaryOperatorHelper(or_, "or")
_OverrideBinaryOperatorHelper(xor_, "xor")
ops.Tensor._override_operator("__invert__", invert_)


def _promote_dtypes_decorator(fn):
  def wrapper(x, y, *args, **kwargs):
    x, y = maybe_promote_tensors(x, y)
    return fn(x, y, *args, **kwargs)
  return tf_decorator.make_decorator(fn, wrapper)


ops.Tensor._override_operator("__lt__", _promote_dtypes_decorator(
    gen_math_ops.less))
ops.Tensor._override_operator("__le__", _promote_dtypes_decorator(
    gen_math_ops.less_equal))
ops.Tensor._override_operator("__gt__", _promote_dtypes_decorator(
    gen_math_ops.greater))
ops.Tensor._override_operator("__ge__", _promote_dtypes_decorator(
    gen_math_ops.greater_equal))


@tf_export("math.equal", "equal")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def equal(x, y, name=None):
  """Returns the truth value of (x == y) element-wise.

  Performs a [broadcast](
  https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) with the
  arguments and then an element-wise equality comparison, returning a Tensor of
  boolean values.

  For example:

  >>> x = tf.constant([2, 4])
  >>> y = tf.constant(2)
  >>> tf.math.equal(x, y)
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True,  False])>

  >>> x = tf.constant([2, 4])
  >>> y = tf.constant([2, 4])
  >>> tf.math.equal(x, y)
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True,  True])>

  Args:
    x: A `tf.Tensor`.
    y: A `tf.Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of type bool with the same size as that of x or y.

  Raises:
    `tf.errors.InvalidArgumentError`: If shapes of arguments are incompatible
  """
  return gen_math_ops.equal(x, y, name=name)


@tf_export("math.not_equal", "not_equal")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def not_equal(x, y, name=None):
  """Returns the truth value of (x != y) element-wise.

  Performs a [broadcast](
  https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) with the
  arguments and then an element-wise inequality comparison, returning a Tensor
  of boolean values.

  For example:

  >>> x = tf.constant([2, 4])
  >>> y = tf.constant(2)
  >>> tf.math.not_equal(x, y)
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  True])>

  >>> x = tf.constant([2, 4])
  >>> y = tf.constant([2, 4])
  >>> tf.math.not_equal(x, y)
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  False])>

  Args:
    x: A `tf.Tensor`.
    y: A `tf.Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of type bool with the same size as that of x or y.

  Raises:
    `tf.errors.InvalidArgumentError`: If shapes of arguments are incompatible
  """
  return gen_math_ops.not_equal(x, y, name=name)


@tf_export("__operators__.eq", v1=[])
@dispatch.add_dispatch_support
def tensor_equals(self, other):
  """The operation invoked by the `Tensor.__eq__` operator.

  Compares two tensors element-wise for equality if they are
  broadcast-compatible; or returns False if they are not broadcast-compatible.
  (Note that this behavior differs from `tf.math.equal`, which raises an
  exception if the two tensors are not broadcast-compatible.)

  Purpose in the API:

    This method is exposed in TensorFlow's API so that library developers
    can register dispatching for `Tensor.__eq__` to allow it to handle
    custom composite tensors & other custom objects.

    The API symbol is not intended to be called by users directly and does
    appear in TensorFlow's generated documentation.

  Args:
    self: The left-hand side of the `==` operator.
    other: The right-hand side of the `==` operator.

  Returns:
    The result of the elementwise `==` operation, or `False` if the arguments
    are not broadcast-compatible.
  """
  if other is None:
    return False
  g = getattr(self, "graph", None)
  if (ops.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions() and
      (g is None or g.building_function)):
    self, other = maybe_promote_tensors(self, other)
    return gen_math_ops.equal(self, other, incompatible_shape_error=False)
  else:
    # In legacy graph mode, tensor equality is object equality
    return self is other


@tf_export("__operators__.ne", v1=[])
@dispatch.add_dispatch_support
def tensor_not_equals(self, other):
  """The operation invoked by the `Tensor.__ne__` operator.

  Compares two tensors element-wise for inequality if they are
  broadcast-compatible; or returns True if they are not broadcast-compatible.
  (Note that this behavior differs from `tf.math.not_equal`, which raises an
  exception if the two tensors are not broadcast-compatible.)

  Purpose in the API:

    This method is exposed in TensorFlow's API so that library developers
    can register dispatching for `Tensor.__ne__` to allow it to handle
    custom composite tensors & other custom objects.

    The API symbol is not intended to be called by users directly and does
    appear in TensorFlow's generated documentation.

  Args:
    self: The left-hand side of the `!=` operator.
    other: The right-hand side of the `!=` operator.

  Returns:
    The result of the elementwise `!=` operation, or `True` if the arguments
    are not broadcast-compatible.
  """
  if other is None:
    return True
  if ops.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions():
    self, other = maybe_promote_tensors(self, other)
    return gen_math_ops.not_equal(self, other, incompatible_shape_error=False)
  else:
    # In legacy graph mode, tensor equality is object equality
    return self is not other


ops.Tensor._override_operator("__eq__", tensor_equals)
ops.Tensor._override_operator("__ne__", tensor_not_equals)


@tf_export("range")
@dispatch.add_dispatch_support
def range(start, limit=None, delta=1, dtype=None, name="range"):  # pylint: disable=redefined-builtin
  """Creates a sequence of numbers.

  Creates a sequence of numbers that begins at `start` and extends by
  increments of `delta` up to but not including `limit`.

  The dtype of the resulting tensor is inferred from the inputs unless
  it is provided explicitly.

  Like the Python builtin `range`, `start` defaults to 0, so that
  `range(n) = range(0, n)`.

  For example:

  >>> start = 3
  >>> limit = 18
  >>> delta = 3
  >>> tf.range(start, limit, delta)
  <tf.Tensor: shape=(5,), dtype=int32,
  numpy=array([ 3,  6,  9, 12, 15], dtype=int32)>

  >>> start = 3
  >>> limit = 1
  >>> delta = -0.5
  >>> tf.range(start, limit, delta)
  <tf.Tensor: shape=(4,), dtype=float32,
  numpy=array([3. , 2.5, 2. , 1.5], dtype=float32)>

  >>> limit = 5
  >>> tf.range(limit)
  <tf.Tensor: shape=(5,), dtype=int32,
  numpy=array([0, 1, 2, 3, 4], dtype=int32)>

  Args:
    start: A 0-D `Tensor` (scalar). Acts as first entry in the range if `limit`
      is not None; otherwise, acts as range limit and first entry defaults to 0.
    limit: A 0-D `Tensor` (scalar). Upper limit of sequence, exclusive. If None,
      defaults to the value of `start` while the first entry of the range
      defaults to 0.
    delta: A 0-D `Tensor` (scalar). Number that increments `start`. Defaults to
      1.
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
    if not isinstance(start, ops.Tensor):
      start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    if not isinstance(limit, ops.Tensor):
      limit = ops.convert_to_tensor(limit, dtype=dtype, name="limit")
    if not isinstance(delta, ops.Tensor):
      delta = ops.convert_to_tensor(delta, dtype=dtype, name="delta")

    # infer dtype if not explicitly provided
    if dtype is None:
      dtype_hierarchy = [
          dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
      ]
      assert all(arg.dtype in dtype_hierarchy for arg in [start, limit, delta])
      inferred_dtype = max([arg.dtype for arg in [start, limit, delta]],
                           key=dtype_hierarchy.index)
    else:
      inferred_dtype = dtype
    # Always try to perform a cast even when start/limit/delta are already
    # tensors. This will resolve the case where start/limit/delta's original's
    # dtype is different from provided dtype.
    start = cast(start, inferred_dtype)
    limit = cast(limit, inferred_dtype)
    delta = cast(delta, inferred_dtype)

    return gen_math_ops._range(start, limit, delta, name=name)


def _range_tensor_conversion_function(value, dtype=None, name=None,
                                      as_ref=False):
  del as_ref
  return range(value.start, value.stop, value.step, dtype=dtype, name=name)


tensor_conversion_registry.register_tensor_conversion_function(
    builtins.range, _range_tensor_conversion_function)


# Reduction operations
def _ReductionDims(x, axis):  # pylint: disable=invalid-name
  """Returns range(0, rank(x)) if axis is None."""
  if axis is not None:
    return axis
  else:
    try:
      x_rank = x.shape.rank
    except AttributeError:
      x_rank = None

    # Fast path: avoid creating Rank and Range ops if ndims is known.
    if x_rank:
      return constant_op.constant(np.arange(x_rank, dtype=np.int32))
    else:
      # Otherwise, we rely on Range and Rank to do the right thing at run-time.
      return range(0, array_ops.rank(x))


def _has_fully_defined_shape(tensor):
  """Returns true if tensor has a fully defined shape."""
  return isinstance(tensor, ops.EagerTensor) or tensor.shape.is_fully_defined()


def _may_reduce_to_scalar(keepdims, axis, output):
  """Set a reduction's output shape to be a scalar if we are certain."""
  if not _has_fully_defined_shape(output) and (not keepdims) and (
      axis is None):
    output.set_shape(())
  return output


@tf_export(v1=["math.reduce_sum", "reduce_sum"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
def reduce_sum_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes the sum of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.add` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> # x has a shape of (2, 3) (two rows and three columns):
    >>> x = tf.constant([[1, 1, 1], [1, 1, 1]])
    >>> x.numpy()
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
    >>> # sum all the elements
    >>> # 1 + 1 + 1 + 1 + 1+ 1 = 6
    >>> tf.reduce_sum(x).numpy()
    6
    >>> # reduce along the first dimension
    >>> # the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> tf.reduce_sum(x, 0).numpy()
    array([2, 2, 2], dtype=int32)
    >>> # reduce along the second dimension
    >>> # the result is [1, 1] + [1, 1] + [1, 1] = [3, 3]
    >>> tf.reduce_sum(x, 1).numpy()
    array([3, 3], dtype=int32)
    >>> # keep the original dimensions
    >>> tf.reduce_sum(x, 1, keepdims=True).numpy()
    array([[3],
           [3]], dtype=int32)
    >>> # reduce along both dimensions
    >>> # the result is 1 + 1 + 1 + 1 + 1 + 1 = 6
    >>> # or, equivalently, reduce along rows, then reduce the resultant array
    >>> # [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> # 2 + 2 + 2 = 6
    >>> tf.reduce_sum(x, [0, 1]).numpy()
    6

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
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while tensorflow returns the same dtype as the input.
  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_sum(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_sum", "reduce_sum", v1=[])
@dispatch.add_dispatch_support
def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the sum of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.add` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> # x has a shape of (2, 3) (two rows and three columns):
    >>> x = tf.constant([[1, 1, 1], [1, 1, 1]])
    >>> x.numpy()
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
    >>> # sum all the elements
    >>> # 1 + 1 + 1 + 1 + 1+ 1 = 6
    >>> tf.reduce_sum(x).numpy()
    6
    >>> # reduce along the first dimension
    >>> # the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> tf.reduce_sum(x, 0).numpy()
    array([2, 2, 2], dtype=int32)
    >>> # reduce along the second dimension
    >>> # the result is [1, 1] + [1, 1] + [1, 1] = [3, 3]
    >>> tf.reduce_sum(x, 1).numpy()
    array([3, 3], dtype=int32)
    >>> # keep the original dimensions
    >>> tf.reduce_sum(x, 1, keepdims=True).numpy()
    array([[3],
           [3]], dtype=int32)
    >>> # reduce along both dimensions
    >>> # the result is 1 + 1 + 1 + 1 + 1 + 1 = 6
    >>> # or, equivalently, reduce along rows, then reduce the resultant array
    >>> # [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> # 2 + 2 + 2 = 6
    >>> tf.reduce_sum(x, [0, 1]).numpy()
    6

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor)]`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while tensorflow returns the same dtype as the input.
  @end_compatibility
  """

  return reduce_sum_with_dims(input_tensor, axis, keepdims, name,
                              _ReductionDims(input_tensor, axis))


def reduce_sum_with_dims(input_tensor,
                         axis=None,
                         keepdims=False,
                         name=None,
                         dims=None):
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._sum(input_tensor, dims, keepdims, name=name))


@tf_export("math.reduce_euclidean_norm")
@dispatch.add_dispatch_support
def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the Euclidean norm of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1, 2, 3], [1, 1, 1]]) # x.dtype is tf.int32
  tf.math.reduce_euclidean_norm(x)  # returns 4 as dtype is tf.int32
  y = tf.constant([[1, 2, 3], [1, 1, 1]], dtype = tf.float32)
  tf.math.reduce_euclidean_norm(y)  # returns 4.1231055 which is sqrt(17)
  tf.math.reduce_euclidean_norm(y, 0)  # [sqrt(2), sqrt(5), sqrt(10)]
  tf.math.reduce_euclidean_norm(y, 1)  # [sqrt(14), sqrt(3)]
  tf.math.reduce_euclidean_norm(y, 1, keepdims=True)  # [[sqrt(14)], [sqrt(3)]]
  tf.math.reduce_euclidean_norm(y, [0, 1])  # sqrt(17)
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
  """
  keepdims = bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops.euclidean_norm(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.count_nonzero", "count_nonzero"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
@deprecation.deprecated_args(
    None, "reduction_indices is deprecated, use axis instead",
    "reduction_indices")
def count_nonzero(input_tensor=None,
                  axis=None,
                  keepdims=None,
                  dtype=dtypes.int64,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None,
                  input=None):  # pylint: disable=redefined-builtin
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
  tf.math.count_nonzero(x)  # 3
  tf.math.count_nonzero(x, 0)  # [1, 2, 0]
  tf.math.count_nonzero(x, 1)  # [1, 2]
  tf.math.count_nonzero(x, 1, keepdims=True)  # [[1], [2]]
  tf.math.count_nonzero(x, [0, 1])  # 3
  ```

  **NOTE** Strings are compared against zero-length empty string `""`. Any
  string with a size greater than zero is already considered as nonzero.

  For example:
  ```python
  x = tf.constant(["", "a", "  ", "b", ""])
  tf.math.count_nonzero(x) # 3, with "a", "  ", and "b" as nonzero strings.
  ```

  Args:
    input_tensor: The tensor to reduce. Should be of numeric type, `bool`, or
      `string`.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    dtype: The output dtype; defaults to `tf.int64`.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.
    input: Overrides input_tensor. For compatibility.

  Returns:
    The reduced tensor (number of nonzero values).
  """
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  input_tensor = deprecation.deprecated_argument_lookup("input", input,
                                                        "input_tensor",
                                                        input_tensor)
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)

  return count_nonzero_v2(input_tensor, axis, keepdims, dtype, name)


@tf_export("math.count_nonzero", v1=[])
@dispatch.add_dispatch_support
def count_nonzero_v2(
    input,  # pylint: disable=redefined-builtin
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
  tf.math.count_nonzero(x)  # 3
  tf.math.count_nonzero(x, 0)  # [1, 2, 0]
  tf.math.count_nonzero(x, 1)  # [1, 2]
  tf.math.count_nonzero(x, 1, keepdims=True)  # [[1], [2]]
  tf.math.count_nonzero(x, [0, 1])  # 3
  ```

  **NOTE** Strings are compared against zero-length empty string `""`. Any
  string with a size greater than zero is already considered as nonzero.

  For example:
  ```python
  x = tf.constant(["", "a", "  ", "b", ""])
  tf.math.count_nonzero(x) # 3, with "a", "  ", and "b" as nonzero strings.
  ```

  Args:
    input: The tensor to reduce. Should be of numeric type, `bool`, or `string`.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input), rank(input))`.
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
            cast(gen_math_ops.not_equal(input, zero), dtypes.int64),
            axis=axis,
            keepdims=keepdims),
        dtype=dtype)


@tf_export(v1=["math.reduce_mean", "reduce_mean"])
@dispatch.add_dispatch_support
def reduce_mean_v1(input_tensor,
                   axis=None,
                   keepdims=None,
                   name=None,
                   reduction_indices=None,
                   keep_dims=None):
  """Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis` by computing the
  mean of elements across the dimensions in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a tensor with a single
  element is returned.

  For example:

  >>> x = tf.constant([[1., 1.], [2., 2.]])
  >>> tf.reduce_mean(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.5>
  >>> tf.reduce_mean(x, 0)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.5, 1.5], dtype=float32)>
  >>> tf.reduce_mean(x, 1)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>

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
  Equivalent to np.mean

  Please note that `np.mean` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `tf.reduce_mean` has an aggressive type inference from `input_tensor`,
  for example:

  >>> x = tf.constant([1, 0, 1, 0])
  >>> tf.reduce_mean(x)
  <tf.Tensor: shape=(), dtype=int32, numpy=0>
  >>> y = tf.constant([1., 0., 1., 0.])
  >>> tf.reduce_mean(y)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.5>

  @end_compatibility
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_mean(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_mean", "reduce_mean", v1=[])
@dispatch.add_dispatch_support
def reduce_mean(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis` by computing the
  mean of elements across the dimensions in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a tensor with a single
  element is returned.

  For example:

  >>> x = tf.constant([[1., 1.], [2., 2.]])
  >>> tf.reduce_mean(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.5>
  >>> tf.reduce_mean(x, 0)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.5, 1.5], dtype=float32)>
  >>> tf.reduce_mean(x, 1)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>

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

  >>> x = tf.constant([1, 0, 1, 0])
  >>> tf.reduce_mean(x)
  <tf.Tensor: shape=(), dtype=int32, numpy=0>
  >>> y = tf.constant([1., 0., 1., 0.])
  >>> tf.reduce_mean(y)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.5>

  @end_compatibility
  """
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops.mean(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export("math.reduce_variance")
@dispatch.add_dispatch_support
def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the variance of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  >>> x = tf.constant([[1., 2.], [3., 4.]])
  >>> tf.math.reduce_variance(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.25>
  >>> tf.math.reduce_variance(x, 0)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], ...)>
  >>> tf.math.reduce_variance(x, 1)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.25, 0.25], ...)>

  Args:
    input_tensor: The tensor to reduce. Should have real or complex type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name scope for the associated operations (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor. Note,  for
    `complex64` or `complex128` input, the returned `Tensor` will be of type
    `float32` or `float64`, respectively.

  @compatibility(numpy)
  Equivalent to np.var

  Please note `np.var` has a `dtype` parameter that could be used to specify the
  output type. By default this is `dtype=float64`. On the other hand,
  `tf.math.reduce_variance` has aggressive type inference from `input_tensor`.
  @end_compatibility
  """
  name = name if name else "reduce_variance"
  with ops.name_scope(name):
    input_tensor = ops.convert_to_tensor(input_tensor)
    means = reduce_mean(input_tensor, axis=axis, keepdims=True)
    if means.dtype.is_integer:
      raise TypeError(f"Input must be either real or complex. "
                      f"Received integer type {means.dtype}.")
    diff = input_tensor - means
    if diff.dtype.is_complex:
      # For complex values we need to take the absolute value before squaring.
      # This is achieved by multiplying with the conjugate.
      real_dtype = diff.dtype.real_dtype
      squared_deviations = gen_math_ops.real(
          gen_math_ops.mul(gen_math_ops.conj(diff), diff), Tout=real_dtype)
    else:
      squared_deviations = gen_math_ops.square(diff)
    return reduce_mean(squared_deviations, axis=axis, keepdims=keepdims)


@tf_export("math.reduce_std")
@dispatch.add_dispatch_support
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the standard deviation of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  >>> x = tf.constant([[1., 2.], [3., 4.]])
  >>> tf.math.reduce_std(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.118034>
  >>> tf.math.reduce_std(x, 0)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>
  >>> tf.math.reduce_std(x, 1)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.5, 0.5], dtype=float32)>

  Args:
    input_tensor: The tensor to reduce. Should have real or complex type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name scope for the associated operations (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor. Note,  for
    `complex64` or `complex128` input, the returned `Tensor` will be of type
    `float32` or `float64`, respectively.

  @compatibility(numpy)
  Equivalent to np.std

  Please note `np.std` has a `dtype` parameter that could be used to specify the
  output type. By default this is `dtype=float64`. On the other hand,
  `tf.math.reduce_std` has aggressive type inference from `input_tensor`.
  @end_compatibility
  """
  name = name if name else "reduce_std"
  with ops.name_scope(name):
    input_tensor = ops.convert_to_tensor(input_tensor)
    variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
    return gen_math_ops.sqrt(variance)


@tf_export("math.reduce_prod", "reduce_prod", v1=[])
@dispatch.add_dispatch_support
def reduce_prod(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `tf.math.multiply` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.multiply` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> tf.math.reduce_prod(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=24.>
    >>> tf.math.reduce_prod(x, 0)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([3., 8.], dtype=float32)>
    >>> tf.math.reduce_prod(x, 1)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 12.],
    dtype=float32)>

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
  Equivalent to np.prod
  @end_compatibility
  """
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops.prod(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_prod", "reduce_prod"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
def reduce_prod_v1(input_tensor,
                   axis=None,
                   keepdims=None,
                   name=None,
                   reduction_indices=None,
                   keep_dims=None):
  """Computes `tf.math.multiply` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.multiply` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> tf.math.reduce_prod(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=24.>
    >>> tf.math.reduce_prod(x, 0)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([3., 8.], dtype=float32)>
    >>> tf.math.reduce_prod(x, 1)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 12.],
    dtype=float32)>

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
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_prod(input_tensor, axis, keepdims, name)


@tf_export(v1=["math.reduce_min", "reduce_min"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
def reduce_min_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes the `tf.math.minimum` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.minimum` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Usage example:

    >>> x = tf.constant([5, 1, 2, 4])
    >>> tf.reduce_min(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=1>
    >>> x = tf.constant([-5, -1, -2, -4])
    >>> tf.reduce_min(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=-5>
    >>> x = tf.constant([4, float('nan')])
    >>> tf.reduce_min(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('nan'), float('nan')])
    >>> tf.reduce_min(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('-inf'), float('inf')])
    >>> tf.reduce_min(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=-inf>

  See the numpy docs for `np.amin` and `np.nanmin` behavior.

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
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_min(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_min", "reduce_min", v1=[])
@dispatch.add_dispatch_support
def reduce_min(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the `tf.math.minimum` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.minimum` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  >>> a = tf.constant([
  ...   [[1, 2], [3, 4]],
  ...   [[1, 2], [3, 4]]
  ... ])
  >>> tf.reduce_min(a)
  <tf.Tensor: shape=(), dtype=int32, numpy=1>

  Choosing a specific axis returns minimum element in the given axis:

  >>> b = tf.constant([[1, 2, 3], [4, 5, 6]])
  >>> tf.reduce_min(b, axis=0)
  <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
  >>> tf.reduce_min(b, axis=1)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 4], dtype=int32)>

  Setting `keepdims` to `True` retains the dimension of `input_tensor`:

  >>> tf.reduce_min(a, keepdims=True)
  <tf.Tensor: shape=(1, 1, 1), dtype=int32, numpy=array([[[1]]], dtype=int32)>
  >>> tf.math.reduce_min(a, axis=0, keepdims=True)
  <tf.Tensor: shape=(1, 2, 2), dtype=int32, numpy=
  array([[[1, 2],
          [3, 4]]], dtype=int32)>

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
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._min(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_max", "reduce_max"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
def reduce_max_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes `tf.math.maximum` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.maximum` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Usage example:

    >>> x = tf.constant([5, 1, 2, 4])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=5>
    >>> x = tf.constant([-5, -1, -2, -4])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=-1>
    >>> x = tf.constant([4, float('nan')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('nan'), float('nan')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('-inf'), float('inf')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=inf>

  See the numpy docs for `np.amax` and `np.nanmax` behavior.

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
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_max(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_max", "reduce_max", v1=[])
@dispatch.add_dispatch_support
def reduce_max(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `tf.math.maximum` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.maximum` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Usage example:

    >>> x = tf.constant([5, 1, 2, 4])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=5>
    >>> x = tf.constant([-5, -1, -2, -4])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=-1>
    >>> x = tf.constant([4, float('nan')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('nan'), float('nan')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('-inf'), float('inf')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=inf>

  See the numpy docs for `np.amax` and `np.nanmax` behavior.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
  return reduce_max_with_dims(input_tensor, axis, keepdims, name,
                              _ReductionDims(input_tensor, axis))


def reduce_max_with_dims(input_tensor,
                         axis=None,
                         keepdims=False,
                         name=None,
                         dims=None):
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._max(input_tensor, dims, keepdims, name=name))


@tf_export(v1=["math.reduce_all", "reduce_all"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
def reduce_all_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes `tf.math.logical_and` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.logical_and` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = tf.constant([[True,  True], [False, False]])
    >>> tf.math.reduce_all(x)
    <tf.Tensor: shape=(), dtype=bool, numpy=False>
    >>> tf.math.reduce_all(x, 0)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False, False])>
    >>> tf.math.reduce_all(x, 1)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>

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
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_all(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_all", "reduce_all", v1=[])
@dispatch.add_dispatch_support
def reduce_all(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `tf.math.logical_and` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.logical_and` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = tf.constant([[True,  True], [False, False]])
    >>> tf.math.reduce_all(x)
    <tf.Tensor: shape=(), dtype=bool, numpy=False>
    >>> tf.math.reduce_all(x, 0)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False, False])>
    >>> tf.math.reduce_all(x, 1)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>

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
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._all(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_any", "reduce_any"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
def reduce_any_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  """Computes `tf.math.logical_or` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.logical_or` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = tf.constant([[True,  True], [False, False]])
    >>> tf.reduce_any(x)
    <tf.Tensor: shape=(), dtype=bool, numpy=True>
    >>> tf.reduce_any(x, 0)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True,  True])>
    >>> tf.reduce_any(x, 1)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>

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
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_any(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_any", "reduce_any", v1=[])
@dispatch.add_dispatch_support
def reduce_any(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `tf.math.logical_or` of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.logical_or` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> x = tf.constant([[True,  True], [False, False]])
    >>> tf.reduce_any(x)
    <tf.Tensor: shape=(), dtype=bool, numpy=True>
    >>> tf.reduce_any(x, 0)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True,  True])>
    >>> tf.reduce_any(x, 1)
    <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>

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
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops._any(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))


@tf_export(v1=["math.reduce_logsumexp", "reduce_logsumexp"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             "keep_dims is deprecated, use keepdims instead",
                             "keep_dims")
def reduce_logsumexp_v1(input_tensor,
                        axis=None,
                        keepdims=None,
                        name=None,
                        reduction_indices=None,
                        keep_dims=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

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
  axis = deprecation.deprecated_argument_lookup("axis", axis,
                                                "reduction_indices",
                                                reduction_indices)
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  return reduce_logsumexp(input_tensor, axis, keepdims, name)


@tf_export("math.reduce_logsumexp", "reduce_logsumexp", v1=[])
@dispatch.add_dispatch_support
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

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
  with ops.name_scope(name, "ReduceLogSumExp", [input_tensor]) as name:
    raw_max = reduce_max(input_tensor, axis=axis, keepdims=True)
    my_max = array_ops.stop_gradient(
        gen_math_ops.select(
            gen_math_ops.is_finite(raw_max), raw_max,
            gen_array_ops.zeros_like(raw_max)))
    result = gen_math_ops.log(
        reduce_sum(
            exp(subtract(input_tensor, my_max)),
            axis=axis,
            keepdims=keepdims))
    if not keepdims:
      my_max = array_ops.reshape(my_max, gen_array_ops.shape(result))
    result = add(result, my_max, name=name)
    return _may_reduce_to_scalar(keepdims, axis, result)


@tf_export("linalg.trace", v1=["linalg.trace", "trace"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("trace")
def trace(x, name=None):
  """Compute the trace of a tensor `x`.

  `trace(x)` returns the sum along the main diagonal of each inner-most matrix
  in x. If x is of rank `k` with shape `[I, J, K, ..., L, M, N]`, then output
  is a tensor of rank `k-2` with dimensions `[I, J, K, ..., L]` where

  `output[i, j, k, ..., l] = trace(x[i, j, k, ..., l, :, :])`

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
@dispatch.add_dispatch_support
def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           output_type=None,
           name=None):
  """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

  The inputs must, following any transpositions, be tensors of rank >= 2
  where the inner 2 dimensions specify valid matrix multiplication dimensions,
  and any further outer dimensions specify matching batch size.

  Both matrices must be of the same type. The supported types are:
  `bfloat16`, `float16`, `float32`, `float64`, `int32`, `int64`,
  `complex64`, `complex128`.

  Either matrix can be transposed or adjointed (conjugated and transposed) on
  the fly by setting one of the corresponding flag to `True`. These are `False`
  by default.

  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
  This optimization is only available for plain matrices (rank-2 tensors) with
  datatypes `bfloat16` or `float32`.

  A simple 2-D tensor matrix multiplication:

  >>> a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
  >>> a  # 2-D tensor
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[1, 2, 3],
         [4, 5, 6]], dtype=int32)>
  >>> b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
  >>> b  # 2-D tensor
  <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
  array([[ 7,  8],
         [ 9, 10],
         [11, 12]], dtype=int32)>
  >>> c = tf.matmul(a, b)
  >>> c  # `a` * `b`
  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
  array([[ 58,  64],
         [139, 154]], dtype=int32)>

  A batch matrix multiplication with batch shape [2]:

  >>> a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
  >>> a  # 3-D tensor
  <tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
  array([[[ 1,  2,  3],
          [ 4,  5,  6]],
         [[ 7,  8,  9],
          [10, 11, 12]]], dtype=int32)>
  >>> b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
  >>> b  # 3-D tensor
  <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
  array([[[13, 14],
          [15, 16],
          [17, 18]],
         [[19, 20],
          [21, 22],
          [23, 24]]], dtype=int32)>
  >>> c = tf.matmul(a, b)
  >>> c  # `a` * `b`
  <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
  array([[[ 94, 100],
          [229, 244]],
         [[508, 532],
          [697, 730]]], dtype=int32)>

  Since python >= 3.5 the @ operator is supported
  (see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
  it simply calls the `tf.matmul()` function, so the following lines are
  equivalent:

  >>> d = a @ b @ [[10], [11]]
  >>> d = tf.matmul(tf.matmul(a, b), [[10], [11]])

  Args:
    a: `tf.Tensor` of type `float16`, `float32`, `float64`, `int32`,
      `complex64`, `complex128` and rank > 1.
    b: `tf.Tensor` with same type and rank as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated and transposed before
      multiplication.
    adjoint_b: If `True`, `b` is conjugated and transposed before
      multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix. Notice, this
      **does not support `tf.sparse.SparseTensor`**, it just makes optimizations
      that assume most values in `a` are zero.
      See `tf.sparse.sparse_dense_matmul`
      for some support for `tf.sparse.SparseTensor` multiplication.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix. Notice, this
      **does not support `tf.sparse.SparseTensor`**, it just makes optimizations
      that assume most values in `b` are zero.
      See `tf.sparse.sparse_dense_matmul`
      for some support for `tf.sparse.SparseTensor` multiplication.
    output_type: The output datatype if needed. Defaults to None in which case
      the output_type is the same as input type. Currently only works when input
      tensors are type (u)int8 and output_type can be int32.
    name: Name for the operation (optional).

  Returns:
    A `tf.Tensor` of the same type as `a` and `b` where each inner-most matrix
    is the product of the corresponding matrices in `a` and `b`, e.g. if all
    transpose or adjoint attributes are `False`:

    `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
    for all indices `i`, `j`.

    Note: This is matrix product, not element-wise product.


  Raises:
    ValueError: If `transpose_a` and `adjoint_a`, or `transpose_b` and
      `adjoint_b` are both set to `True`.
    TypeError: If output_type is specified but the types of `a`, `b` and
      `output_type` is not (u)int8, (u)int8 and int32.
  """

  with ops.name_scope(name, "MatMul", [a, b]) as name:
    if transpose_a and adjoint_a:
      raise ValueError(
          f"Only one of `transpose_a` and `adjoint_a` can be True. "
          f"Received `transpose_a`={transpose_a}, "
          f"`adjoint_a`={adjoint_a}.")
    if transpose_b and adjoint_b:
      raise ValueError(
          f"Only one of `transpose_b` and `adjoint_b` can be True. "
          f"Received `transpose_b`={transpose_b}, "
          f"`adjoint_b`={adjoint_b}.")

    if context.executing_eagerly():
      if not isinstance(a, (ops.EagerTensor, _resource_variable_type)):
        a = ops.convert_to_tensor(a, name="a")
      if not isinstance(b, (ops.EagerTensor, _resource_variable_type)):
        b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")
    else:
      a = ops.convert_to_tensor(a, name="a")
      b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")

    # TODO(apassos) remove _shape_tuple here when it is not needed.
    a_shape = a._shape_tuple()  # pylint: disable=protected-access
    b_shape = b._shape_tuple()  # pylint: disable=protected-access

    output_may_have_non_empty_batch_shape = (
        (a_shape is None or len(a_shape) > 2) or
        (b_shape is None or len(b_shape) > 2))

    # TODO(b/178749687): remove this boolean and all related branches once the
    # bridges are ready.
    # batch_matmul_v3 is for when input type is different from output type.
    use_batch_matmul_v3 = False
    if output_type and (output_type != a.dtype or output_type != b.dtype):
      use_batch_matmul_v3 = True

    if (not a_is_sparse and
        not b_is_sparse) and output_may_have_non_empty_batch_shape:
      # BatchMatmul does not support transpose, so we conjugate the matrix and
      # use adjoint instead. Conj() is a noop for real matrices.
      if transpose_a:
        a = conj(a)
        adjoint_a = True
      if transpose_b:
        b = conj(b)
        adjoint_b = True
      if use_batch_matmul_v3:
        return gen_math_ops.batch_mat_mul_v3(
            a, b, adj_x=adjoint_a, adj_y=adjoint_b, Tout=output_type, name=name)
      else:
        return gen_math_ops.batch_mat_mul_v2(
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
    if (((a.dtype == dtypes.bfloat16 and
          b.dtype not in (dtypes.int8, dtypes.uint8)) or
         (b.dtype == dtypes.bfloat16 and
          a.dtype not in (dtypes.int8, dtypes.uint8))) and a.dtype != b.dtype):
      # matmul currently doesn't handle mixed-precision inputs other than
      # fp16 * int8 which is supported in BatchMatMulV3.
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
      if use_batch_matmul_v3:
        adjoint_a = adjoint_a or transpose_a
        adjoint_b = adjoint_b or transpose_b
        return gen_math_ops.batch_mat_mul_v3(
            a, b, adj_x=adjoint_a, adj_y=adjoint_b, Tout=output_type, name=name)
      else:
        return gen_math_ops.mat_mul(
            a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)


@tf_export("linalg.matvec")
@dispatch.add_dispatch_support
def matvec(a,
           b,
           transpose_a=False,
           adjoint_a=False,
           a_is_sparse=False,
           b_is_sparse=False,
           name=None):
  """Multiplies matrix `a` by vector `b`, producing `a` * `b`.

  The matrix `a` must, following any transpositions, be a tensor of rank >= 2,
  with `shape(a)[-1] == shape(b)[-1]`, and `shape(a)[:-2]` able to broadcast
  with `shape(b)[:-1]`.

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
  c = tf.linalg.matvec(a, b)


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
  c = tf.linalg.matvec(a, b)
  ```

  Args:
    a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
      `complex128` and rank > 1.
    b: `Tensor` with same type as `a` and compatible dimensions.
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


# TODO(b/178650720): Also support numpy-style type promotion in freestanding TF
#   functions (e.g. tf.add).
def matmul_wrapper(a, b, name=None):  # pylint: disable=missing-function-docstring
  if ops._numpy_style_type_promotion:
    return a._matmul(b)
  return matmul(a, b, name=name)
matmul_wrapper.__doc__ = matmul.__doc__
_OverrideBinaryOperatorHelper(matmul_wrapper, "matmul")

sparse_matmul = deprecation.deprecated(None, "Use `tf.linalg.matmul` instead")(
    gen_math_ops.sparse_mat_mul)
tf_export(v1=["sparse_matmul"])(sparse_matmul)
@dispatch.add_dispatch_support


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
  if not isinstance(x, (ops.Tensor, indexed_slices.IndexedSlices)):
    raise TypeError(f"Not a Tensor or IndexedSlices: {type(x)}.")
  if isinstance(x, indexed_slices.IndexedSlices):
    return x
  x_shape = array_ops.shape_internal(x, optimize=optimize)
  return indexed_slices.IndexedSlices(x, range(0, x_shape[0]), x_shape)


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
    raise TypeError(f"Expected a list or tuple, not {type(inputs)}.")
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
          indexed_slices.IndexedSlices(o.values, cast(o.indices, dtypes.int64),
                                       o.dense_shape))
    else:
      casted_outputs.append(o)
  return casted_outputs


@tf_export("math.add", "add")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def add(x, y, name=None):
  """Returns x + y element-wise.

  Example usages below.

  Add a scalar and a list:

  >>> x = [1, 2, 3, 4, 5]
  >>> y = 1
  >>> tf.add(x, y)
  <tf.Tensor: shape=(5,), dtype=int32, numpy=array([2, 3, 4, 5, 6],
  dtype=int32)>

  Note that binary `+` operator can be used instead:

  >>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
  >>> y = tf.convert_to_tensor(1)
  >>> x + y
  <tf.Tensor: shape=(5,), dtype=int32, numpy=array([2, 3, 4, 5, 6],
  dtype=int32)>

  Add a tensor and a list of same shape:

  >>> x = [1, 2, 3, 4, 5]
  >>> y = tf.constant([1, 2, 3, 4, 5])
  >>> tf.add(x, y)
  <tf.Tensor: shape=(5,), dtype=int32,
  numpy=array([ 2,  4,  6,  8, 10], dtype=int32)>

  **Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
  non-tensor, the non-tensor input will adopt (or get casted to) the data type
  of the tensor input. This can potentially cause unwanted overflow or underflow
  conversion.

  For example,

  >>> x = tf.constant([1, 2], dtype=tf.int8)
  >>> y = [2**7 + 1, 2**7 + 2]
  >>> tf.add(x, y)
  <tf.Tensor: shape=(2,), dtype=int8, numpy=array([-126, -124], dtype=int8)>

  When adding two input values of different shapes, `Add` follows NumPy
  broadcasting rules. The two input array shapes are compared element-wise.
  Starting with the trailing dimensions, the two dimensions either have to be
  equal or one of them needs to be `1`.

  For example,

  >>> x = np.ones(6).reshape(1, 2, 1, 3)
  >>> y = np.ones(6).reshape(2, 1, 3, 1)
  >>> tf.add(x, y).shape.as_list()
  [2, 2, 3, 3]

  Another example with two arrays of different dimension.

  >>> x = np.ones([1, 2, 1, 4])
  >>> y = np.ones([3, 4])
  >>> tf.add(x, y).shape.as_list()
  [1, 2, 3, 4]

  The reduction version of this elementwise operation is `tf.math.reduce_sum`

  Args:
    x: A `tf.Tensor`. Must be one of the following types: bfloat16, half,
      float16, float32, float64, uint8, uint16, uint32, uint64, int8, int16,
      int32, int64, complex64, complex128, string.
    y: A `tf.Tensor`. Must have the same type as x.
    name: A name for the operation (optional)
  """
  with ops.name_scope(name, "Add", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, dtype_hint=x.dtype.base_dtype, name="y")
    if x.dtype == dtypes.string:
      return gen_math_ops.add(x, y, name=name)
    else:
      return gen_math_ops.add_v2(x, y, name=name)


@tf_export("math.add_n", "add_n")
@dispatch.add_dispatch_support(iterable_parameters=["inputs"])
def add_n(inputs, name=None):
  """Returns the element-wise sum of a list of tensors.

  All inputs in the list must have the same shape. This op does not
  [broadcast](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
  its inputs. If you need broadcasting, use `tf.math.add` (or the `+` operator)
  instead.

  For example:

  >>> a = tf.constant([[3, 5], [4, 8]])
  >>> b = tf.constant([[1, 6], [2, 9]])
  >>> tf.math.add_n([a, b, a]).numpy()
  array([[ 7, 16],
         [10, 25]], dtype=int32)

  See Also:

  * `tf.reduce_sum(inputs, axis=0)` - This performs the same mathematical
    operation, but `tf.add_n` may be more efficient because it sums the
    tensors directly. `reduce_sum` on the other hand calls
    `tf.convert_to_tensor` on the list of tensors, unnecessarily stacking them
    into a single tensor before summing.

  Args:
    inputs: A list of `tf.Tensor` or `tf.IndexedSlices` objects, each with the
      same shape and type. `tf.IndexedSlices` objects will be converted into
      dense tensors prior to adding.
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of the same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """
  if not inputs or not isinstance(inputs, collections_abc.Iterable):
    raise ValueError("Inputs must be an iterable of at least one "
                     "Tensor/IndexedSlices with the same dtype and shape.")
  inputs = indexed_slices.convert_n_to_tensor_or_indexed_slices(inputs)
  if not all(
      isinstance(x, (ops.Tensor, indexed_slices.IndexedSlices))
      for x in inputs):
    raise ValueError("Inputs must be an iterable of at least one "
                     "Tensor/IndexedSlices with the same dtype and shape.")

  if len(inputs) == 1:
    if isinstance(inputs[0], indexed_slices.IndexedSlices):
      values = ops.convert_to_tensor(inputs[0])
    else:
      values = inputs[0]
    if name:
      return array_ops.identity(values, name=name)
    return values
  return gen_math_ops.add_n(inputs, name=name)


@tf_export("math.accumulate_n", v1=["math.accumulate_n", "accumulate_n"])
@dispatch.add_dispatch_support
@deprecation.deprecated(None, "Use `tf.math.add_n` Instead")
def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
  """Returns the element-wise sum of a list of tensors.

  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
  otherwise, these are inferred.

  For example:

  >>> a = tf.constant([[1, 2], [3, 4]])
  >>> b = tf.constant([[5, 0], [0, 6]])
  >>> tf.math.accumulate_n([a, b, a]).numpy()
  array([[ 7, 4],
         [ 6, 14]], dtype=int32)

  >>> # Explicitly pass shape and type
  >>> tf.math.accumulate_n(
  ...     [a, b, a], shape=[2, 2], tensor_dtype=tf.int32).numpy()
  array([[ 7,  4],
         [ 6, 14]], dtype=int32)

  Note: The input must be a list or tuple. This function does not handle
  `IndexedSlices`

  See Also:

  * `tf.reduce_sum(inputs, axis=0)` - This performe the same mathematical
    operation, but `tf.add_n` may be more efficient because it sums the
    tensors directly. `reduce_sum` on the other hand calls
    `tf.convert_to_tensor` on the list of tensors, unncessairly stacking them
    into a single tensor before summing.
  * `tf.add_n` - This is another python wrapper for the same Op. It has
    nearly identical functionality.

  Args:
    inputs: A list of `Tensor` objects, each with same shape and type.
    shape: Expected shape of elements of `inputs` (optional). Also controls the
      output shape of this op, which may affect type inference in other ops. A
      value of `None` means "infer the input shape from the shapes in `inputs`".
    tensor_dtype: Expected data type of `inputs` (optional). A value of `None`
      means "infer the input dtype from `inputs[0]`".
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
  inputs = indexed_slices.convert_n_to_tensor_or_indexed_slices(inputs)
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
    raise TypeError(
        f"The `tensor_dtype` argument is {tensor_dtype}, but `input` is of "
        f"type {inputs[0].dtype}. These must be equal. Try casting the input "
        f"to the desired type.")

  if len(inputs) == 1 and name is None:
    return inputs[0]
  elif len(inputs) == 1 and name is not None:
    return array_ops.identity(inputs[0], name=name)
  return add_n(inputs, name=name)


@ops.RegisterGradient("AccumulateNV2")
def _accumulate_n_grad(op, grad):
  """Same as gradient for AddN. Copies the gradient to all inputs."""
  # Not broadcasting.
  return [grad] * len(op.inputs)


@tf_export("math.sigmoid", "nn.sigmoid", "sigmoid")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def sigmoid(x, name=None):
  r"""Computes sigmoid of `x` element-wise.

  Formula for calculating $\mathrm{sigmoid}(x) = y = 1 / (1 + \exp(-x))$.

  For $x \in (-\infty, \infty)$, $\mathrm{sigmoid}(x) \in (0, 1)$.

  Example Usage:

  If a positive number is large, then its sigmoid will approach to 1 since the
  formula will be `y = <large_num> / (1 + <large_num>)`

  >>> x = tf.constant([0.0, 1.0, 50.0, 100.0])
  >>> tf.math.sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32,
  numpy=array([0.5, 0.7310586, 1.0, 1.0], dtype=float32)>

  If a negative number is large, its sigmoid will approach to 0 since the
  formula will be `y = 1 / (1 + <large_num>)`

  >>> x = tf.constant([-100.0, -50.0, -1.0, 0.0])
  >>> tf.math.sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=
  array([0.0000000e+00, 1.9287499e-22, 2.6894143e-01, 0.5],
        dtype=float32)>

  Args:
    x: A Tensor with type `float16`, `float32`, `float64`, `complex64`, or
      `complex128`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x`.

  Usage Example:

  >>> x = tf.constant([-128.0, 0.0, 128.0], dtype=tf.float32)
  >>> tf.sigmoid(x)
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([0. , 0.5, 1. ], dtype=float32)>

  @compatibility(scipy)
  Equivalent to scipy.special.expit
  @end_compatibility
  """
  with ops.name_scope(name, "Sigmoid", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.sigmoid(x, name=name)


@tf_export("math.log_sigmoid", v1=["math.log_sigmoid", "log_sigmoid"])
@dispatch.register_unary_elementwise_api
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

  Usage Example:

  If a positive number is large, then its log_sigmoid will approach to 0 since
  the formula will be `y = log( <large_num> / (1 + <large_num>) )` which
  approximates to `log (1)` which is 0.

  >>> x = tf.constant([0.0, 1.0, 50.0, 100.0])
  >>> tf.math.log_sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=
  array([-6.9314718e-01, -3.1326169e-01, -1.9287499e-22, -0.0000000e+00],
        dtype=float32)>

  If a negative number is large, its log_sigmoid will approach to the number
  itself since the formula will be `y = log( 1 / (1 + <large_num>) )` which is
  `log (1) - log ( (1 + <large_num>) )` which approximates to `- <large_num>`
  that is the number itself.

  >>> x = tf.constant([-100.0, -50.0, -1.0, 0.0])
  >>> tf.math.log_sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=
  array([-100.       ,  -50.       ,   -1.3132616,   -0.6931472],
        dtype=float32)>
  """
  with ops.name_scope(name, "LogSigmoid", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.neg(gen_nn_ops.softplus(-x), name=name)  # pylint: disable=invalid-unary-operand-type


@tf_export("math.cumsum", "cumsum")
@dispatch.add_dispatch_support
def cumsum(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:
  For example:

  >>> # tf.cumsum([a, b, c])   # [a, a + b, a + b + c]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([ 2,  6, 12, 20], dtype=int32)>

  >>> # using varying `axis` values
  >>> y = tf.constant([[2, 4, 6, 8], [1,3,5,7]])
  >>> tf.cumsum(y, axis=0)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[ 2,  4,  6,  8],
         [ 3,  7, 11, 15]], dtype=int32)>
  >>> tf.cumsum(y, axis=1)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[ 2,  6, 12, 20],
         [ 1,  4,  9, 16]], dtype=int32)>

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is performed
  instead:

  >>> # tf.cumsum([a, b, c], exclusive=True)  => [0, a, a + b]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x, exclusive=True)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([ 0,  2,  6, 12], dtype=int32)>

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:

  >>> # tf.cumsum([a, b, c], reverse=True)  # [a + b + c, b + c, c]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x, reverse=True)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([20, 18, 14,  8], dtype=int32)>

  This is more efficient than using separate `tf.reverse` ops.
  The `reverse` and `exclusive` kwargs can also be combined:

  >>> # tf.cumsum([a, b, c], exclusive=True, reverse=True)  # [b + c, c, 0]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x, exclusive=True, reverse=True)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([18, 14,  8,  0], dtype=int32)>

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
@dispatch.add_dispatch_support
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


@tf_export("math.cumulative_logsumexp", v1=["math.cumulative_logsumexp"])
@dispatch.add_dispatch_support
def cumulative_logsumexp(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative log-sum-exp of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumulative log-sum-exp, which means
  that the first element of the input is identical to the first element of
  the output.

  This operation is significantly more numerically stable than the equivalent
  tensorflow operation `tf.math.log(tf.math.cumsum(tf.math.exp(x)))`, although
  computes the same result given infinite numerical precision. However, note
  that in some cases, it may be less stable than `tf.math.reduce_logsumexp`
  for a given element, as it applies the "log-sum-exp trick" in a different
  way.

  More precisely, where `tf.math.reduce_logsumexp` uses the following trick:

  ```
  log(sum(exp(x))) == log(sum(exp(x - max(x)))) + max(x)
  ```

  it cannot be directly used here as there is no fast way of applying it
  to each prefix `x[:i]`. Instead, this function implements a prefix
  scan using pairwise log-add-exp, which is a commutative and associative
  (up to floating point precision) operator:

  ```
  log_add_exp(x, y) = log(exp(x) + exp(y))
                    = log(1 + exp(min(x, y) - max(x, y))) + max(x, y)
  ```

  However, reducing using the above operator leads to a different computation
  tree (logs are taken repeatedly instead of only at the end), and the maximum
  is only computed pairwise instead of over the entire prefix. In general, this
  leads to a different and slightly less precise computation.

  Args:
    x: A `Tensor`. Must be one of the following types: `float16`, `float32`,
      `float64`.
    axis: A `Tensor` of type `int32` or `int64` (default: 0). Must be in the
      range `[-rank(x), rank(x))`.
    exclusive: If `True`, perform exclusive cumulative log-sum-exp.
    reverse: If `True`, performs the cumulative log-sum-exp in the reverse
      direction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same shape and type as `x`.
  """
  with ops.name_scope(name, "CumulativeLogsumexp", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.cumulative_logsumexp(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)


@tf_export("math.conj", v1=["math.conj", "conj"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("conj")
def conj(x, name=None):
  r"""Returns the complex conjugate of a complex number.

  Given a tensor `x` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `x`. The
  complex numbers in `x` must be of the form \\(a + bj\\), where `a` is the
  real part and `b` is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

  >>> x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  >>> tf.math.conj(x)
  <tf.Tensor: shape=(2,), dtype=complex128,
  numpy=array([-2.25-4.75j,  3.25-5.75j])>

  If `x` is real, it is returned unchanged.

  For example:

  >>> x = tf.constant([-2.25, 3.25])
  >>> tf.math.conj(x)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([-2.25,  3.25], dtype=float32)>

  Args:
    x: `Tensor` to conjugate.  Must have numeric or variant type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that is the conjugate of `x` (with the same type).

  Raises:
    TypeError: If `x` is not a numeric tensor.

  @compatibility(numpy)
  Equivalent to numpy.conj.
  @end_compatibility
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
          f"Expected numeric or variant tensor, got dtype {x.dtype!r}.")


def reduced_shape(input_shape, axes):
  """Helper function for reduction ops.

  Args:
    input_shape: 1-D Tensor, the shape of the Tensor being reduced.
    axes: 1-D Tensor, the reduction axes.

  Returns:
    A 1-D Tensor, the output shape as if keepdims were set to True.
  """
  # TODO(allenl): Refactor `reduced_shape` to take the tensor corresponding to
  # `input_shape` rather than `tf.shape` of it. Then we can check if the shape
  # is fully defined here, which may be faster executing eagerly than running
  # `tf.shape` and then fetching its constant value.
  constant_input_shape = tensor_util.constant_value(input_shape)
  if constant_input_shape is not None:
    constant_axes = tensor_util.constant_value(axes)
    if constant_axes is not None:
      constant_axes = np.array(constant_axes, dtype=np.int32)
      constant_input_shape = np.array(constant_input_shape, dtype=np.int32)
      constant_input_shape[constant_axes] = 1
      return constant_input_shape

  # Example:
  # cast needed for SparseTensor reductions
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
          array_ops.ones(axes_shape, dtype=dtypes.int32)
      ])  # [1, 1]


def _unsorted_segment_N(data, segment_ids, num_segments):
  """ Helper function for unsorted_segment_mean/_sqrtN.

  Computes the number
      of segment entries with 0-entries set to 1 to allow division by N.
  """
  num_segments = ops.convert_to_tensor(num_segments)
  # bincount doesn't support negative indices so we use unsorted_segment_sum
  segment_ids_shape = array_ops.shape_internal(segment_ids)
  ones_tensor = array_ops.ones(segment_ids_shape, dtype=data.dtype)
  n = gen_math_ops.unsorted_segment_sum(ones_tensor, segment_ids, num_segments)
  # add dimensions for all non-reduced axes
  broadcastable_shape = array_ops.concat(
      [num_segments[array_ops.newaxis],
       array_ops.ones([array_ops.rank(data)
                       - array_ops.rank(segment_ids)],
                      dtype=num_segments.dtype)],
      axis=0)
  n = array_ops.reshape(n, broadcastable_shape)
  return gen_math_ops.maximum(n, 1)


@tf_export(
    "math.unsorted_segment_mean",
    v1=["math.unsorted_segment_mean", "unsorted_segment_mean"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("unsorted_segment_mean")
def unsorted_segment_mean(data, segment_ids, num_segments, name=None):
  r"""Computes the mean along segments of a tensor.

  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  This operator is similar to the `tf.math.unsorted_segment_sum` operator.
  Instead of computing the sum over segments, it computes the mean of all
  entries belonging to a segment such that:

  \\(output_i = 1/N_i \sum_{j...} data[j...]\\) where the sum is over tuples
  `j...` such that `segment_ids[j...] == i` with \\N_i\\ being the number of
  occurrences of id \\i\\.

  If there is no entry for a given segment ID `i`, it outputs 0.

  If the given segment ID `i` is negative, the value is dropped and will not
  be added to the sum of the segment.

  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.

  Args:
    data: A `Tensor` with floating point or complex dtype.
    segment_ids: An integer tensor whose shape is a prefix of `data.shape`.
      The values must be less than `num_segments`.
      The values are always validated to be in range on CPU,
      never validated on GPU.
    num_segments: An integer scalar `Tensor`.  The number of distinct segment
      IDs.
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
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("unsorted_segment_sqrt_n")
def unsorted_segment_sqrt_n(data, segment_ids, num_segments, name=None):
  r"""Computes the sum along segments of a tensor divided by the sqrt(N).

  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  This operator is similar to the `tf.math.unsorted_segment_sum` operator.
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

  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.

  Args:
    data: A `Tensor` with floating point or complex dtype.
    segment_ids: An integer tensor whose shape is a prefix of `data.shape`.
      The values must be in the range `[0, num_segments)`.
      The values are always validated to be in range on CPU,
      never validated on GPU.
    num_segments: An integer scalar `Tensor`.  The number of distinct segment
      IDs.
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
def sparse_segment_sum(data,
                       indices,
                       segment_ids,
                       name=None,
                       num_segments=None):
  r"""Computes the sum along sparse segments of a tensor.

  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  Like `tf.math.segment_sum`, but `segment_ids` can have rank less than `data`'s
  first dimension, selecting a subset of dimension 0, specified by `indices`.
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
  tf.math.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
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
  r"""Computes the sum along sparse segments of a tensor.

  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  Like `tf.math.segment_sum`, but `segment_ids` can have rank less than `data`'s
  first dimension, selecting a subset of dimension 0, specified by `indices`.
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
  tf.math.segment_sum(c, tf.constant([0, 0, 1]))
  ```

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
  return sparse_segment_sum(
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
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  Like `tf.math.segment_mean`, but `segment_ids` can have rank less than
  `data`'s first dimension, selecting a subset of dimension 0, specified by
  `indices`.
  `segment_ids` is allowed to have missing ids, in which case the output will
  be zeros at those indices. In those cases `num_segments` is used to determine
  the size of the output.

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
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
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  Like `tf.math.segment_mean`, but `segment_ids` can have rank less than
  `data`'s first dimension, selecting a subset of dimension 0, specified by
  `indices`.
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
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
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

  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  Like `tf.sparse.segment_mean`, but instead of dividing by the size of the
  segment, `N`, divide by `sqrt(N)` instead.

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
@dispatch.add_dispatch_support
def tensordot(a, b, axes, name=None):
  r"""Tensor contraction of a and b along specified axes and outer product.

  Tensordot (also known as tensor contraction) sums the product of elements
  from `a` and `b` over the indices specified by `axes`.

  This operation corresponds to `numpy.tensordot(a, b, axes)`.

  Example 1: When `a` and `b` are matrices (order 2), the case `axes=1`
  is equivalent to matrix multiplication.

  Example 2: When `a` and `b` are matrices (order 2), the case
  `axes = [[1], [0]]` is equivalent to matrix multiplication.

  Example 3: When `a` and `b` are matrices (order 2), the case `axes=0` gives
  the outer product, a tensor of order 4.

  Example 4: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
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
      `b` must be equal. If `axes=0`, computes the outer product between `a` and
      `b`.
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
      free = [i for i in builtins.range(len(shape_a)) if i not in axes]
      free_dims = [shape_a[i] for i in free]
      prod_free = int(np.prod([shape_a[i] for i in free]))
      prod_axes = int(np.prod([shape_a[i] for i in axes]))
      perm = list(axes) + free if flipped else free + list(axes)
      new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
      if (perm != np.arange(len(shape_a))).any():
        a_trans = array_ops.transpose(a, perm)
      else:
        a_trans = a
      if a_trans.get_shape().as_list() != new_shape:
        reshaped_a = array_ops.reshape(a_trans, new_shape)
      else:
        reshaped_a = a_trans
      return reshaped_a, free_dims, free_dims
    else:
      if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
        shape_a = a.get_shape().as_list()
        axes = [i if i >= 0 else i + len(shape_a) for i in axes]
        free = [i for i in builtins.range(len(shape_a)) if i not in axes]
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
        free, _ = gen_array_ops.list_diff(range(rank_a), axes, dtypes.int32)
      free_dims = array_ops.gather(shape_a, free)
      axes_dims = array_ops.gather(shape_a, axes)
      prod_free_dims = reduce_prod(free_dims)
      prod_axes_dims = reduce_prod(axes_dims)
      if flipped:
        perm = array_ops.concat([axes, free], 0)
        new_shape = array_ops_stack.stack([prod_axes_dims, prod_free_dims])
      else:
        perm = array_ops.concat([free, axes], 0)
        new_shape = array_ops_stack.stack([prod_free_dims, prod_axes_dims])
      reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
      return reshaped_a, free_dims, free_dims_static

  def _tensordot_axes(a, axes):
    """Generates two sets of contraction axes for the two tensor arguments."""
    a_shape = a.get_shape()
    if isinstance(axes, compat.integral_types):
      if axes < 0:
        raise ValueError(f"`axes` must be at least 0. Received: {axes}.")
      if a_shape.ndims is not None:
        if axes > a_shape.ndims:
          raise ValueError(f"`axes` must not be larger than the number of "
                           f"dimensions of tensor {a}.  Received {axes}, vs "
                           f"tensor dimensions {a_shape.ndims}.")
        return (list(builtins.range(a_shape.ndims - axes,
                                    a_shape.ndims)), list(builtins.range(axes)))
      else:
        rank = array_ops.rank(a)
        return (range(rank - axes, rank,
                      dtype=dtypes.int32), range(axes, dtype=dtypes.int32))
    elif isinstance(axes, (list, tuple)):
      if len(axes) != 2:
        raise ValueError(
            f"`axes` must be an integer or have length 2. Received {axes}.")
      a_axes = axes[0]
      b_axes = axes[1]
      if isinstance(a_axes, compat.integral_types) and \
          isinstance(b_axes, compat.integral_types):
        a_axes = [a_axes]
        b_axes = [b_axes]
      if len(a_axes) != len(b_axes):
        raise ValueError(f"Different number of contraction axes `a` and `b`, "
                         f"{len(a_axes)} != {len(b_axes)}.")
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
      if (ab_matmul.get_shape().is_fully_defined() and
          ab_matmul.get_shape().as_list() == a_free_dims + b_free_dims):
        return ab_matmul
      else:
        return array_ops.reshape(
            ab_matmul, a_free_dims + b_free_dims, name=name)
    else:
      a_free_dims = ops.convert_to_tensor(a_free_dims, dtype=dtypes.int32)
      b_free_dims = ops.convert_to_tensor(b_free_dims, dtype=dtypes.int32)
      product = array_ops.reshape(
          ab_matmul, array_ops.concat([a_free_dims, b_free_dims], 0), name=name)
      if a_free_dims_static is not None and b_free_dims_static is not None:
        product.set_shape(a_free_dims_static + b_free_dims_static)
      return product


@tf_export("math.polyval")
@dispatch.add_dispatch_support
def polyval(coeffs, x, name=None):
  r"""Computes the elementwise value of a polynomial.

  If `x` is a tensor and `coeffs` is a list n + 1 tensors,
  this function returns the value of the n-th order polynomial

  `p(x) = coeffs[n-1] + coeffs[n-2] * x + ...  + coeffs[0] * x**(n-1)`

  evaluated using Horner's method, i.e.

  ```python
  p(x) = coeffs[n-1] + x * (coeffs[n-2] + ... + x * (coeffs[1] + x * coeffs[0]))
  ```

  Usage Example:

  >>> coefficients = [1.0, 2.5, -4.2]
  >>> x = 5.0
  >>> y = tf.math.polyval(coefficients, x)
  >>> y
  <tf.Tensor: shape=(), dtype=float32, numpy=33.3>

  Usage Example:

  >>> tf.math.polyval([2, 1, 0], 3) # evaluates 2 * (3**2) + 1 * (3**1) + 0 * (3**0)
  <tf.Tensor: shape=(), dtype=int32, numpy=21>

  `tf.math.polyval` can also be used in polynomial regression. Taking
  advantage of this function can facilitate writing a polynomial equation
  as compared to explicitly writing it out, especially for higher degree
  polynomials.

  >>> x = tf.constant(3)
  >>> theta1 = tf.Variable(2)
  >>> theta2 = tf.Variable(1)
  >>> theta3 = tf.Variable(0)
  >>> tf.math.polyval([theta1, theta2, theta3], x)
  <tf.Tensor: shape=(), dtype=int32, numpy=21>

  Args:
    coeffs: A list of `Tensor` representing the coefficients of the polynomial.
    x: A `Tensor` representing the variable of the polynomial.
    name: A name for the operation (optional).

  Returns:
    A `tensor` of the shape as the expression p(x) with usual broadcasting
    rules for element-wise addition and multiplication applied.

  @compatibility(numpy)
  Equivalent to numpy.polyval.
  @end_compatibility
  """
  if not isinstance(coeffs, list):
    raise ValueError(
        f"Argument coeffs must be list type. Received type {type(coeffs)}.")

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


@tf_export("math.reciprocal_no_nan")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def reciprocal_no_nan(x, name=None):
  """Performs a safe reciprocal operation, element wise.

  If a particular element is zero, the reciprocal for that element is
  also set to zero.

  For example:
  ```python
  x = tf.constant([2.0, 0.5, 0, 1], dtype=tf.float32)
  tf.math.reciprocal_no_nan(x)  # [ 0.5, 2, 0.0, 1.0 ]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64` `complex64` or
      `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.

  Raises:
    TypeError: x must be of a valid dtype.

  """

  with ops.name_scope(name, "reciprocal_no_nan", [x]) as scope:
    x = ops.convert_to_tensor(x, name="x")
    one = constant_op.constant(1, dtype=x.dtype.base_dtype, name="one")
    return gen_math_ops.div_no_nan(one, x, name=scope)


@tf_export("math.xdivy")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def xdivy(x, y, name=None):
  """Computes `x / y`.

  Given `x` and `y`, computes `x / y`. This function safely returns
  zero when `x = 0`, no matter what the value of `y` is.

  Example:

  >>> tf.math.xdivy(1., 2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.5>
  >>> tf.math.xdivy(0., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.math.xdivy(0., 0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.math.xdivy(1., 0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=inf>

  Args:
    x: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    y: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    name: A name for the operation (optional).

  Returns:
    `x / y`.
  """
  with ops.name_scope(name, "xdivy", [x]):
    return gen_math_ops.xdivy(x, y)


@tf_export("math.xlog1py")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def xlog1py(x, y, name=None):
  r"""Compute x * log1p(y).

  Given `x` and `y`, compute `x * log1p(y)`. This function safely returns
  zero when `x = 0`, no matter what the value of `y` is.

  Example:

  >>> tf.math.xlog1py(0., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.>
  >>> tf.math.xlog1py(1., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.6931472>
  >>> tf.math.xlog1py(2., 2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=2.1972246>
  >>> tf.math.xlog1py(0., -1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.>

  Args:
    x: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    y: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    name: A name for the operation (optional).

  Returns:
    `x * log1p(y)`.

  @compatibility(scipy)
  Equivalent to scipy.special.xlog1py
  @end_compatibility
  """
  with ops.name_scope(name, "xlog1py", [x]):
    return gen_math_ops.xlog1py(x, y)


@tf_export("math.erfinv")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def erfinv(x, name=None):
  """Compute inverse error function.

  Given `x`, compute the inverse error function of `x`. This function
  is the inverse of `tf.math.erf`.

  Args:
    x: `Tensor` with type `float` or `double`.
    name: A name for the operation (optional).
  Returns:
    Inverse error function of `x`.
  """
  with ops.name_scope(name, "erfinv", [x]):
    return gen_math_ops.erfinv(x)


@tf_export("math.ndtri")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def ndtri(x, name=None):
  """Compute quantile of Standard Normal.

  Args:
    x: `Tensor` with type `float` or `double`.
    name: A name for the operation (optional).
  Returns:
    Inverse error function of `x`.
  """
  with ops.name_scope(name, "ndtri", [x]):
    return gen_math_ops.ndtri(x)


@tf_export("math.erfcinv")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def erfcinv(x, name=None):
  """Computes the inverse of complementary error function.

  Given `x`, compute the inverse complementary error function of `x`.
  This function is the inverse of `tf.math.erfc`, and is defined on
  `[0, 2]`.

  >>> tf.math.erfcinv([0., 0.2, 1., 1.5, 2.])
  <tf.Tensor: shape=(5,), dtype=float32, numpy=
  array([       inf,  0.9061935, -0.       , -0.4769363,       -inf],
        dtype=float32)>

  Args:
    x: `Tensor` with type `float` or `double`.
    name: A name for the operation (optional).
  Returns:
    Inverse complementary error function of `x`.

  @compatibility(numpy)
  Equivalent to scipy.special.erfcinv
  @end_compatibility
  """
  with ops.name_scope(name, "erfcinv", [x]):
    x = ops.convert_to_tensor(x, name="start")
    return -ndtri(0.5 * x) * np.sqrt(0.5)


@tf_export("math.ceil", v1=["math.ceil", "ceil"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("ceil")
def ceil(x, name=None):
  """Return the ceiling of the input, element-wise.

  For example:

  >>> tf.math.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
  <tf.Tensor: shape=(7,), dtype=float32,
  numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32)>

  Args:
    x: A `tf.Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`. `int32`
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor`. Has the same type as `x`.

  @compatibility(numpy)
  Equivalent to np.ceil
  @end_compatibility
  """
  return gen_math_ops.ceil(x, name)


@tf_export("math.sqrt", "sqrt")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def sqrt(x, name=None):  # pylint: disable=redefined-builtin
  r"""Computes element-wise square root of the input tensor.

  Note: This operation does not support integer types.

  >>> x = tf.constant([[4.0], [16.0]])
  >>> tf.sqrt(x)
  <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[2.],
           [4.]], dtype=float32)>
  >>> y = tf.constant([[-4.0], [16.0]])
  >>> tf.sqrt(y)
  <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[nan],
           [ 4.]], dtype=float32)>
  >>> z = tf.constant([[-1.0], [16.0]], dtype=tf.complex128)
  >>> tf.sqrt(z)
  <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
    array([[0.0+1.j],
           [4.0+0.j]])>

  Note: In order to support complex type, please provide an input tensor
  of `complex64` or `complex128`.

  Args:
    x: A `tf.Tensor` of type `bfloat16`, `half`, `float32`, `float64`,
      `complex64`, `complex128`
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` of same size, type and sparsity as `x`.
  """
  return gen_math_ops.sqrt(x, name)


# pylint: disable=g-docstring-has-escape
@tf_export("math.exp", "exp")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def exp(x, name=None):
  r"""Computes exponential of x element-wise.  \\(y = e^x\\).

  This function computes the exponential of the input tensor element-wise.
  i.e. `math.exp(x)` or \\(e^x\\), where `x` is the input tensor.
  \\(e\\) denotes Euler's number and is approximately equal to 2.718281.
  Output is positive for any real input.

  >>> x = tf.constant(2.0)
  >>> tf.math.exp(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=7.389056>

  >>> x = tf.constant([2.0, 8.0])
  >>> tf.math.exp(x)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([   7.389056, 2980.958   ], dtype=float32)>

  For complex numbers, the exponential value is calculated as
  $$
  e^{x+iy} = {e^x} {e^{iy}} = {e^x} ({\cos (y) + i \sin (y)})
  $$

  For `1+1j` the value would be computed as:
  $$
  e^1 (\cos (1) + i \sin (1)) = 2.7182817 \times (0.5403023+0.84147096j)
  $$

  >>> x = tf.constant(1 + 1j)
  >>> tf.math.exp(x)
  <tf.Tensor: shape=(), dtype=complex128,
  numpy=(1.4686939399158851+2.2873552871788423j)>

  Args:
    x: A `tf.Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor`. Has the same type as `x`.

  @compatibility(numpy)
  Equivalent to np.exp
  @end_compatibility
  """
  return gen_math_ops.exp(x, name)


# pylint: enable=g-docstring-has-escape


@tf_export("math.sobol_sample")
@dispatch.add_dispatch_support
def sobol_sample(dim, num_results, skip=0, dtype=dtypes.float32, name=None):
  """Generates points from the Sobol sequence.

  Creates a Sobol sequence with `num_results` samples. Each sample has dimension
  `dim`. Skips the first `skip` samples.

  Args:
    dim: Positive scalar `Tensor` representing each sample's dimension.
    num_results: Positive scalar `Tensor` of dtype int32. The number of Sobol
        points to return in the output.
    skip: (Optional) Positive scalar `Tensor` of dtype int32. The number of
        initial points of the Sobol sequence to skip. Default value is 0.
    dtype: (Optional) The `tf.Dtype` of the sample. One of: `tf.float32` or
        `tf.float64`. Defaults to `tf.float32`.
    name: (Optional) Python `str` name prefixed to ops created by this function.

  Returns:
    `Tensor` of samples from Sobol sequence with `shape` [num_results, dim].
  """
  with ops.name_scope(name, "sobol", [dim, num_results, skip]):
    return gen_math_ops.sobol_sample(dim, num_results, skip, dtype=dtype)


@tf_export("math.rsqrt", v1=["math.rsqrt", "rsqrt"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("rsqrt")
def rsqrt(x, name=None):
  """Computes reciprocal of square root of x element-wise.

  For example:

  >>> x = tf.constant([2., 0., -2.])
  >>> tf.math.rsqrt(x)
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([0.707, inf, nan], dtype=float32)>

  Args:
    x: A `tf.Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor`. Has the same type as `x`.
  """
  return gen_math_ops.rsqrt(x, name)


@tf_export("math.acos", "acos")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def acos(x, name=None):
  """Computes acos of x element-wise.

  Provided an input tensor, the `tf.math.acos` operation
  returns the inverse cosine of each element of the tensor.
  If `y = tf.math.cos(x)` then, `x = tf.math.acos(y)`.

  Input range is `[-1, 1]` and the output has a range of `[0, pi]`.

  For example:

  >>> x = tf.constant([1.0, -0.5, 3.4, 0.2, 0.0, -2], dtype = tf.float32)
  >>> tf.math.acos(x)
  <tf.Tensor: shape=(6,), dtype=float32,
  numpy= array([0. , 2.0943952, nan, 1.3694383, 1.5707964, nan],
  dtype=float32)>

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as x.
  """
  return gen_math_ops.acos(x, name)


@tf_export("math.floor", "floor")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def floor(x, name=None):
  """Returns element-wise largest integer not greater than x.

  Both input range is `(-inf, inf)` and the
  output range consists of all integer values.

  For example:

  >>> x = tf.constant([1.3324, -1.5, 5.555, -2.532, 0.99, float("inf")])
  >>> tf.floor(x).numpy()
  array([ 1., -2.,  5., -3.,  0., inf], dtype=float32)

  Args:
    x:  A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as x.
  """
  return gen_math_ops.floor(x, name)


# Register elementwise ops that don't have Python wrappers.
# Binary elementwise ops.
dispatch.register_binary_elementwise_api(gen_bitwise_ops.bitwise_and)
dispatch.register_binary_elementwise_api(gen_bitwise_ops.bitwise_or)
dispatch.register_binary_elementwise_api(gen_bitwise_ops.bitwise_xor)
dispatch.register_binary_elementwise_api(gen_bitwise_ops.left_shift)
dispatch.register_binary_elementwise_api(gen_bitwise_ops.right_shift)
dispatch.register_unary_elementwise_api(gen_bitwise_ops.invert)
dispatch.register_binary_elementwise_api(gen_math_ops.atan2)
dispatch.register_binary_elementwise_api(gen_math_ops.floor_div)
dispatch.register_binary_elementwise_api(gen_math_ops.floor_mod)
dispatch.register_binary_elementwise_api(gen_math_ops.greater)
dispatch.register_binary_elementwise_api(gen_math_ops.greater_equal)
dispatch.register_binary_elementwise_api(gen_math_ops.less)
dispatch.register_binary_elementwise_api(gen_math_ops.less_equal)
dispatch.register_binary_elementwise_api(gen_math_ops.logical_and)
dispatch.register_binary_elementwise_api(gen_math_ops.logical_or)
dispatch.register_binary_elementwise_api(gen_math_ops.maximum)
dispatch.register_binary_elementwise_api(gen_math_ops.minimum)
dispatch.register_binary_elementwise_api(gen_math_ops.real_div)
dispatch.register_binary_elementwise_api(gen_math_ops.squared_difference)
dispatch.register_binary_elementwise_api(gen_math_ops.truncate_div)
dispatch.register_binary_elementwise_api(gen_math_ops.truncate_mod)
dispatch.register_binary_elementwise_api(gen_math_ops.xlogy)
dispatch.register_binary_elementwise_api(gen_math_ops.zeta)

# Unary elementwise ops.
dispatch.register_unary_elementwise_api(gen_math_ops.acosh)
dispatch.register_unary_elementwise_api(gen_math_ops.asin)
dispatch.register_unary_elementwise_api(gen_math_ops.asinh)
dispatch.register_unary_elementwise_api(gen_math_ops.atan)
dispatch.register_unary_elementwise_api(gen_math_ops.atanh)
dispatch.register_unary_elementwise_api(gen_math_ops.cos)
dispatch.register_unary_elementwise_api(gen_math_ops.cosh)
dispatch.register_unary_elementwise_api(gen_math_ops.digamma)
dispatch.register_unary_elementwise_api(gen_math_ops.erf)
dispatch.register_unary_elementwise_api(gen_math_ops.erfc)
dispatch.register_unary_elementwise_api(gen_math_ops.expm1)
dispatch.register_unary_elementwise_api(gen_math_ops.is_finite)
dispatch.register_unary_elementwise_api(gen_math_ops.is_inf)
dispatch.register_unary_elementwise_api(gen_math_ops.is_nan)
dispatch.register_unary_elementwise_api(gen_math_ops.lgamma)
dispatch.register_unary_elementwise_api(gen_math_ops.log)
dispatch.register_unary_elementwise_api(gen_math_ops.log1p)
dispatch.register_unary_elementwise_api(gen_math_ops.logical_not)
dispatch.register_unary_elementwise_api(gen_math_ops.neg)
dispatch.register_unary_elementwise_api(gen_math_ops.next_after)
dispatch.register_unary_elementwise_api(gen_math_ops.reciprocal)
dispatch.register_unary_elementwise_api(gen_math_ops.rint)
dispatch.register_unary_elementwise_api(gen_math_ops.sin)
dispatch.register_unary_elementwise_api(gen_math_ops.sinh)
dispatch.register_unary_elementwise_api(gen_math_ops.square)
dispatch.register_unary_elementwise_api(gen_math_ops.tan)
dispatch.register_unary_elementwise_api(gen_math_ops.tanh)
