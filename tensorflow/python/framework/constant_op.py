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

See the [constants guide](https://tensorflow.org/api_guides/python/constant_op).
"""

# Must be separate from array_ops to avoid a cyclic dependency.

import contextlib
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.profiler import trace
from tensorflow.python.util.tf_export import tf_export


def _eager_reshape(tensor, shape, ctx):
  """Eager-only version of Reshape op; requires tensor is an eager Tensor."""
  attr_t = tensor._datatype_enum()  # pylint: disable=protected-access
  attr_tshape, (shape,) = execute.args_to_matching_eager(
      [shape], ctx, [dtypes.int32, dtypes.int64], dtypes.int32)
  inputs_flat = [tensor, shape]
  attrs = ("T", attr_t, "Tshape", attr_tshape)
  [result] = execute.execute(
      b"Reshape", 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
  return result


def _eager_fill(dims, value, ctx):
  """Eager-only version of Fill op; requires value is an eager Tensor."""
  attr_t = value.dtype.as_datatype_enum
  dims = convert_to_eager_tensor(dims, ctx, dtypes.int32)
  inputs_flat = [dims, value]
  attrs = ("T", attr_t, "index_type", types_pb2.DT_INT32)
  [result] = execute.execute(
      b"Fill", 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
  return result


def _eager_identity(tensor, ctx):
  """Eager-only version of Identity op; requires tensor is an eager Tensor."""
  attrs = ("T", tensor.dtype.as_datatype_enum)
  [result] = execute.execute(
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
      raise TypeError(f"Expected tensor {value} with dtype {dtype!r}, but got "
                      f"dtype {value.dtype!r}.")
    return value
  if dtype is not None:
    try:
      dtype = dtype.as_datatype_enum
    except AttributeError:
      dtype = dtypes.as_dtype(dtype).as_datatype_enum
  ctx.ensure_initialized()
  return ops.EagerTensor(value, ctx.device_name, dtype)


@tf_export(v1=["constant"])
def constant_v1(
    value, dtype=None, shape=None, name="Const", verify_shape=False):
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

  `tf.constant` differs from `tf.fill` in a few ways:

  *   `tf.constant` supports arbitrary constants, not just uniform scalar
      Tensors like `tf.fill`.
  *   `tf.constant` creates a `Const` node in the computation graph with the
      exact value at graph construction time. On the other hand, `tf.fill`
      creates an Op in the graph that is expanded at runtime.
  *   Because `tf.constant` only embeds constant values in the graph, it does
      not support dynamic shapes based on other runtime Tensors, whereas
      `tf.fill` does.

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
  return _constant_impl(value, dtype, shape, name, verify_shape=verify_shape,
                        allow_broadcast=False)


@tf_export("constant", v1=[])
def constant(value, dtype=None, shape=None, name="Const"):
  """Creates a constant tensor from a tensor-like object.

  Note: All eager `tf.Tensor` values are immutable (in contrast to
  `tf.Variable`). There is nothing especially _constant_ about the value
  returned from `tf.constant`. This function is not fundamentally different from
  `tf.convert_to_tensor`. The name `tf.constant` comes from the `value` being
  embedded in a `Const` node in the `tf.Graph`. `tf.constant` is useful
  for asserting that the value can be embedded that way.

  If the argument `dtype` is not specified, then the type is inferred from
  the type of `value`.

  >>> # Constant 1-D Tensor from a python list.
  >>> tf.constant([1, 2, 3, 4, 5, 6])
  <tf.Tensor: shape=(6,), dtype=int32,
      numpy=array([1, 2, 3, 4, 5, 6], dtype=int32)>
  >>> # Or a numpy array
  >>> a = np.array([[1, 2, 3], [4, 5, 6]])
  >>> tf.constant(a)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    array([[1, 2, 3],
           [4, 5, 6]])>

  If `dtype` is specified, the resulting tensor values are cast to the requested
  `dtype`.

  >>> tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float64)
  <tf.Tensor: shape=(6,), dtype=float64,
      numpy=array([1., 2., 3., 4., 5., 6.])>

  If `shape` is set, the `value` is reshaped to match. Scalars are expanded to
  fill the `shape`:

  >>> tf.constant(0, shape=(2, 3))
    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)>
  >>> tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)>

  `tf.constant` has no effect if an eager Tensor is passed as the `value`, it
  even transmits gradients:

  >>> v = tf.Variable([0.0])
  >>> with tf.GradientTape() as g:
  ...     loss = tf.constant(v + v)
  >>> g.gradient(loss, v).numpy()
  array([2.], dtype=float32)

  But, since `tf.constant` embeds the value in the `tf.Graph` this fails for
  symbolic tensors:

  >>> with tf.compat.v1.Graph().as_default():
  ...   i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)
  ...   t = tf.constant(i)
  Traceback (most recent call last):
  ...
  TypeError: ...

  `tf.constant` will create tensors on the current device. Inputs which are
  already tensors maintain their placements unchanged.

  Related Ops:

  * `tf.convert_to_tensor` is similar but:
    * It has no `shape` argument.
    * Symbolic tensors are allowed to pass through.

    >>> with tf.compat.v1.Graph().as_default():
    ...   i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)
    ...   t = tf.convert_to_tensor(i)

  * `tf.fill`: differs in a few ways:
    *   `tf.constant` supports arbitrary constants, not just uniform scalar
        Tensors like `tf.fill`.
    *   `tf.fill` creates an Op in the graph that is expanded at runtime, so it
        can efficiently represent large tensors.
    *   Since `tf.fill` does not embed the value, it can produce dynamically
        sized outputs.

  Args:
    value: A constant value (or list) of output type `dtype`.
    dtype: The type of the elements of the resulting tensor.
    shape: Optional dimensions of resulting tensor.
    name: Optional name for the tensor.

  Returns:
    A Constant Tensor.

  Raises:
    TypeError: if shape is incorrectly specified or unsupported.
    ValueError: if called on a symbolic tensor.
  """
  return _constant_impl(value, dtype, shape, name, verify_shape=False,
                        allow_broadcast=True)


def maybe_convert_tensor_to_weak_type(tensor, weak_type):
  """Converts the Tensor inplace, to a weakly-typed one if needed.

  Args:
    tensor: A Tensor instance in either Eager or Graph mode.
    weak_type: Boolean type, whether `tensor` needs to be converted.
  """
  if weak_type and not dtypes.is_weak_type(tensor.dtype):
    tensor._weak_dtype = dtypes.type_to_weak_type[tensor.dtype]  # pylint: disable=protected-access


def _constant_impl(
    value, dtype, shape, name, verify_shape, allow_broadcast):
  """Implementation of constant."""
  is_weak_type = dtypes.is_weak_type(dtype)

  ctx = context.context()
  if ctx.executing_eagerly():
    with contextlib.ExitStack() as stack:
      if trace.enabled:
        stack.enter_context(trace.Trace("tf.constant"))
      res = _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
      maybe_convert_tensor_to_weak_type(res, is_weak_type)
      return res

  const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access
      value, dtype, shape, name, verify_shape, allow_broadcast
  )
  maybe_convert_tensor_to_weak_type(const_tensor, is_weak_type)
  return const_tensor


def _constant_eager_impl(ctx, value, dtype, shape, verify_shape):
  """Creates a constant on the current device."""
  t = convert_to_eager_tensor(value, ctx, dtype)
  if shape is None:
    return t
  shape = tensor_shape.as_shape(shape)
  if shape == t.shape:
    return t
  if verify_shape:
    raise TypeError(f"Expected Tensor {t} (converted from {value}) with shape "
                    f"{tuple(shape)}, but got shape {tuple(t.shape)}.")
  num_t = t.shape.num_elements()
  # TODO(josh11b): Implement shape -> eager tensor conversion.
  if num_t == shape.num_elements():
    return _eager_reshape(t, shape.as_list(), ctx)
  if num_t == 1:
    if t.dtype == dtypes.bool:
      # We don't have a Fill kernel for bool dtype on GPU. So we first run
      # Fill on CPU and then copy to GPU if needed.
      with ops.device("/device:CPU:0"):
        x = _eager_fill(shape.as_list(), _eager_identity(t, ctx), ctx)
      return _eager_identity(x, ctx)
    else:
      return _eager_fill(shape.as_list(), t, ctx)
  raise TypeError("Eager execution of tf.constant with unsupported shape. "
                  f"Tensor {t} (converted from {value}) has {num_t:d} "
                  f"elements, but got `shape` {shape} with "
                  f"{shape.num_elements()} elements).")


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


tensor_conversion_registry.register_tensor_conversion_function(
    (list, tuple), _constant_tensor_conversion_function, 100)
tensor_conversion_registry.register_tensor_conversion_function(
    object, _constant_tensor_conversion_function, 200)


def _tensor_shape_tensor_conversion_function(s,
                                             dtype=None,
                                             name=None,
                                             as_ref=False):
  """Function to convert TensorShape to Tensor."""
  _ = as_ref
  if not s.is_fully_defined():
    raise ValueError(
        f"Cannot convert a partially known TensorShape {s} to a Tensor.")
  s_list = s.as_list()
  int64_value = 0
  for dim in s_list:
    if dim >= 2**31:
      int64_value = dim
      break

  if dtype is not None:
    if dtype not in (dtypes.int32, dtypes.int64):
      raise TypeError(f"Cannot convert TensorShape {s} to dtype {dtype}. "
                      "Allowed dtypes are tf.int32 and tf.int64.")
    if dtype == dtypes.int32 and int64_value:
      raise ValueError(f"Cannot convert TensorShape {s} to dtype int32; "
                       f"a dimension is too large. Consider using tf.int64.")
  else:
    dtype = dtypes.int64 if int64_value else dtypes.int32
  if name is None:
    name = "shape_as_tensor"
  return constant(s_list, dtype=dtype, name=name)


tensor_conversion_registry.register_tensor_conversion_function(
    tensor_shape.TensorShape, _tensor_shape_tensor_conversion_function, 100)


def _dimension_tensor_conversion_function(d,
                                          dtype=None,
                                          name=None,
                                          as_ref=False):
  """Function to convert Dimension to Tensor."""
  _ = as_ref
  if d.value is None:
    raise ValueError(f"Cannot convert unknown Dimension {d} to a Tensor.")
  if dtype is not None:
    if dtype not in (dtypes.int32, dtypes.int64):
      raise TypeError(f"Cannot convert Dimension {d} to dtype {dtype}. "
                      "Allowed dtypes are tf.int32 and tf.int64.")
  else:
    dtype = dtypes.int32
  if name is None:
    name = "shape_as_tensor"
  return constant(d.value, dtype=dtype, name=name)


tensor_conversion_registry.register_tensor_conversion_function(
    tensor_shape.Dimension, _dimension_tensor_conversion_function, 100)
