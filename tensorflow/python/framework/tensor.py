# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tensor and TensorSpec classes."""

from typing import Type

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


# TODO(b/249802365): Sanitize all TensorSpec names.
def sanitize_spec_name(name: str) -> str:
  """Sanitizes Spec names. Matches Graph Node and Python naming conventions.

  Without sanitization, names that are not legal Python parameter names can be
  set which makes it challenging to represent callables supporting the named
  calling capability.

  Args:
    name: The name to sanitize.

  Returns:
    A string that meets Python parameter conventions.
  """
  if not name:
    return "unknown"

  # Lower case and replace non-alphanumeric chars with '_'
  swapped = "".join([c if c.isalnum() else "_" for c in name.lower()])

  if swapped[0].isalpha():
    return swapped
  else:
    return "tensor_" + swapped


class DenseSpec(type_spec.TypeSpec):
  """Describes a dense object with shape, dtype, and name."""

  __slots__ = ["_shape", "_dtype", "_name"]

  _component_specs = property(lambda self: self)

  def __init__(self, shape, dtype=dtypes.float32, name=None):
    """Creates a TensorSpec.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      name: Optional name for the Tensor.

    Raises:
      TypeError: If shape is not convertible to a `tf.TensorShape`, or dtype is
        not convertible to a `tf.DType`.
    """
    self._shape = tensor_shape.TensorShape(shape)
    self._dtype = dtypes.as_dtype(dtype)
    self._name = name

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self._shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    return self._dtype

  @property
  def name(self):
    """Returns the (optionally provided) name of the described tensor."""
    return self._name

  def is_compatible_with(self, spec_or_value):
    return (isinstance(spec_or_value, (DenseSpec, self.value_type)) and
            self._dtype.is_compatible_with(spec_or_value.dtype) and
            self._shape.is_compatible_with(spec_or_value.shape))

  def __repr__(self):
    return "{}(shape={}, dtype={}, name={})".format(
        type(self).__name__, self.shape, repr(self.dtype), repr(self.name))

  def __hash__(self):
    return hash((self._shape, self.dtype))

  def __eq__(self, other):
    # pylint: disable=protected-access
    return (type(self) is type(other) and self._shape == other._shape and
            self._dtype == other._dtype and self._name == other._name)

  def __ne__(self, other):
    return not self == other

  def _serialize(self):
    return (self._shape, self._dtype, self._name)

  def _to_legacy_output_types(self):
    return self._dtype

  def _to_legacy_output_shapes(self):
    return self._shape

  def _to_legacy_output_classes(self):
    return self.value_type


@tf_export("TensorSpec")
@type_spec_registry.register("tf.TensorSpec")
class TensorSpec(DenseSpec, type_spec.BatchableTypeSpec,
                 trace_type.Serializable, internal.TensorSpec):
  """Describes the type of a tf.Tensor.

  >>> t = tf.constant([[1,2,3],[4,5,6]])
  >>> tf.TensorSpec.from_tensor(t)
  TensorSpec(shape=(2, 3), dtype=tf.int32, name=None)

  Contains metadata for describing the the nature of `tf.Tensor` objects
  accepted or returned by some TensorFlow APIs.

  For example, it can be used to constrain the type of inputs accepted by
  a tf.function:

  >>> @tf.function(input_signature=[tf.TensorSpec([1, None])])
  ... def constrained_foo(t):
  ...   print("tracing...")
  ...   return t

  Now the `tf.function` is able to assume that `t` is always of the type
  `tf.TensorSpec([1, None])` which will avoid retracing as well as enforce the
  type restriction on inputs.

  As a result, the following call with tensor of type `tf.TensorSpec([1, 2])`
  triggers a trace and succeeds:
  >>> constrained_foo(tf.constant([[1., 2]])).numpy()
  tracing...
  array([[1., 2.]], dtype=float32)

  The following subsequent call with tensor of type `tf.TensorSpec([1, 4])`
  does not trigger a trace and succeeds:
  >>> constrained_foo(tf.constant([[1., 2, 3, 4]])).numpy()
  array([[1., 2., 3., 4.], dtype=float32)

  But the following call with tensor of type `tf.TensorSpec([2, 2])` fails:
  >>> constrained_foo(tf.constant([[1., 2], [3, 4]])).numpy()
  Traceback (most recent call last):
  ...
  TypeError: Binding inputs to tf.function `constrained_foo` failed ...

  """

  __slots__ = []

  @classmethod
  def experimental_type_proto(cls) -> Type[struct_pb2.TensorSpecProto]:
    """Returns the type of proto associated with TensorSpec serialization."""
    return struct_pb2.TensorSpecProto

  @classmethod
  def experimental_from_proto(
      cls, proto: struct_pb2.TensorSpecProto) -> "TensorSpec":
    """Returns a TensorSpec instance based on the serialized proto."""
    return TensorSpec(
        shape=tensor_shape.TensorShape.experimental_from_proto(proto.shape),
        dtype=proto.dtype,
        name=proto.name if proto.name else None)

  def experimental_as_proto(self) -> struct_pb2.TensorSpecProto:
    """Returns a proto representation of the TensorSpec instance."""
    return struct_pb2.TensorSpecProto(
        shape=self.shape.experimental_as_proto(),
        dtype=self.dtype.experimental_as_proto().datatype,
        name=self.name)

  def is_compatible_with(self, spec_or_tensor):  # pylint:disable=useless-super-delegation,arguments-renamed
    """Returns True if spec_or_tensor is compatible with this TensorSpec.

    Two tensors are considered compatible if they have the same dtype
    and their shapes are compatible (see `tf.TensorShape.is_compatible_with`).

    Args:
      spec_or_tensor: A tf.TensorSpec or a tf.Tensor

    Returns:
      True if spec_or_tensor is compatible with self.
    """
    return super(TensorSpec, self).is_compatible_with(spec_or_tensor)

  def is_subtype_of(self, other):
    if not isinstance(other, TensorSpec):
      return False

    return (
        (not self.name or self.name == other.name)
        and self.shape.is_subtype_of(other.shape)
        and self.dtype.is_subtype_of(other.dtype)
    )

  def placeholder_value(self, placeholder_context):
    """Generates a graph_placholder with the given TensorSpec information."""
    if placeholder_context.unnest_only:
      return self

    name = self.name or placeholder_context.naming_scope
    context_graph = placeholder_context.context_graph
    if placeholder_context.with_none_control_dependencies:
      # Note: setting ops.control_dependencies(None) ensures we always put
      # capturing placeholders outside of any control flow context.
      with ops.control_dependencies(None):
        placeholder = self._graph_placeholder(context_graph, name=name)
    else:
      placeholder = self._graph_placeholder(context_graph, name=name)

    if name is not None:
      # Record the requested/user-specified name in case it's different than
      # the uniquified name, for validation when exporting signatures.
      placeholder.op._set_attr(  # pylint: disable=protected-access
          "_user_specified_name",
          attr_value_pb2.AttrValue(s=compat.as_bytes(name)))
    # TODO(b/263894631): Add an assertion for a TensorSpec of type resource or
    # variant which must have handle data associated with it.
    if ((self.dtype == dtypes.resource or self.dtype == dtypes.variant)
        and placeholder_context.has_handledata(id(self))):
      handle_data = placeholder_context.get_handledata(id(self))
      if (handle_data is not None
          and handle_data.is_set
          and handle_data.shape_and_type):
        handle_data_util.set_handle_data(placeholder, handle_data)

    # Record the composite device as an attribute to the placeholder.
    # This attribute would be propagated into the arg_attr of the FunctionDef.
    # Currently, a packed eager tensor is always placed on a CompositeDevice.
    if placeholder_context.composite_device_name is not None:
      placeholder.op._set_attr(  # pylint: disable=protected-access
          "_composite_device",
          attr_value_pb2.AttrValue(s=compat.as_bytes(
              placeholder_context.composite_device_name)))

    return placeholder

  def _graph_placeholder(self, graph, name=None):
    """Graph-only version of tf.compat.v1.placeholder(), for internal use only."""
    dtype = self.dtype.base_dtype
    shape = self.shape
    dtype_value = attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
    if isinstance(shape, (list, tuple)):
      shape = tensor_shape.TensorShape(shape)
    shape = attr_value_pb2.AttrValue(shape=shape.as_proto())
    attrs = {"dtype": dtype_value, "shape": shape}
    try:
      op = graph._create_op_internal(  # pylint: disable=protected-access
          "Placeholder", [], [dtype], input_types=[],
          attrs=attrs, name=name)
    except ValueError as e:
      # TODO(b/262413656) Sometimes parameter names are not valid op names, in
      # which case an unnamed placeholder is created instead. Update this logic
      # to sanitize the name instead of falling back on unnamed placeholders.
      logging.warning(e)
      op = graph._create_op_internal(  # pylint: disable=protected-access
          "Placeholder", [], [dtype], input_types=[], attrs=attrs)
    (result,) = op.outputs
    if op_callbacks.should_invoke_op_callbacks():
      # TODO(b/147670703): Once the special-op creation code paths
      # are unified. Remove this `if` block.
      callback_outputs = op_callbacks.invoke_op_callbacks(
          "Placeholder", tuple(), attrs, tuple(op.outputs),
          op_name=name, graph=graph)
      if callback_outputs is not None:
        (result,) = callback_outputs
    return result

  def _to_tensors(self, value):
    assert isinstance(value, ops.Tensor)
    return [value]

  def _cast(self, value, casting_context):
    """Cast value to a tensor that is a subtype of this TensorSpec."""
    # This method is mainly used to cast Python primitives to tensor.
    # Currently, cast tensor to tensor with different types are not supported.
    # For example, casting int32 to float32 would raise a ValueError.
    if casting_context.allow_specs and isinstance(value, TensorSpec):
      assert value.is_subtype_of(self), f"Can not cast {value!r} to {self!r}"
      return self

    value = ops.convert_to_tensor(value, self.dtype)
    value_spec = TensorSpec(value.shape, value.dtype, self.name)

    if not value_spec.is_subtype_of(self):
      if self.is_subtype_of(value_spec):
        gen_array_ops.ensure_shape(value, self.shape)
      else:
        raise AssertionError(f"Can not cast {value_spec!r} to {self!r}")

    return value

  @classmethod
  def from_spec(cls, spec, name=None):
    """Returns a `TensorSpec` with the same shape and dtype as `spec`.

    >>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="OriginalName")
    >>> tf.TensorSpec.from_spec(spec, "NewName")
    TensorSpec(shape=(8, 3), dtype=tf.int32, name='NewName')

    Args:
      spec: The `TypeSpec` used to create the new `TensorSpec`.
      name: The name for the new `TensorSpec`.  Defaults to `spec.name`.
    """
    return cls(spec.shape, spec.dtype, name or spec.name)

  @classmethod
  def from_tensor(cls, tensor, name=None):
    """Returns a `TensorSpec` that describes `tensor`.

    >>> tf.TensorSpec.from_tensor(tf.constant([1, 2, 3]))
    TensorSpec(shape=(3,), dtype=tf.int32, name=None)

    Args:
      tensor: The `tf.Tensor` that should be described.
      name: A name for the `TensorSpec`.  Defaults to `tensor.op.name`.

    Returns:
      A `TensorSpec` that describes `tensor`.
    """
    if isinstance(tensor, ops.EagerTensor):
      return TensorSpec(tensor.shape, tensor.dtype, name)
    elif isinstance(tensor, ops.Tensor):
      # TODO(b/249802365): Return a sanitized version of op name or no name.
      return TensorSpec(tensor.shape, tensor.dtype, name or tensor.op.name)
    else:
      raise ValueError(
          f"`tensor` should be a tf.Tensor, but got type {type(tensor)}.")

  @property
  def value_type(self):
    """The Python type for values that are compatible with this TypeSpec."""
    return ops.Tensor

  def _to_components(self, value):
    assert isinstance(value, core_tf_types.Tensor)
    return value

  def _from_components(self, components):
    return components

  def _from_compatible_tensor_list(self, tensor_list):
    # TODO(b/112266545): It would be cleaner to create a new `ensure_shape()`
    # op here and return that, instead of mutating the input's shape using
    # `Tensor.set_shape()`. However, that would add extra ops, which could
    # impact performance. When this bug is resolved, we should be able to add
    # the `ensure_shape()` ops and optimize them away using contextual shape
    # information.
    assert len(tensor_list) == 1
    tensor_list[0].set_shape(self._shape)
    return tensor_list[0]

  def _to_batchable_tensor_list(self, value, batched=False):
    if batched and self._shape.merge_with(value.shape).ndims == 0:
      raise ValueError("Unbatching a tensor is only supported for rank >= 1")
    return self._to_components(value)

  def _batch(self, batch_size):
    return TensorSpec(
        tensor_shape.TensorShape([batch_size]).concatenate(self._shape),
        self._dtype)

  def _unbatch(self):
    if self._shape.ndims == 0:
      raise ValueError("Unbatching a tensor is only supported for rank >= 1")
    return TensorSpec(self._shape[1:], self._dtype)

  @property
  def _flat_tensor_specs(self):
    return [self]

  def _to_tensor_list(self, value):
    return [self._to_components(value)]

  def _to_batched_tensor_list(self, value):
    return self._to_tensor_list(value)

  # TODO(b/206014848): Helper function to support logic that does not consider
  # Tensor name. Will be removed once load-bearing usages of Tensor name are
  # fixed.
  def _without_tensor_names(self) -> "TensorSpec":
    """Returns a version of `TensorSpec` with the name removed."""
    if self.name is None:
      return self
    else:
      return TensorSpec(self.shape, self.dtype)

trace_type.register_serializable(TensorSpec)
trace_type.register_tensor_type(TensorSpec)


class _TensorCodec:
  """Codec for Tensor."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, ops.Tensor)

  def do_encode(self, tensor_value, encode_fn):
    """Returns an encoded `TensorProto` for the given `tf.Tensor`."""
    del encode_fn
    encoded_tensor = struct_pb2.StructuredValue()
    if isinstance(tensor_value, ops.EagerTensor):
      encoded_tensor.tensor_value.CopyFrom(
          tensor_util.make_tensor_proto(tensor_value.numpy())
      )
    else:
      if tensor_value.op.type == "Const":
        encoded_tensor.tensor_value.CopyFrom(tensor_value.op.get_attr("value"))
      else:
        raise nested_structure_coder.NotEncodableError(
            f"No encoder for object {str(tensor_value)} of type"
            f" {type(tensor_value)}."
        )
    return encoded_tensor

  def can_decode(self, value):
    return value.HasField("tensor_value")

  def do_decode(self, value, decode_fn):
    """Returns the `tf.Tensor` encoded by the proto `value`."""
    del decode_fn
    tensor_proto = value.tensor_value
    tensor = constant_op.constant(tensor_util.MakeNdarray(tensor_proto))
    return tensor


nested_structure_coder.register_codec(_TensorCodec())


class _TensorSpecCodec:
  """Codec for `TensorSpec`."""

  def can_encode(self, pyobj):
    # BoundedTensorSpec has its own decoder.
    return (isinstance(pyobj, TensorSpec) and
            not isinstance(pyobj, BoundedTensorSpec))

  def do_encode(self, tensor_spec_value, encode_fn):
    encoded_tensor_spec = struct_pb2.StructuredValue()
    encoded_tensor_spec.tensor_spec_value.CopyFrom(
        struct_pb2.TensorSpecProto(
            shape=encode_fn(tensor_spec_value.shape).tensor_shape_value,
            dtype=encode_fn(tensor_spec_value.dtype).tensor_dtype_value,
            name=tensor_spec_value.name))
    return encoded_tensor_spec

  def can_decode(self, value):
    return value.HasField("tensor_spec_value")

  def do_decode(self, value, decode_fn):
    name = value.tensor_spec_value.name
    return TensorSpec(
        shape=decode_fn(
            struct_pb2.StructuredValue(
                tensor_shape_value=value.tensor_spec_value.shape)),
        dtype=decode_fn(
            struct_pb2.StructuredValue(
                tensor_dtype_value=value.tensor_spec_value.dtype)),
        name=(name if name else None))


nested_structure_coder.register_codec(_TensorSpecCodec())


# TODO(b/133606651): Should is_compatible_with should check min/max bounds?
@type_spec_registry.register("tf.BoundedTensorSpec")
class BoundedTensorSpec(TensorSpec, trace_type.Serializable):
  """A `TensorSpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
  tf_minimum = tf.convert_to_tensor(spec.minimum, dtype=spec.dtype)
  tf_maximum = tf.convert_to_tensor(spec.maximum, dtype=spec.dtype)
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by tensors
  with values in the set {0, 1, 2}:
  ```python
  spec = tensor_spec.BoundedTensorSpec((3, 5), tf.int32, 0, 2)
  ```
  """

  __slots__ = ("_minimum", "_maximum")

  def __init__(self, shape, dtype, minimum, maximum, name=None):
    """Initializes a new `BoundedTensorSpec`.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      minimum: Number or sequence specifying the minimum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If `minimum` or `maximum` are not provided or not
        broadcastable to `shape`.
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    super(BoundedTensorSpec, self).__init__(shape, dtype, name)

    if minimum is None:
      raise ValueError("`minimum` can not be None.")
    if maximum is None:
      raise ValueError("`maximum` can not be None.")

    try:
      minimum_shape = np.shape(minimum)
      common_shapes.broadcast_shape(
          tensor_shape.TensorShape(minimum_shape), self.shape)
    except ValueError as exception:
      raise ValueError(
          f"`minimum` {minimum} is not compatible with shape {self.shape}."
      ) from exception

    try:
      maximum_shape = np.shape(maximum)
      common_shapes.broadcast_shape(
          tensor_shape.TensorShape(maximum_shape), self.shape)
    except ValueError as exception:
      raise ValueError(
          f"`maximum` {maximum} is not compatible with shape {self.shape}."
      ) from exception

    self._minimum = np.array(minimum, dtype=self.dtype.as_numpy_dtype)
    self._minimum.setflags(write=False)

    self._maximum = np.array(maximum, dtype=self.dtype.as_numpy_dtype)
    self._maximum.setflags(write=False)

  @classmethod
  def experimental_type_proto(cls) -> Type[struct_pb2.BoundedTensorSpecProto]:
    """Returns the type of proto associated with BoundedTensorSpec serialization."""
    return struct_pb2.BoundedTensorSpecProto

  @classmethod
  def experimental_from_proto(
      cls, proto: struct_pb2.BoundedTensorSpecProto) -> "BoundedTensorSpec":
    """Returns a BoundedTensorSpec instance based on the serialized proto."""
    return BoundedTensorSpec(
        shape=tensor_shape.TensorShape.experimental_from_proto(proto.shape),
        dtype=proto.dtype,
        minimum=tensor_util.MakeNdarray(proto.minimum),
        maximum=tensor_util.MakeNdarray(proto.maximum),
        name=proto.name if proto.name else None)

  def experimental_as_proto(self) -> struct_pb2.BoundedTensorSpecProto:
    """Returns a proto representation of the BoundedTensorSpec instance."""
    return struct_pb2.BoundedTensorSpecProto(
        shape=self.shape.experimental_as_proto(),
        dtype=self.dtype.experimental_as_proto().datatype,
        minimum=tensor_util.make_tensor_proto(self._minimum),
        maximum=tensor_util.make_tensor_proto(self._maximum),
        name=self.name)

  @classmethod
  def from_spec(cls, spec):
    """Returns a `TensorSpec` with the same shape and dtype as `spec`.

    If `spec` is a `BoundedTensorSpec`, then the new spec's bounds are set to
    `spec.minimum` and `spec.maximum`; otherwise, the bounds are set to
    `spec.dtype.min` and `spec.dtype.max`.

    >>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="x")
    >>> BoundedTensorSpec.from_spec(spec)
    BoundedTensorSpec(shape=(8, 3), dtype=tf.int32, name='x',
        minimum=array(-2147483648, dtype=int32),
        maximum=array(2147483647, dtype=int32))

    Args:
      spec: The `TypeSpec` used to create the new `BoundedTensorSpec`.
    """
    dtype = dtypes.as_dtype(spec.dtype)
    minimum = getattr(spec, "minimum", dtype.min)
    maximum = getattr(spec, "maximum", dtype.max)
    return BoundedTensorSpec(spec.shape, dtype, minimum, maximum, spec.name)

  @property
  def minimum(self):
    """Returns a NumPy array specifying the minimum bounds (inclusive)."""
    return self._minimum

  @property
  def maximum(self):
    """Returns a NumPy array specifying the maximum bounds (inclusive)."""
    return self._maximum

  def _cast(self, value, casting_context):
    if casting_context.allow_specs and isinstance(value, BoundedTensorSpec):
      assert value.is_subtype_of(self), f"Can not cast {value!r} to {self!r}"
      return self

    actual_spec = TensorSpec(shape=self.shape, dtype=self.dtype, name=self.name)
    return actual_spec._cast(value, casting_context)  # pylint: disable=protected-access

  def __repr__(self):
    s = "BoundedTensorSpec(shape={}, dtype={}, name={}, minimum={}, maximum={})"
    return s.format(self.shape, repr(self.dtype), repr(self.name),
                    repr(self.minimum), repr(self.maximum))

  def __eq__(self, other):
    tensor_spec_eq = super(BoundedTensorSpec, self).__eq__(other)
    return (tensor_spec_eq and np.allclose(self.minimum, other.minimum) and
            np.allclose(self.maximum, other.maximum))

  def __hash__(self):
    return hash((self._shape, self.dtype))

  def __reduce__(self):
    return BoundedTensorSpec, (self._shape, self._dtype, self._minimum,
                               self._maximum, self._name)

  def _serialize(self):
    return (self._shape, self._dtype, self._minimum, self._maximum, self._name)


class _BoundedTensorSpecCodec:
  """Codec for `BoundedTensorSpec`."""

  def can_encode(self, pyobj):
    return isinstance(pyobj, BoundedTensorSpec)

  def do_encode(self, bounded_tensor_spec_value, encode_fn):
    """Returns an encoded proto for the given `tf.BoundedTensorSpec`."""
    encoded_bounded_tensor_spec = struct_pb2.StructuredValue()
    encoded_bounded_tensor_spec.bounded_tensor_spec_value.CopyFrom(
        struct_pb2.BoundedTensorSpecProto(
            shape=encode_fn(bounded_tensor_spec_value.shape).tensor_shape_value,
            dtype=encode_fn(bounded_tensor_spec_value.dtype).tensor_dtype_value,
            name=bounded_tensor_spec_value.name,
            minimum=tensor_util.make_tensor_proto(
                bounded_tensor_spec_value.minimum),
            maximum=tensor_util.make_tensor_proto(
                bounded_tensor_spec_value.maximum)))
    return encoded_bounded_tensor_spec

  def can_decode(self, value):
    return value.HasField("bounded_tensor_spec_value")

  def do_decode(self, value, decode_fn):
    btsv = value.bounded_tensor_spec_value
    name = btsv.name
    return BoundedTensorSpec(
        shape=decode_fn(
            struct_pb2.StructuredValue(tensor_shape_value=btsv.shape)),
        dtype=decode_fn(
            struct_pb2.StructuredValue(tensor_dtype_value=btsv.dtype)),
        minimum=tensor_util.MakeNdarray(btsv.minimum),
        maximum=tensor_util.MakeNdarray(btsv.maximum),
        name=(name if name else None))


nested_structure_coder.register_codec(_BoundedTensorSpecCodec())

trace_type.register_serializable(BoundedTensorSpec)
_pywrap_utils.RegisterType("TensorSpec", TensorSpec)

# Note: we do not include Tensor names when constructing TypeSpecs.
type_spec.register_type_spec_from_value_converter(
    ops.Tensor, lambda tensor: TensorSpec(tensor.shape, tensor.dtype))

type_spec.register_type_spec_from_value_converter(
    np.ndarray, lambda array: TensorSpec(array.shape, array.dtype))
