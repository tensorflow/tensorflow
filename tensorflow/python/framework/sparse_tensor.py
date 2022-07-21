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
"""Sparse tensors."""
# pylint: disable=g-bad-name
import collections

import numpy as np

from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export

# pylint: disable=protected-access
_eval_using_default_session = ops._eval_using_default_session
_override_helper = ops._override_helper
# pylint: enable=protected-access


@tf_export("sparse.SparseTensor", "SparseTensor")
class SparseTensor(internal.NativeObject, composite_tensor.CompositeTensor):
  """Represents a sparse tensor.

  TensorFlow represents a sparse tensor as three separate dense tensors:
  `indices`, `values`, and `dense_shape`.  In Python, the three tensors are
  collected into a `SparseTensor` class for ease of use.  If you have separate
  `indices`, `values`, and `dense_shape` tensors, wrap them in a `SparseTensor`
  object before passing to the ops below.

  Concretely, the sparse tensor `SparseTensor(indices, values, dense_shape)`
  comprises the following components, where `N` and `ndims` are the number
  of values and number of dimensions in the `SparseTensor`, respectively:

  * `indices`: A 2-D int64 tensor of shape `[N, ndims]`, which specifies the
    indices of the elements in the sparse tensor that contain nonzero values
    (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]` specifies
    that the elements with indexes of [1,3] and [2,4] have nonzero values.

  * `values`: A 1-D tensor of any type and shape `[N]`, which supplies the
    values for each element in `indices`. For example, given `indices=[[1,3],
    [2,4]]`, the parameter `values=[18, 3.6]` specifies that element [1,3] of
    the sparse tensor has a value of 18, and element [2,4] of the tensor has a
    value of 3.6.

  * `dense_shape`: A 1-D int64 tensor of shape `[ndims]`, which specifies the
    dense_shape of the sparse tensor. Takes a list indicating the number of
    elements in each dimension. For example, `dense_shape=[3,6]` specifies a
    two-dimensional 3x6 tensor, `dense_shape=[2,3,4]` specifies a
    three-dimensional 2x3x4 tensor, and `dense_shape=[9]` specifies a
    one-dimensional tensor with 9 elements.

  The corresponding dense tensor satisfies:

  ```python
  dense.shape = dense_shape
  dense[tuple(indices[i])] = values[i]
  ```

  By convention, `indices` should be sorted in row-major order (or equivalently
  lexicographic order on the tuples `indices[i]`). This is not enforced when
  `SparseTensor` objects are constructed, but most ops assume correct ordering.
  If the ordering of sparse tensor `st` is wrong, a fixed version can be
  obtained by calling `tf.sparse.reorder(st)`.

  Example: The sparse tensor

  ```python
  SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
  ```

  represents the dense tensor

  ```python
  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
  ```
  """

  @classmethod
  def from_value(cls, sparse_tensor_value):
    if not is_sparse(sparse_tensor_value):
      raise TypeError(f"Argument sparse_tensor_value={sparse_tensor_value} "
                      "is neither a SparseTensor nor SparseTensorValue.")
    return SparseTensor(
        indices=sparse_tensor_value.indices,
        values=sparse_tensor_value.values,
        dense_shape=sparse_tensor_value.dense_shape)

  def __init__(self, indices, values, dense_shape):
    """Creates a `SparseTensor`.

    Args:
      indices: A 2-D int64 tensor of shape `[N, ndims]`.
      values: A 1-D tensor of any type and shape `[N]`.
      dense_shape: A 1-D int64 tensor of shape `[ndims]`.

    Raises:
      ValueError: When building an eager SparseTensor if `dense_shape` is
        unknown or contains unknown elements (None or -1).
    """
    with ops.name_scope(None, "SparseTensor", [indices, values, dense_shape]):
      indices = ops.convert_to_tensor(
          indices, name="indices", dtype=dtypes.int64)
      # TODO(touts): Consider adding mutable_values() when 'values'
      # is a VariableOp and updating users of SparseTensor.
      values = ops.convert_to_tensor(values, name="values")

      dense_shape = ops.convert_to_tensor(
          dense_shape, name="dense_shape", dtype=dtypes.int64)
      dense_shape_default = tensor_util.constant_value_as_shape(dense_shape)

    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape
    self._dense_shape_default = dense_shape_default

    indices_shape = indices.shape.with_rank(2)
    values_shape = values.shape.with_rank(1)
    dense_shape_shape = dense_shape.shape.with_rank(1)

    # Assert number of rows in indices match the number of elements in values.
    indices_shape.dims[0].assert_is_compatible_with(values_shape.dims[0])
    # Assert number of columns in indices matches the number of elements in
    # dense_shape.
    indices_shape.dims[1].assert_is_compatible_with(dense_shape_shape.dims[0])

  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    """
    return self._dense_shape_default

  @property
  def indices(self):
    """The indices of non-zero values in the represented dense tensor.

    Returns:
      A 2-D Tensor of int64 with dense_shape `[N, ndims]`, where `N` is the
        number of non-zero values in the tensor, and `ndims` is the rank.
    """
    return self._indices

  @property
  def values(self):
    """The non-zero values in the represented dense tensor.

    Returns:
      A 1-D Tensor of any data type.
    """
    return self._values

  def with_values(self, new_values):
    """Returns a copy of `self` with `values` replaced by `new_values`.

    This method produces a new `SparseTensor` that has the same nonzero
    `indices` and same `dense_shape`, but updated values.

    Args:
      new_values: The values of the new `SparseTensor`. Needs to have the same
        shape as the current `.values` `Tensor`. May have a different type than
        the current `values`.

    Returns:
      A `SparseTensor` with identical indices and shape but updated values.

    Example usage:

    >>> st = tf.sparse.from_dense([[1, 0, 2, 0], [3, 0, 0, 4]])
    >>> tf.sparse.to_dense(st.with_values([10, 20, 30, 40]))  # 4 nonzero values
    <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
    array([[10,  0, 20,  0],
           [30,  0,  0, 40]], dtype=int32)>

    """
    return SparseTensor(self._indices, new_values, self._dense_shape)

  @property
  def op(self):
    """The `Operation` that produces `values` as an output."""
    return self._values.op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._values.dtype

  @property
  def dense_shape(self):
    """A 1-D Tensor of int64 representing the shape of the dense tensor."""
    return self._dense_shape

  @property
  def shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    """
    return self._dense_shape_default

  @property
  def graph(self):
    """The `Graph` that contains the index, value, and dense_shape tensors."""
    return self._indices.graph

  def __str__(self):
    return "SparseTensor(indices=%s, values=%s, dense_shape=%s)" % (
        self._indices, self._values, self._dense_shape)

  def eval(self, feed_dict=None, session=None):
    """Evaluates this sparse tensor in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `SparseTensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this sparse
        tensor. If none, the default session will be used.

    Returns:
      A `SparseTensorValue` object.
    """
    indices, values, dense_shape = _eval_using_default_session(
        [self.indices, self.values, self.dense_shape], feed_dict, self.graph,
        session)
    return SparseTensorValue(indices, values, dense_shape)

  @staticmethod
  def _override_operator(operator, func):
    _override_helper(SparseTensor, operator, func)

  @property
  def _type_spec(self):
    return SparseTensorSpec(self.shape, self.dtype)

  def _shape_invariant_to_type_spec(self, shape):
    # From the tf.while_loop docs: "If a loop variable is a SparseTensor, the
    # shape invariant must be TensorShape([r]) where r is the rank of the dense
    # tensor represented by the sparse tensor. It means the shapes of the three
    # tensors of the SparseTensor are ([None], [None, r], [r]). NOTE: The shape
    # invariant here is the shape of the SparseTensor.dense_shape property. It
    # must be the shape of a vector.
    if shape.ndims is not None and shape.ndims != 1:
      raise ValueError(f"Expected a shape with 1 dimension. Obtained: {shape} "
                       f"which has {shape.ndims} dimensions.")
    rank = tensor_shape.dimension_value(shape[0])
    return SparseTensorSpec(tensor_shape.unknown_shape(rank), self.dtype)

  def consumers(self):
    return self._consumers()


SparseTensorValue = collections.namedtuple("SparseTensorValue",
                                           ["indices", "values", "dense_shape"])
tf_export(v1=["SparseTensorValue"])(SparseTensorValue)
_pywrap_utils.RegisterType("SparseTensorValue", SparseTensorValue)


@tf_export("SparseTensorSpec")
@type_spec.register("tf.SparseTensorSpec")
class SparseTensorSpec(type_spec.BatchableTypeSpec):
  """Type specification for a `tf.sparse.SparseTensor`."""

  __slots__ = ["_shape", "_dtype"]

  value_type = property(lambda self: SparseTensor)

  def __init__(self, shape=None, dtype=dtypes.float32):
    """Constructs a type specification for a `tf.sparse.SparseTensor`.

    Args:
      shape: The dense shape of the `SparseTensor`, or `None` to allow any dense
        shape.
      dtype: `tf.DType` of values in the `SparseTensor`.
    """
    self._shape = tensor_shape.as_shape(shape)
    self._dtype = dtypes.as_dtype(dtype)

  def _serialize(self):
    return (self._shape, self._dtype)

  @property
  def dtype(self):
    """The `tf.dtypes.DType` specified by this type for the SparseTensor."""
    return self._dtype

  @property
  def shape(self):
    """The `tf.TensorShape` specified by this type for the SparseTensor."""
    return self._shape

  @property
  def _component_specs(self):
    rank = self._shape.ndims
    num_values = None
    return [
        tensor_spec.TensorSpec([num_values, rank], dtypes.int64),
        tensor_spec.TensorSpec([num_values], self._dtype),
        tensor_spec.TensorSpec([rank], dtypes.int64)]

  def _to_components(self, value):
    if isinstance(value, SparseTensorValue):
      value = SparseTensor.from_value(value)
    return [value.indices, value.values, value.dense_shape]

  def _from_components(self, tensor_list):
    if (all(isinstance(t, np.ndarray) for t in tensor_list) and
        not tf2.enabled()):
      return SparseTensorValue(*tensor_list)
    else:
      return SparseTensor(*tensor_list)

  # The SparseTensorSpec tensor_list encoding uses (de)serialize_sparse ops
  # to (un)box the component tensors in a way that allows for batching &
  # unbatching.
  @property
  def _flat_tensor_specs(self):
    # NOTE(mrry): The default flat shape of a boxed `SparseTensor` is `(3,)`,
    # but a `SparseTensorSpec` can also represent a batch of boxed
    # `SparseTensor` objects with shape `(..., 3)` (and batches of batches,
    # etc.), so the flat shape must be unknown.
    return [tensor_spec.TensorSpec(None, dtypes.variant)]

  def _to_tensor_list(self, value):
    value = SparseTensor.from_value(value)
    return [gen_sparse_ops.serialize_sparse(
        value.indices, value.values, value.dense_shape,
        out_type=dtypes.variant)]

  def _to_batched_tensor_list(self, value):
    dense_shape = tensor_util.constant_value_as_shape(value.dense_shape)
    if self._shape.merge_with(dense_shape).ndims == 0:
      raise ValueError(
          "Unbatching a sparse tensor is only supported for rank >= 1. "
          f"Obtained input: {value}.")
    return [gen_sparse_ops.serialize_many_sparse(
        value.indices, value.values, value.dense_shape,
        out_type=dtypes.variant)]

  def _from_compatible_tensor_list(self, tensor_list):
    tensor_list = gen_sparse_ops.deserialize_sparse(tensor_list[0], self._dtype)
    indices, values, dense_shape = tensor_list
    rank = self._shape.ndims
    indices.set_shape([None, rank])
    # We restore the dense_shape from the SparseTypeSpec. This is necessary
    # for shape inference when using placeholder SparseTensors in function
    # tracing.
    if self._shape.is_fully_defined():
      dense_shape = ops.convert_to_tensor(
          self._shape, dtype=dtypes.int64, name="shape")
    elif (self._shape.rank is not None and
          any(dim.value is not None for dim in self._shape.dims)):
      # array_ops imports sparse_tensor.py. Local import to avoid import cycle.
      from tensorflow.python.ops import array_ops  # pylint: disable=g-import-not-at-top
      pieces = array_ops.unstack(dense_shape, num=self._shape.rank)
      for i, dim in enumerate(self._shape.dims):
        if dim.value is not None:
          pieces[i] = constant_op.constant(dim.value, dense_shape.dtype)
      dense_shape = array_ops.stack(pieces)
    else:
      dense_shape.set_shape([rank])

    return SparseTensor(indices, values, dense_shape)

  def _batch(self, batch_size):
    return SparseTensorSpec(
        tensor_shape.TensorShape([batch_size]).concatenate(self._shape),
        self._dtype)

  def _unbatch(self):
    if self._shape.ndims == 0:
      raise ValueError("Unbatching a tensor is only supported for rank >= 1")
    return SparseTensorSpec(self._shape[1:], self._dtype)

  def _to_legacy_output_types(self):
    return self._dtype

  def _to_legacy_output_shapes(self):
    return self._shape

  def _to_legacy_output_classes(self):
    return SparseTensor

  @classmethod
  def from_value(cls, value):
    if isinstance(value, SparseTensor):
      return cls(value.shape, value.dtype)
    if isinstance(value, SparseTensorValue):
      if isinstance(value.values, np.ndarray):
        return cls(value.dense_shape, value.values.dtype)
      else:
        return cls.from_value(SparseTensor.from_value(value))
    else:
      raise TypeError("Expected SparseTensor or SparseTensorValue. Received: "
                      f"{value} of type {type(value).__name__}.")


# TODO(b/133606651) Delete the SparseTensor registration when CompositeTensor
# is updated to define a _type_spec field (since registration will be
# automatic).  Do *not* delete the SparseTensorValue registration.
type_spec.register_type_spec_from_value_converter(
    SparseTensor, SparseTensorSpec.from_value)
type_spec.register_type_spec_from_value_converter(
    SparseTensorValue, SparseTensorSpec.from_value)


@tf_export(v1=["convert_to_tensor_or_sparse_tensor"])
def convert_to_tensor_or_sparse_tensor(value, dtype=None, name=None):
  """Converts value to a `SparseTensor` or `Tensor`.

  Args:
    value: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
      registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `SparseTensor` or `Tensor` based on `value`.

  Raises:
    RuntimeError: If result type is incompatible with `dtype`.
  """
  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  if isinstance(value, SparseTensorValue):
    value = SparseTensor.from_value(value)
  if isinstance(value, SparseTensor):
    if dtype and not dtype.is_compatible_with(value.dtype):
      raise RuntimeError(f"Sparse dtype mismatch. Requested: {dtype.name}, "
                         f" Actual: {value.dtype.name}")
    return value
  return ops.convert_to_tensor(value, dtype=dtype, name=name)


def is_sparse(x):
  """Check whether `x` is sparse.

  Check whether an object is a `tf.sparse.SparseTensor` or
  `tf.compat.v1.SparseTensorValue`.

  Args:
    x: A python object to check.

  Returns:
    `True` iff `x` is a `tf.sparse.SparseTensor` or
    `tf.compat.v1.SparseTensorValue`.
  """
  return isinstance(x, (SparseTensor, SparseTensorValue))
