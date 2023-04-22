# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Indexed slices."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.types import internal
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export


# Use LazyLoader to avoid circular dependencies.
#
# Note: these can all be changed to regular imports once all code has been
# updated to refer the symbols defined in this module directly, rather than
# using the backwards-compatible aliases in ops.py.  (E.g.,
# "indexed_slices.IndexedSlices" rather than "ops.IndexedSlices".)
math_ops = LazyLoader(
    "math_ops", globals(),
    "tensorflow.python.ops.math_ops")
ops = LazyLoader(
    "ops", globals(), "tensorflow.python.framework.ops")
tensor_spec = LazyLoader(
    "tensor_spec", globals(),
    "tensorflow.python.framework.tensor_spec")
tensor_util = LazyLoader(
    "tensor_util", globals(),
    "tensorflow.python.framework.tensor_util")


# TODO(mdan): Should IndexedSlices be a "tensor"?
@tf_export("IndexedSlices")
class IndexedSlices(internal.NativeObject, composite_tensor.CompositeTensor):
  """A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. `tf.gather`).

  >>> v = tf.Variable([[0.,1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]])
  >>> with tf.GradientTape() as tape:
  ...   r = tf.gather(v, [1,3])
  >>> index_slices = tape.gradient(r,v)
  >>> index_slices
  <...IndexedSlices object ...>
  >>> index_slices.indices.numpy()
  array([1, 3], dtype=int32)
  >>> index_slices.values.numpy()
  array([[1., 1., 1.],
         [1., 1., 1.]], dtype=float32)

  Contrast this representation with
  `tf.sparse.SparseTensor`,
  which uses multi-dimensional indices and scalar values.
  """

  def __init__(self, values, indices, dense_shape=None):
    """Creates an `IndexedSlices`."""
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape

  @property
  def values(self):
    """A `Tensor` containing the values of the slices."""
    return self._values

  @property
  def indices(self):
    """A 1-D `Tensor` containing the indices of the slices."""
    return self._indices

  @property
  def dense_shape(self):
    """A 1-D `Tensor` containing the shape of the corresponding dense tensor."""
    return self._dense_shape

  @property
  def shape(self):
    """Gets the `tf.TensorShape` representing the shape of the dense tensor.

    Returns:
      A `tf.TensorShape` object.
    """
    if self._dense_shape is None:
      return tensor_shape.TensorShape(None)

    return tensor_util.constant_value_as_shape(self._dense_shape)

  @property
  def name(self):
    """The name of this `IndexedSlices`."""
    return self.values.name

  @property
  def device(self):
    """The name of the device on which `values` will be produced, or `None`."""
    return self.values.device

  @property
  def op(self):
    """The `Operation` that produces `values` as an output."""
    return self.values.op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self.values.dtype

  @property
  def graph(self):
    """The `Graph` that contains the values, indices, and shape tensors."""
    return self._values.graph

  def __str__(self):
    return "IndexedSlices(indices=%s, values=%s%s)" % (
        self._indices, self._values,
        (", dense_shape=%s" %
         (self._dense_shape,)) if self._dense_shape is not None else "")

  def __neg__(self):
    return IndexedSlices(-self.values, self.indices, self.dense_shape)

  @property
  def _type_spec(self):
    indices_shape = self._indices.shape.merge_with(self._values.shape[:1])
    dense_shape = tensor_shape.TensorShape([None]).concatenate(
        self._values.shape[1:])
    if self._dense_shape is not None:
      dense_shape_dtype = self._dense_shape.dtype
      dense_shape = dense_shape.merge_with(
          tensor_util.constant_value_as_shape(self._dense_shape))
    else:
      dense_shape_dtype = None
    return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype,
                             dense_shape_dtype, indices_shape)

  def _shape_invariant_to_type_spec(self, shape):
    # From tf.while_loop docs: "If a loop variable is an IndexedSlices, the
    # shape invariant must be a shape invariant of the values tensor of the
    # IndexedSlices. It means the shapes of the three tensors of the
    # IndexedSlices are (shape, [shape[0]], [shape.ndims])."
    indices_shape = shape[:1]
    dense_shape = tensor_shape.TensorShape([None]).concatenate(shape[1:])
    if self._dense_shape is None:
      dense_shape_dtype = None
    else:
      dense_shape_dtype = self._dense_shape.dtype
    return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype,
                             dense_shape_dtype, indices_shape)

  def consumers(self):
    return self._consumers()


IndexedSlicesValue = collections.namedtuple(
    "IndexedSlicesValue", ["values", "indices", "dense_shape"])


@tf_export("IndexedSlicesSpec")
class IndexedSlicesSpec(type_spec.TypeSpec):
  """Type specification for a `tf.IndexedSlices`."""

  __slots__ = ["_shape", "_values_dtype", "_indices_dtype",
               "_dense_shape_dtype", "_indices_shape"]

  value_type = property(lambda self: IndexedSlices)

  def __init__(self, shape=None, dtype=dtypes.float32,
               indices_dtype=dtypes.int64, dense_shape_dtype=None,
               indices_shape=None):
    """Constructs a type specification for a `tf.IndexedSlices`.

    Args:
      shape: The dense shape of the `IndexedSlices`, or `None` to allow any
        dense shape.
      dtype: `tf.DType` of values in the `IndexedSlices`.
      indices_dtype: `tf.DType` of the `indices` in the `IndexedSlices`.  One
        of `tf.int32` or `tf.int64`.
      dense_shape_dtype: `tf.DType` of the `dense_shape` in the `IndexedSlices`.
        One of `tf.int32`, `tf.int64`, or `None` (if the `IndexedSlices` has
        no `dense_shape` tensor).
      indices_shape: The shape of the `indices` component, which indicates
        how many slices are in the `IndexedSlices`.
    """
    self._shape = tensor_shape.as_shape(shape)
    self._values_dtype = dtypes.as_dtype(dtype)
    self._indices_dtype = dtypes.as_dtype(indices_dtype)
    if dense_shape_dtype is None:
      self._dense_shape_dtype = None
    else:
      self._dense_shape_dtype = dtypes.as_dtype(dense_shape_dtype)
    self._indices_shape = tensor_shape.as_shape(indices_shape).with_rank(1)

  def _serialize(self):
    return (self._shape, self._values_dtype, self._indices_dtype,
            self._dense_shape_dtype, self._indices_shape)

  @property
  def _component_specs(self):
    value_shape = self._indices_shape.concatenate(self._shape[1:])
    specs = [
        tensor_spec.TensorSpec(value_shape, self._values_dtype),
        tensor_spec.TensorSpec(self._indices_shape, self._indices_dtype)]
    if self._dense_shape_dtype is not None:
      specs.append(
          tensor_spec.TensorSpec([self._shape.ndims], self._dense_shape_dtype))
    return tuple(specs)

  def _to_components(self, value):
    if value.dense_shape is None:
      return (value.values, value.indices)
    else:
      return (value.values, value.indices, value.dense_shape)

  def _from_components(self, tensor_list):
    if (all(isinstance(t, np.ndarray) for t in tensor_list) and
        not tf2.enabled()):
      if len(tensor_list) == 2:
        return IndexedSlicesValue(tensor_list[0], tensor_list[1], None)
      else:
        return IndexedSlicesValue(*tensor_list)
    else:
      return IndexedSlices(*tensor_list)


@tf_export(v1=["convert_to_tensor_or_indexed_slices"])
def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None):
  """Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` or `SparseTensor` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.

  Returns:
    A `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  return internal_convert_to_tensor_or_indexed_slices(
      value=value, dtype=dtype, name=name, as_ref=False)


def internal_convert_to_tensor_or_indexed_slices(value,
                                                 dtype=None,
                                                 name=None,
                                                 as_ref=False):
  """Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` or `SparseTensor` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  if isinstance(value, ops.EagerTensor) and not context.executing_eagerly():
    return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)
  # TODO(mdan): Name says tensor_or_indexed_slices. So do explicitly just that?
  elif isinstance(value, internal.NativeObject):
    if dtype and not dtypes.as_dtype(dtype).is_compatible_with(value.dtype):
      raise ValueError(
          "Tensor conversion requested dtype %s for Tensor with dtype %s: %r" %
          (dtypes.as_dtype(dtype).name, value.dtype.name, str(value)))
    return value
  else:
    return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)


def internal_convert_n_to_tensor_or_indexed_slices(values,
                                                   dtype=None,
                                                   name=None,
                                                   as_ref=False):
  """Converts `values` to a list of `Tensor` or `IndexedSlices` objects.

  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned
  unmodified.

  Args:
    values: An iterable of `None`, `IndexedSlices`, `SparseTensor`, or objects
      that can be consumed by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A list of `Tensor`, `IndexedSlices`, `SparseTensor` and/or `None` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections_abc.Iterable):
    raise TypeError("values must be iterable.")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(
          internal_convert_to_tensor_or_indexed_slices(
              value, dtype=dtype, name=n, as_ref=as_ref))
  return ret


def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None):
  """Converts `values` to a list of `Output` or `IndexedSlices` objects.

  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned
  unmodified.

  Args:
    values: A list of `None`, `IndexedSlices`, `SparseTensor`, or objects that
      can be consumed by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor`
      `IndexedSlices`.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.

  Returns:
    A list of `Tensor`, `IndexedSlices`, and/or `SparseTensor` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  return internal_convert_n_to_tensor_or_indexed_slices(
      values=values, dtype=dtype, name=name, as_ref=False)


# Warn the user if we convert a sparse representation to dense with at
# least this number of elements.
_LARGE_SPARSE_NUM_ELEMENTS = 100000000


def _indexed_slices_to_tensor(value, dtype=None, name=None, as_ref=False):
  """Converts an IndexedSlices object `value` to a Tensor.

  NOTE(mrry): This function is potentially expensive.

  Args:
    value: An ops.IndexedSlices object.
    dtype: The dtype of the Tensor to be returned.
    name: Optional name to use for the returned Tensor.
    as_ref: True if a ref is requested.

  Returns:
    A dense Tensor representing the values in the given IndexedSlices.

  Raises:
    ValueError: If the IndexedSlices does not have the same dtype.
  """
  _ = as_ref
  if dtype and not dtype.is_compatible_with(value.dtype):
    raise ValueError(
        "Tensor conversion requested dtype %s for IndexedSlices with dtype %s" %
        (dtype.name, value.dtype.name))
  if value.dense_shape is None:
    raise ValueError(
        "Tensor conversion requested for IndexedSlices without dense_shape: %s"
        % str(value))
  # TODO(mrry): Consider adding static shape information to
  # IndexedSlices, to avoid using numpy here.
  if not context.executing_eagerly():
    dense_shape_value = tensor_util.constant_value(value.dense_shape)
    if dense_shape_value is not None:
      num_elements = np.prod(dense_shape_value)
      if num_elements >= _LARGE_SPARSE_NUM_ELEMENTS:
        warnings.warn(
            "Converting sparse IndexedSlices to a dense Tensor with %d "
            "elements. This may consume a large amount of memory." %
            num_elements)
    else:
      if value.dense_shape.op.type != "VariableShape":
        # VariableShape may hide static shapes behind a resource handle
        # producing a warning that isn't that useful to users.
        warnings.warn(
            "Converting sparse IndexedSlices(%s) to a dense Tensor of unknown "
            "shape. This may consume a large amount of memory." % value)
  return math_ops.unsorted_segment_sum(
      value.values, value.indices, value.dense_shape[0], name=name)


tensor_conversion_registry.register_tensor_conversion_function(
    IndexedSlices, _indexed_slices_to_tensor)
