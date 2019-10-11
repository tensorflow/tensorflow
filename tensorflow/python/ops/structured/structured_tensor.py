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
"""Structured Tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util


class StructuredTensor(composite_tensor.CompositeTensor):
  """A multidimensional collection of structures with the same schema.

  A **`StructuredTensor`** is a multi-dimensional collection of ***structures***
  with the same ***schema***, where:

  * A ***schema*** is a collection of fields, each of which has a name and type.
  * A ***structure*** maps each field in the schema to a tensor value (which
    could be a nested StructuredTensor).

  As an important special case, a 1D `StructuredTensor` encodes a 2D table,
  where columns are heterogeneous `Tensor`s, and rows are the aligned elements
  in each of those `Tensor`s.

  Internally, StructuredTensors use a "field-major" encoding: for each leaf
  field, there is a single tensor that stores the value of that field for all
  structures in the `StructuredTensor`.

  ### Examples

  ```python
  >>> # A scalar StructuredTensor describing a single person.
  >>> s1 = tf.structured.constant({"age": 82, "nicknames": ["Bob", "Bobby"]})
  >>> print s1.shape
  ()
  >>> print s1["age"]
  tf.Tensor(82, shape=(), dtype=int32)

  >>> # A vector StructuredTensor describing three people.
  >>> s2 = stf.struct.constant([
  ...     {"age": 12, "nicknames": ["Josaphine"]},
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]},
  ...     {"age": 82, "nicknames": ["Elmo"]}])
  >>> print s2.shape
  (3,)
  >>> print s2[0]["age"]
  tf.Tensor(12, shape=(), dtype=int32)
  ```

  ### Field Paths

  A *field path* is a tuple of field names, specifying the path to a nested
  field.
  """

  #=============================================================================
  # Constructor & Factory Methods
  #=============================================================================

  # TODO(edloper): Add optional shape validation:
  # Check that the fields all have the same runtime-shape.  (We check static
  # shape now, but that doesn't capture ragged shapes or shapes that aren't
  # statically known.)  I.e., if shape validation is turned on, then check that
  # the outer shape.rank dimensions of each value in fields is the same.  For
  # ragged tensors, this means checking their row-splits.
  def __init__(self, shape, fields):
    """Creates a `StructuredTensor` from a dictionary of fields.

    Args:
      shape: A `TensorShape`: static information about the shape of the
        `StructuredTensor`.  Must have a known `rank`.
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`, providing the values for individual fields in each
        structure.  If `ndims > 0`, then every tensor in `fields` must have the
        same shape in the first `shape.rank` dimensions; and that shape must be
        compatible with `shape`.

    Returns:
      A `StructuredTensor`.
    """
    shape = tensor_shape.as_shape(shape)
    if shape.rank is None:
      raise ValueError("StructuredTensor's shape must have known rank.")
    if not isinstance(fields, dict):
      raise TypeError('fields must be a dictionary, got %s' %
                      type(fields).__name__)
    self._fields = {}
    with ops.name_scope(None, 'StructuredTensor', fields.values()):
      for (key, value) in fields.items():
        if not isinstance(key, str):
          raise TypeError('Unexpected type for key in `fields`: %r' % key)
        if not _FIELD_NAME_RE.match(key):
          raise ValueError('Field name %r is not currently allowed.' % key)
        if not isinstance(
            value, (ops.Tensor, ragged_tensor.RaggedTensor, StructuredTensor)):
          if ragged_tensor.is_ragged(value):
            value = ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
          else:
            try:
              value = ops.convert_to_tensor(value)
            except (ValueError, TypeError):
              raise TypeError('Unexpected type for value in `fields`: %r' %
                              value)
        self._fields[key] = value

    # Check the static TensorShape for this StructuredTensor.
    shape = tensor_shape.as_shape(shape)
    rank = shape.ndims
    if rank is None:
      raise ValueError("StructuredTensor's shape must have known rank.")
    self._static_shape = shape
    if rank > 0:
      for value in self._fields.values():
        self._static_shape = self._static_shape.merge_with(value.shape[:rank])

    self._nested_row_splits = []
    if rank > 1:
      # If any fields are ragged, then check that all row-splits match.
      shared_row_splits = []
      for field in self._fields.values():
        # TODO(edloper): A field shouldn't count as ragged if it has
        # uniform_row_length defined for all the dimensions in question.
        if isinstance(field, ragged_tensor.RaggedTensor):
          shared_row_splits.append(field.nested_row_splits[:rank - 1])
        elif isinstance(field, StructuredTensor) and field.ragged_rank > 0:
          shared_row_splits.append(field.nested_row_splits[:rank - 1])
      if shared_row_splits:
        if len(shared_row_splits) != len(self._fields):
          raise ValueError('Ragged StructuredTensor contains non-ragged fields')

        # Check if the splits are identical.  This should be the common case.
        identical_splits = True
        for splits in shared_row_splits[1:]:
          if len(splits) != len(shared_row_splits[0]):
            raise ValueError('Fields have inconsistent ragged_rank')
          for (s1, s2) in zip(splits, shared_row_splits[0]):
            if s1 is not s2:
              identical_splits = False

        if identical_splits:
          self._nested_row_splits = shared_row_splits[0]
        else:
          # If splits aren't identical, then add assertions to check that they
          # match.
          with ops.control_dependencies(
              ragged_util.assert_splits_match(shared_row_splits)):
            self._nested_row_splits = [array_ops.identity(splits)
                                       for splits in shared_row_splits[0]]

          # TODO(edloper): Rebuild all fields to ensure that they use the
          # identical row_splits tensor.

  @classmethod
  def from_row_splits(cls, values, row_splits, validate=True):
    """Creates a ragged StructuredTensor with rows partitioned by `row_splits`.

    See `tf.RaggedTensor` for information about row_splits.

    Args:
      values: A `StructuredTensor` with shape `[nvals, ...]`.
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be
        empty, and must be sorted in ascending order.  `row_splits[0]` must be
        zero and `row_splits[-1]` must be `nvals`.
      validate: If true, then use assertions to check that the arguments form
        a valid ragged `StructuredTensor`.

    Returns:
      A ragged `StructuredTensor`.  `result.rank = values.rank + 1`.
    """
    if not isinstance(values, StructuredTensor):
      raise TypeError('values must be a StructuredTensor.')
    if values.shape.rank == 0:
      raise ValueError('Shape %s must have rank at least 1' % values.shape)
    row_splits = ops.convert_to_tensor(row_splits, name='row_splits')
    row_splits.shape.assert_has_rank(1)
    if tensor_shape.dimension_value(row_splits.shape[0]) == 0:
      raise ValueError('row_splits may not be empty')
    if row_splits.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError('Row-partitioning tensors must have dtype '
                       'int32 or int64')

    if (row_splits.shape and
        tensor_shape.dimension_value(row_splits.shape[0]) is not None):
      nrows = tensor_shape.dimension_value(row_splits.shape[0]) - 1
    else:
      nrows = None
    result_shape = tensor_shape.TensorShape([nrows, None
                                            ]).concatenate(values.shape[1:])
    result_fields = {}
    for (name, field) in values._fields.items():
      if isinstance(field, StructuredTensor):
        result_fields[name] = StructuredTensor.from_row_splits(
            field, row_splits)
      else:
        result_fields[name] = ragged_tensor.RaggedTensor.from_row_splits(
            field, row_splits, validate=validate)
    return cls(result_shape, result_fields)

  # @TODO(edloper): Add from_row_lengths, etc.

  #=============================================================================
  # Properties
  #=============================================================================

  @property
  def rank(self):
    """The rank of this StructuredTensor.  Guaranteed not to be `None`."""
    return self._static_shape.rank

  @property
  def shape(self):
    """The static shape of this StructuredTensor.

    The returned `TensorShape` is guaranteed to have a known rank, but the
    individual dimension sizes may be unknown.

    Returns:
      `tf.TensorShape`
    """
    return self._static_shape

  @property
  def nested_row_splits(self):
    """A tuple containing the row_splits for all ragged dimensions.

    If non-empty, then every `field` in this StructuredTensor is ragged, and
    has these `nested_row_splits` as their outermost row-splits tensors.

    Returns:
      A `tuple` of 1-D integer `Tensor`s.  The length of this tuple will
      always be less than `self.rank`.
    """
    return self._nested_row_splits

  @property
  def ragged_rank(self):
    """The number of ragged dimensions in this StructuredTensor.

    Returns:
      A Python `int` indicating the number of ragged dimensions in this ragged
      tensor.  The outermost dimension is not considered ragged.
    """
    return len(self._nested_row_splits)

  #=============================================================================
  # Encoding
  #=============================================================================

  def field_names(self):
    """Returns the string field names for this `StructuredTensor`."""
    return tuple(self._fields.keys())

  def field_value(self, field_name):
    """Returns the tensor value for the specified field or path.

    If `field_name` is a `string`, then it names a field directly owned by this
    `StructuredTensor`.  If this `StructuredTensor` has shape `[D1...DN]`, then
    the returned tensor will have shape `[D1...DN, V1...VM]`, where the slice
    `result[d1...dN]`contains the field value for the structure at
    `self[d1...dN]`.

    If `field_name` is a `tuple` of `string`, then it specifies a path to a
    field owned by nested `StructuredTensor`.  In particular,
    `struct.field_value((f1, f2, ..., fN))` is equivalent to
    `struct.field_value(f1).field_value(f2)....field_value(fN)`

    Args:
      field_name: `string` or `tuple` of `string`: The field whose values should
        be returned.

    Returns:
      `Tensor`, `StructuredTensor`, or `RaggedTensor`.
    """
    if isinstance(field_name, (list, tuple)):
      value = self
      for f in field_name:
        value = value.field_value(f)
      return value
    return self._fields[field_name]

  def __repr__(self):
    return 'StructuredTensor(%s, %r)' % (self._static_shape, self._fields)

  #=============================================================================
  # Composite Tensor
  #=============================================================================

  @property
  def _type_spec(self):
    return StructuredTensorSpec.from_value(self)


class StructuredTensorSpec(type_spec.BatchableTypeSpec):
  """Type specification for `StructuredTensor`s."""

  __slots__ = ['_shape', '_field_specs']

  def __init__(self, shape, field_specs):
    """Build a type specification for a StructuredTensor.

    Args:
      shape: The shape of the StructuredTensor.  shape.ndims must not be None.
      field_specs: A dictionary mapping from field name to TypeSpec, specifying
        the tensor type used to encode each field. These TypeSpecs should
        specify the type of the entire field (including outer dimensions which
        correspond to `shape`).  For example, if `shape=[2, 3]`, and field 'x'
        contains an int32 vector of size `10` for each structure, then
        `field_specs['x']` should be `tf.TensorSpec([2, 3, 10], tf.int32)`.
    """
    self._shape = tensor_shape.as_shape(shape)
    self._field_specs = dict(field_specs)

    # Perform a few sanity checks on the inputs.
    if self._shape.ndims is None:
      raise TypeError("StructuredTensor's shape must have known rank.")
    if not isinstance(self._field_specs, dict):
      raise TypeError('field_specs must be a dictionary')
    for key, value in self._field_specs.items():
      if not isinstance(key, str):
        raise TypeError('field_specs must be a dictionary with string keys.')
      if not isinstance(value, (StructuredTensorSpec, tensor_spec.TensorSpec,
                                ragged_tensor.RaggedTensorSpec)):
        raise TypeError('field_spec must be a dictionary with TypeSpec values.')

  @property
  def value_type(self):
    return StructuredTensor

  def _to_components(self, value):
    return value._fields

  def _from_components(self, components):
    return StructuredTensor(self._shape, components)

  @property
  def _component_specs(self):
    return self._field_specs

  @classmethod
  def from_value(cls, value):
    field_specs = dict((k, type_spec.type_spec_from_value(v))
                       for (k, v) in value._fields.items())
    return cls(value.shape, field_specs)

  def _serialize(self):
    return (self._shape, self._field_specs)

  def _batch(self, batch_size):
    # pylint: disable=protected-access
    return StructuredTensorSpec(
        tensor_shape.TensorShape([batch_size]).concatenate(self._shape),
        dict((k, v._batch(batch_size)) for (k, v) in self._field_specs.items()))

  def _unbatch(self):
    # pylint: disable=protected-access
    return StructuredTensorSpec(
        self._shape[1:],
        dict((k, v._unbatch()) for (k, v) in self._field_specs.items()))


# Regular expression used to determine whether a string is a valid field name.
# Note: we plan to relax (or possibly eliminate) this in the future; you
# should not rely on the fact that some field names are currently disallowed.
_FIELD_NAME_RE = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')
