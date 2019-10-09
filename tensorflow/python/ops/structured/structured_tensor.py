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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.ragged import ragged_tensor


class StructuredTensor(object):
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

  def __init__(self, implementation):
    """Private constructor -- use factory methods instead."""
    if not isinstance(implementation,
                      (_DenseStructuredTensor, _RaggedStructuredTensor)):
      raise TypeError('Invalid implementation class.')
    self._impl = implementation

  @classmethod
  def from_fields(cls, shape, fields):
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

    if shape.rank > 1 and any(ragged_tensor.is_ragged(v)
                              for v in fields.values()):
      return cls._from_ragged_fields(shape, fields)
    else:
      return cls(_DenseStructuredTensor(shape, fields))

  @classmethod
  def from_row_splits(cls, values, row_splits):
    """Creates a ragged StructuredTensor with rows partitioned by `row_splits`.

    The returned `StructuredTensor` corresponds with the python list defined by:

    ```python
    result = [values[row_splits[i]:row_splits[i + 1]]
              for i in range(len(row_splits) - 1)]
    ```

    Args:
      values: A `StructuredTensor` with shape `[nvals, ...]`.
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be
        empty, and must be sorted in ascending order.  `row_splits[0]` must be
        zero and `row_splits[-1]` must be `nvals`.

    Returns:
      A ragged `StructuredTensor`.  `result.rank = values.rank + 1`.
    """
    return cls(_RaggedStructuredTensor(values, row_splits))

  # @TODO(edloper): Add from_row_lengths, etc.

  @classmethod
  def _from_ragged_fields(cls, shape, fields):
    """Creates a `StructuredTensor` from a dictionary of ragged fields."""
    bad_splits = ('`fields` are not consistent in the outer %d dimensions' %
                  shape.rank)
    # All of the fields must have matching row_splits.  Check that they
    # actually match, and pick one to use as the row_splits for the result.
    # TODO(edloper): If the ragged tensors have uniform_row_length, then
    # check that use that instead.
    if any(not ragged_tensor.is_ragged(v) for v in fields.values()):
      raise ValueError(bad_splits)
    shared_row_splits = [v.row_splits for v in fields.values()]
    row_splits = shared_row_splits[0]
    checks = [check_ops.assert_equal(row_splits, s, message=bad_splits)
              for s in shared_row_splits[1:]]
    with ops.control_dependencies(checks):
      row_splits = array_ops.identity(row_splits)

    # Build a values for the RaggedStructuredTensor by stripping the outer
    # row_splits off of each field.
    values_shape = tensor_shape.TensorShape([None]).concatenate(shape[2:])
    values = cls.from_fields(values_shape,
                             dict((k, v.values) for (k, v) in fields.items()))

    return cls.from_row_splits(values, row_splits)

  #=============================================================================
  # Properties
  #=============================================================================

  @property
  def rank(self):
    """The rank of this StructuredTensor.  Guaranteed not to be `None`."""
    return self._impl.rank

  @property
  def shape(self):
    """The static shape of this StructuredTensor.

    The returned `TensorShape` is guaranteed to have a known rank, but the
    individual dimension sizes may be unknown.

    Returns:
      `tf.TensorShape`
    """
    return self._impl.shape

  #=============================================================================
  # Encoding
  #=============================================================================

  def field_names(self):
    """Returns the string field names for this `StructuredTensor`."""
    return self._impl.field_names()

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
    return self._impl.field_value(field_name)


# Regular expression used to determine whether a string is a valid field name.
# Note: we plan to relax (or possibly eliminate) this in the future; you
# should not rely on the fact that some field names are currently disallowed.
_FIELD_NAME_RE = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')


class _DenseStructuredTensor(object):
  """A StructuredTensor with uniform dimensions.

  ### Encoding

  Internally, each `_DenseStructuredTensor` is encoded using two objects:

    * `shape`: A `TensorShape` specifying the shape of the `StructuredTensor`.
      E.g., if `st.shape=[5, 10]` then `st` is a 5x10 matrix of structures.
      The rank must be statically known -- i.e., `shape.ndims` may not be
      `None`.

    * `fields`: A python dictionary mapping each field name to a `Tensor` or
      `CompositeTensor` encoding that field's values.  If `s` is an
      N-dimensional `StructuredTensor`, then for each field `(f, v)` in
      `s.fields.items()`:

        * `s.shape` = `v.shape[:N]`
        * `s[i1...iN][f]` = `v[i1...iN]`

      For scalar `StructuredTensor`s (where `N=0` and `s.shape=()`), this
      simplifies to just:

        * `s[f]` = v
  """

  def __init__(self, shape, fields):
    """Creates a `_DenseStructuredTensor`."""
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
    # TODO(edloper): For RaggedTensor fields, check that the outer `rank`
    # dimensions are all uniform.  (Only need to check if rank>1.)

  @property
  def rank(self):
    return self._static_shape.rank

  @property
  def shape(self):
    return self._static_shape

  def __repr__(self):
    return 'StructuredTensor(%s, %r)' % (self._static_shape, self._fields)

  def field_names(self):
    """See StructuredTensor.field_names for documentation."""
    return tuple(self._fields.keys())

  def field_value(self, field_name):
    """See StructuredTensor.field_value for documentation."""
    if isinstance(field_name, (list, tuple)):
      value = self
      for f in field_name:
        value = value.field_value(f)
      return value
    return self._fields[field_name]


class _RaggedStructuredTensor(object):
  """A StructuredTensor with ragged dimensions.

  ### Encoding

  Internally, each `_RaggedStructuredTensor` is encoded using:

    * `values`: A `StructuredTensor` with `rank>=1`, containing a
      concatenation of all the rows in this `RaggedStructuredTensor`.

    * One or more "row-partitioning tensors" that indicate how the concatenated
      `values` tensor is divided into rows.

  For more information on this encoding, see the `RaggedTensor` documentation.
  """

  def __init__(self,
               values,
               row_splits,
               row_lengths=None,
               value_rowids=None,
               nrows=None,
               uniform_row_length=None):
    """Creates a `_RaggedStructuredTensor`."""
    # Validate values.
    if not isinstance(values, StructuredTensor):
      raise TypeError('values must be a StructuredTensor')
    values.shape.with_rank_at_least(1)

    # Convert row-partitioning tensors.
    with ops.name_scope(None, 'RaggedStructuredTensor', [
        values, row_splits, row_lengths, value_rowids, nrows, uniform_row_length
    ]):
      if row_splits is not None:
        row_splits = ops.convert_to_tensor(
            row_splits, name='row_splits', preferred_dtype=dtypes.int64)
      if row_lengths is not None:
        row_lengths = ops.convert_to_tensor(
            row_lengths, name='row_lengths', preferred_dtype=dtypes.int64)
      if value_rowids is not None:
        value_rowids = ops.convert_to_tensor(
            value_rowids, name='value_rowids', preferred_dtype=dtypes.int64)
      if nrows is not None:
        nrows = ops.convert_to_tensor(
            nrows, name='nrows', preferred_dtype=dtypes.int64)
      if uniform_row_length is not None:
        uniform_row_length = ops.convert_to_tensor(
            uniform_row_length,
            name='uniform_row_length',
            preferred_dtype=dtypes.int64)

    # Validate row-partitioning tensors.
    partitions = [
        row_splits, row_lengths, value_rowids, nrows, uniform_row_length
    ]
    partition_dtypes = set([p.dtype for p in partitions if p is not None])
    if len(partition_dtypes) != 1:
      raise ValueError('Inconsistent dtypes for row-partitioning tensors')
    if list(partition_dtypes)[0] not in (dtypes.int32, dtypes.int64):
      raise ValueError('Row-partitioning tensors must have dtype '
                       'int32 or int64, got %s' % list(partition_dtypes)[0])
    if row_splits is not None:
      row_splits.shape.assert_has_rank(1)
      if tensor_shape.dimension_value(row_splits.shape[0]) == 0:
        raise ValueError('row_splits may not be empty')
    if row_lengths is not None:
      row_lengths.shape.assert_has_rank(1)
    if value_rowids is not None:
      value_rowids.shape.assert_has_rank(1)
    if nrows is not None:
      nrows.shape.assert_has_rank(0)
    if uniform_row_length is not None:
      uniform_row_length.shape.assert_has_rank(0)

    self._values = values
    self._row_splits = row_splits
    self._row_lengths = row_lengths
    self._value_rowids = value_rowids
    self._nrows = nrows
    self._uniform_row_length = uniform_row_length

  @property
  def rank(self):
    return self._values.shape.rank + 1

  @property
  def shape(self):
    """The statically known shape of this RaggedStructuredTensor."""
    nrows = tensor_shape.dimension_at_index(self._row_splits.shape, 0) - 1

    if self._uniform_row_length is not None:
      row_length = tensor_util.constant_value(self._uniform_row_length)
    else:
      row_length = None

    values_shape = self._values.shape
    value_shape = values_shape[1:]
    return tensor_shape.TensorShape([nrows,
                                     row_length]).concatenate(value_shape)

  @property
  def ragged_rank(self):
    """The number of ragged dimensions in this RaggedStructuredTensor."""
    values_is_ragged = isinstance(self._values, _RaggedStructuredTensor)
    return self._values.ragged_rank + 1 if values_is_ragged else 1

  def __repr__(self):
    return 'StructuredTensor(%s, %r)' % (self._static_shape, self._fields)

  def field_names(self):
    """See StructuredTensor.field_names for documentation."""
    return tuple(self._values.field_names())

  def field_value(self, field_name):
    """See StructuredTensor.field_value for documentation."""
    if isinstance(field_name, (list, tuple)):
      value = self
      for f in field_name:
        value = value.field_value(f)
      return value
    return ragged_tensor.RaggedTensor(
        values=self._values.field_value(field_name),
        row_splits=self._row_splits,
        cached_row_lengths=self._row_lengths,
        cached_value_rowids=self._value_rowids,
        cached_nrows=self._nrows,
        uniform_row_length=self._uniform_row_length,
        internal=True)
