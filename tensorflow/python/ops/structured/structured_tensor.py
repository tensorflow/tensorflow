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

import numpy as np

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest


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
  >>> s1 = StructuredTensor.from_pyval(
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]})
  >>> s1.shape
  TensorShape([])
  >>> s1["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=82>

  >>> # A vector StructuredTensor describing three people.
  >>> s2 = StructuredTensor.from_pyval([
  ...     {"age": 12, "nicknames": ["Josaphine"]},
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]},
  ...     {"age": 42, "nicknames": ["Elmo"]}])
  >>> s2.shape
  TensorShape([3])
  >>> s2[0]["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=12>
  ```

  ### Field Paths

  A *field path* is a tuple of field names, specifying the path to a nested
  field.
  """

  #=============================================================================
  # Constructor & Factory Methods
  #=============================================================================

  def __init__(self, fields, shape, nrows, row_partitions, internal=False):
    """Private constructor -- use factory methods to create StructuredTensors.

    This constructor builds a `StructuredTensor` from the given attributes,
    performing minimal validation.

    Args:
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`.  (This dict is not copied, so the caller must ensure
        that it does not get mutated via leaked references.)
      shape: `tf.TensorShape` with statically known rank.
      nrows: scalar integer `tf.Tensor`, or `None` if `shape.rank==0`.
      row_partitions: tuple of `RowPartition`s, with length `shape.rank-1`.
      internal: Private key value, required to ensure that this private
        constructor is *only* called from the factory methods.
    """
    if internal is not _structured_tensor_factory_key:
      raise ValueError('StructuredTensor constructor is private; please use '
                       'one of the factory methods instead (e.g., '
                       'StructuredTensor.from_fields())')
    assert isinstance(fields, dict), fields
    assert isinstance(shape, tensor_shape.TensorShape), shape
    assert nrows is None or isinstance(nrows, ops.Tensor), nrows
    assert isinstance(row_partitions, tuple), row_partitions
    self._fields = fields
    self._shape = shape
    self._nrows = nrows
    self._row_partitions = row_partitions

  @classmethod
  def from_fields(cls,
                  fields,
                  shape=(),
                  nrows=None,
                  row_partitions=None,
                  validate=False):
    """Creates a `StructuredTensor` from a dictionary of fields.

    Args:
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`, providing the values for individual fields in each
        structure.  If `shape.rank > 0`, then every tensor in `fields` must have
        the same shape in the first `shape.rank` dimensions; and that shape must
        be compatible with `shape`; and
        `result[i1...iN][key] = fields[key][i1...iN]` (where `N==shape.rank`).
      shape: A `TensorShape`: static information about the shape of the
        `StructuredTensor`.  Must have a known `rank`.  Defaults to scalar
        shape (i.e. `rank=0`).
      nrows: scalar integer tensor containing the number of rows in this
        `StructuredTensor`.  Should only be specified if `shape.rank > 0`.
        Default value is inferred from the `fields` values.  If `fields` is
        empty, then this must be specified.
      row_partitions: A list of `RowPartition`s describing the (possibly ragged)
        shape of this `StructuredTensor`.  Should only be specified if
        `shape.rank > 1`.  Default value is inferred from the `fields` values.
        If `fields` is empty, then this must be specified.
      validate: If true, then add runtime validation ops that check that the
        field values all have compatible shapes in the outer `shape.rank`
        dimensions.

    Returns:
      A `StructuredTensor`.

    Examples:

      >>> StructuredTensor.from_fields({'x': 1, 'y': [1, 2, 3]})
      <StructuredTensor(fields={
                            x: tf.Tensor(1, shape=(), dtype=int32),
                            y: tf.Tensor([1 2 3], shape=(3,), dtype=int32)},
                        shape=())>

      >>> StructuredTensor.from_fields({'foo': [1, 2], 'bar': [3, 4]},
      ...                              shape=[2])
      <StructuredTensor(fields={
                            bar: tf.Tensor([3 4], shape=(2,), dtype=int32),
                            foo: tf.Tensor([1 2], shape=(2,), dtype=int32)},
                        shape=(2,))>

    """
    shape = tensor_shape.as_shape(shape)
    rank = shape.rank
    if rank is None:
      raise ValueError("StructuredTensor's shape must have known rank.")
    if not isinstance(fields, dict):
      raise TypeError('fields must be a dictionary, got %s' %
                      type(fields).__name__)
    if rank < 2 and row_partitions:
      raise ValueError('row_partitions must be None or [] if shape.rank<2')
    if rank == 0 and nrows is not None:
      raise ValueError('nrows must be None if shape.rank==0')
    if row_partitions is not None:
      row_partitions = tuple(row_partitions)
      if len(row_partitions) != max(0, rank - 1):
        raise ValueError('len(row_partitions) must be shape.rank-1')
    elif rank < 2:
      row_partitions = ()

    fields = dict(fields)  # Make a private copy.
    with ops.name_scope(None, 'StructuredTensor', fields.values()):

      # Validate keys and convert field values to tensors.
      for key, value in fields.items():
        if not isinstance(key, str):
          raise TypeError('Unexpected type for key in `fields`: %r' % key)
        if not _FIELD_NAME_RE.match(key):
          raise ValueError('Field name %r is not currently allowed.' % key)
        fields[key] = _convert_to_structured_field_value(value)

      # Determine dtype for row_partitions and nrows.
      shape_dtype = _find_shape_dtype(fields, nrows, row_partitions)
      if nrows is not None:
        nrows = ops.convert_to_tensor(nrows, shape_dtype)

      # Get the static TensorShape for this StructuredTensor.
      if rank > 0:
        for key, value in fields.items():
          if not shape.is_compatible_with(value.shape[:rank]):
            raise ValueError('Field {} has shape {}, which is incompatible '
                             'with the shape that was specified or inferred '
                             'from other fields: {}'.format(
                                 key, value.shape[:rank], shape))
          shape = shape.merge_with(value.shape[:rank])

      if rank == 1:
        # Find a consistent value for `nrows`.
        static_nrows = tensor_shape.dimension_at_index(shape, 0)
        for value in fields.values():
          nrows, static_nrows = _merge_nrows(nrows, static_nrows, value,
                                             shape_dtype, validate)
        if nrows is None:
          if static_nrows.value is None:
            raise ValueError('nrows must be specified if rank==1 '
                             'and `fields` is empty.')
          else:
            nrows = constant_op.constant(static_nrows.value, shape_dtype)

      if rank > 1:
        # Find a consistent list of RowPartitions.
        for value in fields.values():
          row_partitions = _merge_row_partitions(row_partitions, value, rank,
                                                 shape_dtype, validate)
        if row_partitions is None:
          if not shape.is_fully_defined():
            raise ValueError('row_partitions must be specified if rank>1 '
                             'and `fields` is empty.')
          else:
            row_partitions = _row_partitions_for_uniform_shape(
                np.array(shape.as_list(), dtype=shape_dtype.as_numpy_dtype),
                shape.rank)
        assert len(row_partitions) == rank - 1
        nrows = row_partitions[0].nrows()
        # Update all field values to use the shared RowPartition objects.
        fields = dict([(k, _replace_row_partitions(v, row_partitions))
                       for (k, v) in fields.items()])

    return cls(
        fields,
        shape,
        nrows,
        row_partitions,
        internal=_structured_tensor_factory_key)

  #=============================================================================
  # Properties
  #=============================================================================

  @property
  def rank(self):
    """The rank of this StructuredTensor.  Guaranteed not to be `None`."""
    return self._shape.rank

  @property
  def shape(self):
    """The static shape of this StructuredTensor.

    The returned `TensorShape` is guaranteed to have a known rank, but the
    individual dimension sizes may be unknown.

    Returns:
      `tf.TensorShape`
    """
    return self._shape

  # TODO(edloper): Make this a func instead of a property?  Or make nrows
  # a property instead of a func?  Seems like these should be consistent.
  @property
  def row_partitions(self):
    """A tuple of `RowPartition`s defining the shape of this `StructuredTensor`.

    If this `StructuredTensor` has a ragged shape, then all fields will be
    encoded as either `RaggedTensor`s or `StructuredTensor`s with these
    `RowPartition`s used to define their outermost `self.rank` dimensions.

    If this `StructuredTensor` has a uniform (non-ragged) shape, then these
    row partitions will all be defined using `uniform_row_length`.

    Returns:
      A `tuple` of `RowPartition` objects with length `self.rank - 1`
      (or `0` if `self.rank < 2`).
    """
    return self._row_partitions

  def nrows(self):
    """The number of rows in this StructuredTensor (if rank>0).

    Returns:
      A scalar integer `Tensor` (or `None` if `self.rank == 0`).
    """
    return self._nrows

  def _is_eager(self):
    """True if all fields are composed of eager tensors."""
    tensors = nest.flatten(self, expand_composites=True)
    return all(isinstance(t, ops.EagerTensor) for t in tensors)

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
    `result[d1...dN]` contains the field value for the structure at
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

    Raises:
      KeyError: If the given field_name is not found.
    """
    if isinstance(field_name, (list, tuple)):
      value = self
      for f in field_name:
        if not isinstance(value, StructuredTensor):
          raise KeyError('Field path {} not found in {}'.format(
              field_name, self))
        value = value.field_value(f)
      return value
    return self._fields[field_name]

  #=============================================================================
  # Operators
  #=============================================================================

  # TODO(edloper): Add support for ellipsis and/or newaxis?
  def __getitem__(self, key):
    """Returns the specified piece of this StructuredTensor.

    * If `struct_tensor` is scalar (i.e., a single structure), then
      `struct_tensor[f]` returns the value of field `f` (where `f` must be a
      string).

    * If `struct_tensor` is non-scalar (i.e., a vector or higher-dimensional
      tensor of structures), `struct_tensor[i]` selects an element or slice of
      the tensor using standard Python semantics (e.g., negative values index
      from the end).  `i` may have any of the following types:

      * `int` constant
      * `string` constant
      * scalar integer `Tensor`
      * `slice` containing integer constants and/or scalar integer
        `Tensor`s

    #### Multidimensional indexing

    `StructuredTensor` supports multidimensional indexing.  I.e., `key` may be a
    `tuple` of values, indexing or slicing multiple dimensions at once.  For
    example, if `people` is a vector of structures, each of which has a vector-
    valued `names` field, then `people[3, 'names', 0]` is equivalent to
    `people[3]['names'][0]`; and `people[:, 'names', :]` will return a (possibly
    ragged) matrix of names, with shape `[num_people, num_names_per_person]`.

    Args:
      key: Indicates which piece of the StructuredTensor to return.
    Returns:
      A `Tensor`, `StructuredTensor`, or `RaggedTensor`.
    """
    if isinstance(key, list):
      key = tuple(key)
    elif not isinstance(key, tuple):
      key = (key,)
    if not key:
      return self

    if self._shape.rank == 0:
      return self._scalar_getitem(key)
    else:
      return self._tensor_getitem(key)

  def _scalar_getitem(self, key):
    if (isinstance(key[0], slice) and key[0].start is None and
        key[0].stop is None and key[0].step is None):
      fields = dict((field_name, field_value.__getitem__(key[1:]))
                    for (field_name, field_value) in self._fields.items())
      return StructuredTensor.from_fields(fields, self._shape)

    elif not isinstance(key[0], compat.bytes_or_text_types):
      raise ValueError('Key for indexing a StructuredTensor must be a '
                       "string or a full slice (':')")

    return self._fields[key[0]].__getitem__(key[1:])

  def _tensor_getitem(self, key):
    rank = self._shape.rank
    if len(key) <= rank:
      new_fields = dict((field_name, field_value.__getitem__(key))
                        for (field_name, field_value) in self._fields.items())
      result_shape = self.shape.as_list()
      for d, k in enumerate(key):
        if isinstance(k, slice):
          if not (k.start is None and k.stop is None and k.step is None):
            # TODO(edloper): Better static shape analysis here.
            result_shape[d] = None
        elif isinstance(k, (int, ops.Tensor)):
          result_shape[d] = -1  # mark for deletion
        elif k is None:
          raise ValueError('Slicing not supported for tf.newaxis')
        else:
          # Ellipsis, tf.newaxis:
          raise ValueError('Slicing not supported for %r' % k)
      result_shape = [d for d in result_shape if d != -1]
      return StructuredTensor.from_fields(new_fields, result_shape)

    else:
      if not isinstance(key[rank], compat.bytes_or_text_types):
        # TODO(edloper): Also support full slice here?
        raise ValueError('Key for indexing a StructuredTensor must be a string')
      return self._fields[key[rank]].__getitem__(key[:rank] + key[rank + 1:])

  def __repr__(self):
    return '<StructuredTensor(fields={%s}, shape=%s)>' % (', '.join(
        '"%s": %s' % (k, v)
        for k, v in sorted(self._fields.items())), self._shape)

  #=============================================================================
  # Conversion
  #=============================================================================

  def to_pyval(self):
    """Returns this StructuredTensor as a nested Python dict or list of dicts.

    Converts this `StructuredTensor` to a nested python value:

    * `StructTensors` with `rank=0` are converted into a dictionary, with an
      entry for each field.  Field names are used as keys and field values are
      converted to python values.  In particular:

      * Scalar Tensor fields are converted to simple values (such as
        `int` or `float` or `string`)
      * Non-scalar Tensor fields and RaggedTensor fields are converted to
        nested lists of simple values.
      * StructuredTensor fields are converted recursively using `to_pyval`.

    * `StructTensors` with `rank>0` are converted to nested python `list`s,
      containing one dictionary for each structure (where each structure's
      dictionary is defined as described above).

    Requires that all fields are Eager tensors.

    >>> StructuredTensor.from_fields(
    ...     {'a': [1, 2, 3]}, [3]).to_pyval()
    [{'a': 1}, {'a': 2}, {'a': 3}]

    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.

    Returns:
      A nested Python dict or list of dicts.
    """
    if not self._is_eager():
      raise ValueError(
          'StructuredTensor.to_pyval() is only supported in eager mode.')

    # Convert each field value to a nested list.
    result = {}
    for (key, value) in self._fields.items():
      if isinstance(value, ops.EagerTensor):
        value = value.numpy()
      if isinstance(value, np.ndarray):
        value = value.tolist()
      elif isinstance(value, ragged_tensor.RaggedTensor):
        value = value.to_list()
      elif isinstance(value, StructuredTensor):
        value = value.to_pyval()
      # TODO(edloper): Throw an exception if value is an unexpected type.
      result[key] = value

    # If rank>0, then re-group each value from dict-of-list to list-of-dict.
    if len(self._shape) > 0:  # pylint: disable=g-explicit-length-test
      return _pyval_field_major_to_node_major(
          list(result.keys()), list(result.values()), self._shape.as_list())
    else:
      return result

  @classmethod
  def from_pyval(cls, pyval, typespec=None):
    """Constructs a StructuredTensor from a nested Python structure.

    >>> StructuredTensor.from_pyval(
    ...     {'a': [1, 2, 3], 'b': [[4, 5], [6, 7]]})
    <StructuredTensor(fields={
                          a: tf.Tensor([1 2 3], shape=(3,), dtype=int32),
                          b: <tf.RaggedTensor [[4, 5], [6, 7]]>},
                      shape=())>

    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.

    Args:
      pyval: The nested Python structure that should be used to create the new
        `StructuredTensor`.
      typespec: A `StructuredTensorSpec` specifying the expected type for each
        field. If not specified, then all nested dictionaries are turned into
        StructuredTensors, and all nested lists are turned into Tensors (if
        rank<2) or RaggedTensors (if rank>=2).

    Returns:
      A `StructuredTensor`.
    """
    if isinstance(pyval, dict):
      return cls._from_pydict(pyval, typespec)
    elif isinstance(pyval, (list, tuple)):
      keys = set()
      rank = _pyval_find_struct_keys_and_depth(pyval, keys)
      if rank is not None:
        return cls._from_pylist_of_dict(pyval, keys, rank, typespec)
      else:
        return cls._from_pylist_of_value(pyval, typespec)
    else:
      return cls._from_pyscalar(pyval, typespec)

  @classmethod
  def _from_pydict(cls, pyval, typespec):
    """Converts python dictionary `pyval` to a StructuredTensor with rank=0."""
    if typespec is None:
      fields = dict((k, cls.from_pyval(v)) for (k, v) in pyval.items())
    else:
      spec_shape = typespec._shape  # pylint: disable=protected-access
      field_specs = typespec._field_specs  # pylint: disable=protected-access
      if not (isinstance(typespec, StructuredTensorSpec) and
              spec_shape.rank == 0 and set(pyval) == set(field_specs)):
        raise ValueError('Value does not match typespec: %r vs %r' %
                         (pyval, typespec))
      fields = dict(
          (k, cls.from_pyval(v, field_specs[k])) for (k, v) in pyval.items())
    return StructuredTensor.from_fields(fields=fields, shape=(), validate=False)

  @classmethod
  def _from_pylist_of_dict(cls, pyval, keys, rank, typespec):
    """Converts python list `pyval` to a StructuredTensor with rank>1."""
    fields = dict((key, []) for key in keys)
    for child in pyval:
      _pyval_update_fields(child, fields, 1)
    if typespec is None:
      shape = tensor_shape.TensorShape([None] * rank)
      for (key, target) in fields.items():
        fields[key] = cls.from_pyval(target)
    else:
      field_specs = typespec._field_specs  # pylint: disable=protected-access
      if ((not isinstance(typespec, StructuredTensorSpec)) or
          (set(fields) - set(field_specs))):
        raise ValueError('Value does not match typespec: %r vs %r' %
                         (pyval, typespec))
      shape = typespec._shape
      if shape.rank < rank:
        raise ValueError('Value does not match typespec (rank mismatch): '
                         '%r vs %r' % (pyval, typespec))
      for (key, spec) in field_specs.items():
        fields[key] = cls.from_pyval(fields.get(key, []), spec)
    return StructuredTensor.from_fields(
        fields=fields, shape=shape, validate=False)

  @classmethod
  def _from_pylist_of_value(cls, pyval, typespec):
    """Converts python list `pyval` to a Tensor or RaggedTensor with rank>1."""
    if typespec is None:
      return ragged_factory_ops.constant(pyval)
    elif isinstance(typespec, tensor_spec.TensorSpec):
      result = constant_op.constant(pyval, typespec.dtype)
      if not typespec.shape.is_compatible_with(result.shape):
        raise ValueError('Value does not match typespec: %r vs %r' %
                         (typespec, pyval))
      return result
    elif isinstance(typespec, ragged_tensor.RaggedTensorSpec):
      # pylint: disable=protected-access
      return ragged_factory_ops.constant(
          pyval,
          dtype=typespec._dtype,
          ragged_rank=typespec._ragged_rank,
          row_splits_dtype=typespec._row_splits_dtype,
          inner_shape=typespec._shape[typespec._ragged_rank + 1:])
    elif isinstance(typespec, StructuredTensorSpec):
      empty_rank = _pyval_empty_list_depth(pyval)
      if empty_rank is None:
        raise ValueError('Value does not match typespec: %r vs %r' %
                         (typespec, pyval))
      else:
        return cls._from_pylist_of_dict(pyval, set(), empty_rank, typespec)
    else:
      raise ValueError('Value does not match typespec: %r vs %r' %
                       (typespec, pyval))

  @classmethod
  def _from_pyscalar(cls, pyval, typespec):
    """Converts python scalar value `pyval` to a Tensor."""
    if typespec is None:
      return constant_op.constant(pyval)
    else:
      if not (isinstance(typespec, tensor_spec.TensorSpec) and
              typespec.shape.rank == 0):
        raise ValueError('Value does not match typespec: %r vs %r' %
                         (typespec, pyval))
      # TODO(edloper): Check that typespec.shape matches.
      return constant_op.constant(pyval, typespec.dtype)

  #=============================================================================
  # Transforms
  #=============================================================================

  # TODO(edloper): Add a 'validate' option here?
  # TODO(edloper): Unify nomenclature with RaggedTensor.  Should RaggedTensor
  # have a partition_outer_dimension method?
  def partition_outer_dimension(self, row_partition):
    """Partitions the outer dimension of this StructuredTensor.

    Returns a new `StructuredTensor` with the same values as `self`, where
    the outer dimension is partitioned into two (possibly ragged) dimensions.
    Requires that this StructuredTensor have an outer dimension (i.e.,
    `self.shape.rank > 0`).

    >>> st = StructuredTensor.from_pyval(
    ...     [{'foo': 12}, {'foo': 33}, {'foo': 99}])
    >>> partition = RowPartition.from_row_lengths([2, 0, 1])
    >>> st.partition_outer_dimension(partition)
    <StructuredTensor(fields={
                          foo: <tf.RaggedTensor [[12, 33], [], [99]]>},
                      shape=(3, None))>

    Args:
      row_partition: A `RowPartition`.

    Returns:
      A `StructuredTensor` with rank `values.rank + 1`.
    """
    if not isinstance(row_partition, RowPartition):
      raise TypeError('row_partition must be a RowPartition.')
    if self.shape.rank == 0:
      raise ValueError('Shape %s must have rank at least 1' % self.shape)
    return _partition_outer_dimension(self, row_partition)

  def merge_dims(self, outer_axis, inner_axis):
    """Merges outer_axis...inner_axis into a single dimension.

    Returns a copy of this RaggedTensor with the specified range of dimensions
    flattened into a single dimension, with elements in row-major order.

    >>> st = StructuredTensor.from_pyval(
    ...     [[{'foo': 12}, {'foo': 33}], [], [{'foo': 99}]])
    >>> st.merge_dims(0, 1)
    <StructuredTensor(fields={
                          foo: tf.Tensor([12 33 99], shape=(3,), dtype=int32)},
                      shape=(3,))>

    Args:
      outer_axis: `int`: The first dimension in the range of dimensions to
        merge. May be negative (to index from the last dimension).
      inner_axis: `int`: The last dimension in the range of dimensions to merge.
        May be negative (to index from the last dimension).

    Returns:
      A copy of this tensor, with the specified dimensions merged into a
      single dimension.  The shape of the returned tensor will be
      `self.shape[:outer_axis] + [N] + self.shape[inner_axis + 1:]`, where `N`
      is the total number of slices in the merged dimensions.
    """
    outer_axis = array_ops.get_positive_axis(
        outer_axis,
        self.shape.rank,
        axis_name='outer_axis',
        ndims_name='rank(self)')
    inner_axis = array_ops.get_positive_axis(
        inner_axis,
        self.shape.rank,
        axis_name='inner_axis',
        ndims_name='rank(self)')
    if not outer_axis < inner_axis:
      raise ValueError('Expected outer_axis (%d) to be less than '
                       'inner_axis (%d)' % (outer_axis, inner_axis))
    return _merge_dims(self, outer_axis, inner_axis)

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
      shape: The shape of the StructuredTensor.  shape.rank must not be None.
      field_specs: A dictionary mapping from field name to TypeSpec, specifying
        the tensor type used to encode each field. These TypeSpecs should
        specify the type of the entire field (including outer dimensions which
        correspond to `shape`).  For example, if `shape=[2, 3]`, and field 'x'
        contains an int32 vector of size `10` for each structure, then
        `field_specs['x']` should be `tf.TensorSpec([2, 3, 10], tf.int32)`.
    """
    shape = tensor_shape.as_shape(shape)

    # Perform a few sanity checks on the inputs.
    if shape.rank is None:
      raise TypeError("StructuredTensor's shape must have known rank.")
    if not isinstance(field_specs, dict):
      raise TypeError('field_specs must be a dictionary.')
    for key, value in field_specs.items():
      if not isinstance(key, str):
        raise TypeError('field_specs must be a dictionary with string keys.')
      if not isinstance(value, (StructuredTensorSpec, tensor_spec.TensorSpec,
                                ragged_tensor.RaggedTensorSpec)):
        raise TypeError('field_specs must be a dictionary with '
                        'TypeSpec values.')

    self._shape = shape
    self._field_specs = dict(field_specs)

  @property
  def value_type(self):
    return StructuredTensor

  def _to_components(self, value):
    return value._fields

  def _from_components(self, components):
    return StructuredTensor.from_fields(components, self._shape, validate=False)

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


#=============================================================================
# Helper funtions
#=============================================================================
# TODO(edloper): Move some of these helpers to row_partition.py?


def _convert_to_structured_field_value(value):
  """Converts `value` to a Tensor, RaggedTensor, or StructuredTensor."""
  if isinstance(value,
                (ops.Tensor, ragged_tensor.RaggedTensor, StructuredTensor)):
    return value
  elif ragged_tensor.is_ragged(value):
    return ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
  else:
    try:
      return ops.convert_to_tensor(value)
    except (ValueError, TypeError):
      raise TypeError('Unexpected type for value in `fields`: %r' % value)


def _find_shape_dtype(fields, nrows, row_partitions):
  """Return a consistent dtype for fields, nrows, & row_partitions."""
  shape_dtypes = set()
  for value in fields.values():
    if isinstance(value, ragged_tensor.RaggedTensor):
      shape_dtypes.add(value.row_splits.dtype)
    elif isinstance(value, StructuredTensor) and value.rank > 0:
      shape_dtypes.add(value.nrows().dtype)
  if isinstance(nrows, ops.Tensor):
    shape_dtypes.add(nrows.dtype)
  if row_partitions is not None:
    for partition in row_partitions:
      shape_dtypes.add(partition.dtype)
  if len(shape_dtypes) > 1:
    raise ValueError('field values have incompatible row_partition dtypes.')
  elif shape_dtypes:
    return shape_dtypes.pop()
  else:
    return dtypes.int64


def _merge_nrows(nrows, static_nrows, value, dtype, validate):
  """Merges `nrows` with `nrows(value)`.

  Checks that `value` has the expected number of rows (`nrows`), and returns
  `nrows`.  If `validate` is true, then add validation ops that check that
  the `nrows` values match.

  Args:
    nrows: scalar integer Tensor.
    static_nrows: tf.Dimension: static value of nrows, if known.
    value: Tensor or RaggedTensor or StructuredTensor
    dtype: dtype for `nrows`.
    validate: bool -- whether to add validation ops.

  Returns:
    A tuple `(nrows, static_nrows)`.
  """
  static_value_nrows = tensor_shape.dimension_at_index(value.shape, 0)
  if isinstance(value, ops.Tensor):
    value_nrows = array_ops.shape(value, out_type=dtype)[0]
  else:
    value_nrows = value.nrows()
  if nrows is None:
    nrows = value_nrows
  elif (static_value_nrows.value is not None and
        static_nrows.value is not None):
    if not static_value_nrows.is_compatible_with(static_nrows):
      raise ValueError('fields have incompatible nrows')
    nrows = value_nrows  # No need to add an assertion op.
  elif validate:
    nrows = control_flow_ops.with_dependencies([
        check_ops.assert_equal(nrows, value_nrows,
                               message='fields have incompatible nrows')
    ], nrows)
  return nrows, static_nrows.merge_with(static_value_nrows)


def _merge_row_partitions(row_partitions, value, rank, dtype, validate):
  """Merges `row_partitions` with `row_partitions(value)`."""
  if isinstance(value, ops.Tensor):
    value_row_partitions = _row_partitions_for_tensor(value, rank, dtype)

  elif isinstance(value, ragged_tensor.RaggedTensor):
    value_row_partitions = _row_partitions_for_ragged_tensor(value, rank, dtype)

  else:
    assert isinstance(value, StructuredTensor), type(value)
    value_row_partitions = value.row_partitions[:rank - 1]

  assert len(value_row_partitions) == rank - 1
  if row_partitions is None:
    return tuple(value_row_partitions)
  else:
    return tuple([
        p1.merge_precomputed_encodings(p2, validate)
        for (p1, p2) in zip(row_partitions, value_row_partitions)
    ])


def _row_partitions_for_tensor(value, rank, dtype):
  """Returns the row partitions for a tf.Tensor."""
  shape = array_ops.shape(value, out_type=dtype)
  return _row_partitions_for_uniform_shape(shape, rank)


def _row_partitions_for_ragged_tensor(value, rank, dtype):
  """Returns the row partitions for a tf.RaggedTensor."""
  assert rank > 1
  value_row_partitions = value._nested_row_partitions[:rank - 1]  # pylint: disable=protected-access
  if len(value_row_partitions) < (rank - 1):
    value_row_partitions += _row_partitions_for_tensor(
        value.flat_values, rank - len(value_row_partitions), dtype)
  assert len(value_row_partitions) == rank - 1
  return value_row_partitions


def _row_partitions_for_uniform_shape(shape, rank):
  """Returns row partitions for the given shape Tensor.

  Args:
    shape: A vector describing a uniform shape.
    rank: The number of dimensions to generate row partitions for

  Returns:
    A list of (rank-1) `RowPartition`s with uniform row length.
  """
  shape_cumprod = math_ops.cumprod(shape[:rank])
  # pylint: disable=g-complex-comprehension
  return tuple([
      RowPartition.from_uniform_row_length(
          uniform_row_length=shape[i + 1],
          nvals=shape_cumprod[i + 1],
          nrows=shape_cumprod[i]) for i in range(rank - 1)
  ])


def _pyval_field_major_to_node_major(keys, values, shape):
  """Regroup each field (k, v) from dict-of-list to list-of-dict.

  Given a "field-major" encoding of the StructuredTensor (which maps each key to
  a single nested list containing the values for all structs), return a
  corresponding "node-major" encoding, consisting of a nested list of dicts.
  `shape` is used to determine how far to recurse; and if `keys` is empty
  it is used to determine the sizes for empty lists.

  Args:
    keys: The field names (list of string).
    values: The field values (list of python values).  Must have the same length
      as `keys`.
    shape: A tuple specifying the shape of the `StructuredTensor`.

  Returns:
    A nested list of dict.
  """
  if not shape:
    return dict(zip(keys, values))
  elif not keys:
    if shape[0] in (0, None):
      return []
    else:
      return [_pyval_field_major_to_node_major((), (), shape[1:])] * shape[0]
  else:
    nvals = len(values[0])
    assert all(nvals == len(values[i]) for i in range(1, len(values)))
    return [
        _pyval_field_major_to_node_major(keys, value_slice, shape[1:])
        for value_slice in zip(*values)
    ]


def _pyval_find_struct_keys_and_depth(pyval, keys):
  """Finds the keys & depth of nested dictionaries in `pyval`.

  Args:
    pyval: A nested structure of lists, tuples, and dictionaries.
    keys: (output parameter) A set, which will be updated with any keys that are
      found in the nested dictionaries.

  Returns:
    The nesting depth of dictionaries in `pyval`, or `None` if `pyval` does
    not contain any dictionaries.
  Raises:
    ValueError: If dictionaries have inconsistent depth.
  """
  if isinstance(pyval, dict):
    keys.update(pyval.keys())
    return 0
  elif isinstance(pyval, (list, tuple)):
    depth = None
    for child in pyval:
      child_depth = _pyval_find_struct_keys_and_depth(child, keys)
      if child_depth is not None:
        if depth is None:
          depth = child_depth + 1
        elif depth != child_depth + 1:
          raise ValueError('Inconsistent depth of dictionaries')
    return depth
  else:
    return None


def _pyval_update_fields(pyval, fields, depth):
  """Append the field values from `pyval` to `fields`.

  Args:
    pyval: A python `dict`, or nested list/tuple of `dict`, whose value(s)
      should be appended to `fields`.
    fields: A dictionary mapping string keys to field values.  Field values
      extracted from `pyval` are appended to this dictionary's values.
    depth: The depth at which `pyval` should be appended to the field values.
  """
  if not isinstance(pyval, (dict, list, tuple)):
    raise ValueError('Expected dict or nested list/tuple of dict')

  for (key, target) in fields.items():
    for _ in range(1, depth):
      target = target[-1]
    target.append(pyval[key] if isinstance(pyval, dict) else [])

  if isinstance(pyval, (list, tuple)):
    for child in pyval:
      _pyval_update_fields(child, fields, depth + 1)


def _pyval_empty_list_depth(pyval):
  """Find the max depth for nested empty lists.

  Args:
    pyval: A nested python list.

  Returns:
    The maximum depth of empty lists in `pyval`, or None if `pyval` contains
    anything other than nested empty lists.
  """
  if isinstance(pyval, list):
    if not pyval:
      return 1
    depths = [_pyval_empty_list_depth(v) for v in pyval]
    if any(depth is None for depth in depths):
      return None
    else:
      return max(depths) + 1
  else:
    return None


def _replace_row_partitions(value, new_partitions):
  """Updates `value` to use `new_partitions` as its (outer) row partitions.

  This is used to ensure that all fields in a `StructuredTensor` use identical
  `RowPartition` objects for the shared dimensions.  In particular,
  `StructuredTensor.from_fields` first merges all of the row partitions from
  any fields, and then replaces the outer row partitions of all fields with
  the merged row partitions (using this function).

  Args:
    value: A `Tensor`, `RaggedTensor`, or `StructuredTensor`.
    new_partitions: A list of row-partitions that should be used by `value`.
      Must be equivalent to `value`'s current row partitions.

  Returns:
    A value that is equivalent to `value`, where outer row partitions have been
    replaced by `new_partitions`.
  """
  if isinstance(value, ops.Tensor) or not new_partitions:
    return value

  elif isinstance(value, ragged_tensor.RaggedTensor):
    return ragged_tensor.RaggedTensor._from_row_partition(  # pylint: disable=protected-access
        values=_replace_row_partitions(value.values, new_partitions[1:]),
        row_partition=new_partitions[0])

  else:
    assert isinstance(value, StructuredTensor)
    new_fields = dict((k, _replace_row_partitions(v, new_partitions))
                      for (k, v) in value._fields.items())
    return StructuredTensor(
        fields=new_fields,
        shape=value.shape,
        nrows=value.nrows(),
        row_partitions=new_partitions,
        internal=_structured_tensor_factory_key)


def _partition_outer_dimension(value, row_partition):
  """Partitions the outer dimension of `value` using `row_partitions`.

  Examples:

    >>> partition = row_partition.RowPartition.from_row_lengths([2, 0, 1])
    >>> _partition_outer_dimension(tf.constant([1, 2, 3]), partition)
    <tf.RaggedTensor [[1, 2], [], [3]]>

    >>> struct_value = StructuredTensor.from_pyval(
    ...     [{'x': 1}, {'x': 2}, {'x': 3}])
    >>> _partition_outer_dimension(struct_value, partition)
    <StructuredTensor(fields={
                          x: <tf.RaggedTensor [[1, 2], [], [3]]>},
                      shape=(3, None))>

  Args:
    value: Tensor, RaggedTensor, or StructuredTensor
    row_partition: RowPartition

  Returns:
    A value with the same type as `value`, where
    `result.rank = value.rank + 1`.
  """
  is_ragged = row_partition.uniform_row_length() is None
  if isinstance(value, ops.Tensor) and not is_ragged:
    new_shape = array_ops.concat(
        [[row_partition.nrows(),
          row_partition.uniform_row_length()],
         array_ops.shape(value, out_type=row_partition.dtype)[1:]],
        axis=0)
    return array_ops.reshape(value, new_shape)
  elif isinstance(value, (ops.Tensor, ragged_tensor.RaggedTensor)):
    return ragged_tensor.RaggedTensor._from_row_partition(  # pylint: disable=protected-access
        value, row_partition)
  else:
    assert isinstance(value, StructuredTensor)
    nrows = row_partition.static_nrows
    ncols = row_partition.static_uniform_row_length
    shape = tensor_shape.TensorShape([nrows, ncols]).concatenate(
        value.shape[1:])
    fields = dict((k, _partition_outer_dimension(v, row_partition))
                  for (k, v) in value._fields.items())
    return StructuredTensor(
        fields,
        shape,
        row_partition.nrows(), (row_partition,) + value.row_partitions,
        internal=_structured_tensor_factory_key)


def _merge_dims(value, outer_axis, inner_axis):
  """Merges `outer_axis...inner_axis` of `value` into a single dimension."""
  assert outer_axis < inner_axis
  if isinstance(value, (ops.Tensor, ragged_tensor.RaggedTensor)):
    return ragged_tensor.merge_dims(value, outer_axis, inner_axis)
  else:
    assert isinstance(value, StructuredTensor)

    # Build the new fields.
    fields = dict((k, _merge_dims(v, outer_axis, inner_axis))
                  for (k, v) in value._fields.items())

    # Build the new shape.
    value_shape = value.shape
    shape = (
        value_shape[:outer_axis] +
        [value_shape[outer_axis:inner_axis].num_elements()] +
        value_shape[inner_axis + 1:])

    # Build the new row_partitions & nrows
    if outer_axis == 0:
      if inner_axis == value.shape.rank - 1:
        partitions = ()
        nrows = value.row_partitions[-1].nvals()
      else:
        partitions = value.row_partitions[inner_axis:]
        nrows = partitions[0].nrows()
    else:
      # Use tf.gather to merge row_splits from the merged row partitions.
      merged_splits = value.row_partitions[outer_axis - 1].row_splits()
      for dim in range(outer_axis, inner_axis):
        merged_splits = array_ops.gather(value.row_partitions[dim].row_splits(),
                                         merged_splits)

      partitions = (
          value.row_partitions[:outer_axis - 1] +
          (RowPartition.from_row_splits(merged_splits),) +
          value.row_partitions[inner_axis:])
      nrows = partitions[0].nrows()

    return StructuredTensor(
        fields,
        shape,
        nrows,
        partitions,
        internal=_structured_tensor_factory_key)


_structured_tensor_factory_key = object()  # unique private object
