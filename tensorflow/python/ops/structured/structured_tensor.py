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

import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Each field may contain one of the following types of Tensors.
_FieldValue = Union[
    tensor.Tensor,
    ragged_tensor.RaggedTensor,
    'StructuredTensor',
    extension_type.ExtensionType
]
# Function that takes a FieldValue as input and returns the transformed
# FieldValue.
_FieldFn = Callable[[_FieldValue], _FieldValue]


@tf_export('experimental.StructuredTensor')
class StructuredTensor(extension_type.BatchableExtensionType):
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

  >>> # A scalar StructuredTensor describing a single person.
  >>> s1 = tf.experimental.StructuredTensor.from_pyval(
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]})
  >>> s1.shape
  TensorShape([])
  >>> s1["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=82>

  >>> # A vector StructuredTensor describing three people.
  >>> s2 = tf.experimental.StructuredTensor.from_pyval([
  ...     {"age": 12, "nicknames": ["Josaphine"]},
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]},
  ...     {"age": 42, "nicknames": ["Elmo"]}])
  >>> s2.shape
  TensorShape([3])
  >>> s2[0]["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=12>


  ### Field Paths

  A *field path* is a tuple of field names, specifying the path to a nested
  field.
  """
  _fields: Mapping[str, _FieldValue]
  _ragged_shape: dynamic_ragged_shape.DynamicRaggedShape

  __name__ = 'tf.StructuredTensor'
  #=============================================================================
  # Common Types
  #=============================================================================
  # pylint: disable=invalid-name
  # Field names work as key, and they can be a sequence to refer to the
  # sub-levels (embedded) StructuredTensor's.
  FieldName = Union[str, Sequence[str]]

  # pylint: enable=invalid-name

  #=============================================================================
  # Constructor & Factory Methods
  #=============================================================================
  def __init__(self, fields: Mapping[str, _FieldValue],
               ragged_shape: dynamic_ragged_shape.DynamicRaggedShape):
    self._fields = fields
    self._ragged_shape = ragged_shape

  @classmethod
  def _old_init(cls, fields, shape, nrows, row_partitions, internal=False):
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
      internal: ignored argument.

    Returns:
      a StructuredTensor.
    """
    assert isinstance(fields, dict), fields
    assert isinstance(shape, tensor_shape.TensorShape), shape
    assert nrows is None or isinstance(nrows, tensor.Tensor), nrows
    assert row_partitions is None or isinstance(row_partitions,
                                                tuple), row_partitions
    return StructuredTensor(
        fields=fields,
        ragged_shape=_dynamic_ragged_shape_init(fields, shape, nrows,
                                                row_partitions))

  @classmethod
  def from_shape(
      cls, ragged_shape: dynamic_ragged_shape.DynamicRaggedShape
  ) -> 'StructuredTensor':
    """Creates a `StructuredTensor` with no fields and ragged_shape.

    Args:
      ragged_shape: the shape of the structured tensor.

    Returns:
      a StructuredTensor with no fields and ragged_shape.
    """
    return StructuredTensor(fields={}, ragged_shape=ragged_shape)

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
        be compatible with `shape`; and `result[i1...iN][key] =
        fields[key][i1...iN]` (where `N==shape.rank`).
      shape: A `TensorShape`: static information about the shape of the
        `StructuredTensor`.  Must have a known `rank`.  Defaults to scalar shape
        (i.e. `rank=0`).
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

      >>> tf.experimental.StructuredTensor.from_fields({'x': 1, 'y': [1, 2, 3]})
      <StructuredTensor(
        fields={
          "x": tf.Tensor(1, shape=(), dtype=int32),
          "y": tf.Tensor([1 2 3], shape=(3,), dtype=int32)},
        shape=())>

      >>> tf.experimental.StructuredTensor.from_fields(
      ...     {'foo': [1, 2], 'bar': [3, 4]}, shape=[2])
      <StructuredTensor(
        fields={
          "bar": tf.Tensor([3 4], shape=(2,), dtype=int32),
          "foo": tf.Tensor([1 2], shape=(2,), dtype=int32)},
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
      # TODO(martinz): Make this have better errors.
      shape = _dynamic_ragged_shape_init(fields, shape, nrows, row_partitions)

      # TODO(martinz): This may not need to be done if all fields are dense.
      if shape.rank > 1:
        shape = shape._with_num_row_partitions(shape.rank - 1)

      # Validate keys and convert field values to tensors.
      for key, value in fields.items():
        if not isinstance(key, str):
          raise TypeError(f'Unexpected type for key in `fields`: {key}')
        if not _FIELD_NAME_RE.match(key):
          raise ValueError('Field name %r is not currently allowed.' % key)
        fields[key] = _convert_to_structured_field_value(value)

        fields = dict([(k, _replace_row_partitions(v, row_partitions))
                       for (k, v) in fields.items()])
      return cls(fields=fields, ragged_shape=shape)

  @classmethod
  def from_fields_and_rank(
      cls,
      fields: Mapping[str, _FieldValue],
      rank: int,
      validate: bool = False,
      dtype: Optional[dtypes.DType] = None) -> 'StructuredTensor':
    """Creates a `StructuredTensor` from a nonempty dictionary of fields.

    Note that if the shape dtype is not specified, the shape dtype will be
    inferred from any fields that have a shape dtype. If fields differ, then
    int64 will be preferred to int32, because coercing from int32 to int64 is
    safer than coercing from int64 to int32.

    If there are no ragged fields, then it will be int64 by default, but this
    will be changed to int32 in the future.

    Args:
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`, providing the values for individual fields in each
        structure.  If `rank > 0`, then every tensor in `fields` must have the
        same shape in the first `rank` dimensions. Cannot be empty.
      rank: The rank of the resulting structured tensor.
      validate: If true, then add runtime validation ops that check that the
        field values all have compatible shapes in the outer `rank` dimensions.
      dtype: If specified, then forces dtype of the shape to be this.

    Returns:
      A `StructuredTensor`.
    Examples:
      >>> tf.experimental.StructuredTensor.from_fields_and_rank(
      ...     {'x': 1, 'y': [1, 2, 3]}, 0)
      <StructuredTensor(
        fields={
          "x": tf.Tensor(1, shape=(), dtype=int32),
          "y": tf.Tensor([1 2 3], shape=(3,), dtype=int32)},
        shape=())>
      >>> StructuredTensor.from_fields_and_rank({'foo': [1, 2], 'bar': [3, 4]},
      ...                              1)
      <StructuredTensor(
        fields={
          "bar": tf.Tensor([3 4], shape=(2,), dtype=int32),
          "foo": tf.Tensor([1 2], shape=(2,), dtype=int32)},
        shape=(2,))>
    """
    if not fields:
      raise ValueError('Must provide at least one field')
    if not isinstance(rank, int):
      raise ValueError('rank must be an integer')
    if rank < 0:
      raise ValueError('rank must be nonnegative')
    fields = {
        k: _convert_to_structured_field_value(v) for (k, v) in fields.items()
    }
    if dtype is None:
      dtype = _find_shape_dtype(fields, None, None)
    fields = _fields_with_dtype(fields, dtype)

    shape = _shape_from_fields(fields, rank, dtype)
    if rank > 1:
      shape = shape._with_num_row_partitions(rank - 1)
    new_rp = shape._row_partitions  # pylint: disable=protected-access
    fields = {
        k: _replace_row_partitions(v, new_rp) for (k, v) in fields.items()
    }
    return StructuredTensor(fields=fields, ragged_shape=shape)

  def with_updates(self,
                   updates: Dict[FieldName, Union[_FieldValue, _FieldFn, None]],
                   validate: bool = False) -> 'StructuredTensor':
    """Creates a new `StructuredTensor` with the updated fields.

    If this `StructuredTensor` is a scalar, and `k` is the `FieldName` being
    updated and `v` the new value, then:

    ```
    result[k] = v              # If (k, v) is in updates and v is a FieldValue
    result[k] = f(self[k])     # If (k, f) is in updates and f is a FieldFn
    result[k] = self[k]        # If k is in self.field_names but not in updates
    ```

    If this `StructuredTensor` has rank `N` and shape `[D1...DN]`, then each
    FieldValue `v` in `updates` must have shape `[D1...DN, ...]`, that is,
    prefixed with the same shape as the `StructuredTensor`. Then the resulting
    `StructuredTensor` will have:

    ```
    result[i1...iN][k] = v[i1...iN]                        # (k, v) in updates
    result[i1...iN][k] = f(self.field_value(k))[i1...iN]   # (k, f) in updates
    result[i1...iN][k] = self[i1...iN][k]                  # k not in updates
    ```

    Note that `result.shape` is always equal to `self.shape` (but the shapes
    of nested StructuredTensors may be changed if they are updated with new
    values).

    Args:
      updates: A dictionary mapping `FieldName` to either a `FieldValue` to be
        used to update, or a `FieldFn` that will transform the value for the
        given `FieldName`. `FieldName` can be a string for a direct field, or a
        sequence of strings to refer to a nested sub-field. `FieldFn` is a
        function that takes a `FieldValue` as input and should return a
        `FieldValue`. All other fields are copied over to the new
        `StructuredTensor`. New `FieldName` can be given (to add new fields),
        but only to existing `StructuredTensor`, it won't automatically create
        new nested structures -- but one can create a whole `StructureTensor`
        sub-structure and set that into an existing structure. If the new value
        is set to `None`, it is removed.
      validate: If true, then add runtime validation ops that check that the
        field values all have compatible shapes in the outer `shape.rank`
        dimensions.

    Returns:
      A `StructuredTensor`.

    Raises:
      `ValueError`: If the any of the `FieldName` keys points to non-existent
        sub-structures, if parent and child nodes are updated, if shapes
        change, if a delete update is given for a non-existent field, or if a
        `FieldFn` transforming function is given for a `FieldName` that doesn't
        yet exist.

    Examples:

    >>> shoes_us = tf.experimental.StructuredTensor.from_pyval([
    ...    {"age": 12, "nicknames": ["Josaphine"],
    ...       "shoes": {"sizes": [8.0, 7.5, 7.5]}},
    ...    {"age": 82, "nicknames": ["Bob", "Bobby"],
    ...        "shoes": {"sizes": [11.0, 11.5, 12.0]}},
    ...    {"age": 42, "nicknames": ["Elmo"],
    ...        "shoes": {"sizes": [9.0, 9.5, 10.0]}}])
    >>> def us_to_europe(t):
    ...   return tf.round(t * 2.54 + 17.0)  # Rough approximation.
    >>> shoe_sizes_key = ("shoes", "sizes")
    >>> shoes_eu = shoes_us.with_updates({shoe_sizes_key: us_to_europe})
    >>> shoes_eu.field_value(shoe_sizes_key)
    <tf.RaggedTensor [[37.0, 36.0, 36.0], [45.0, 46.0, 47.0],
    [40.0, 41.0, 42.0]]>
    """
    updates_items = [(_normalize_field_name_to_tuple(name), value)
                     for name, value in updates.items()]

    # Sort by keys and check for updates of both parent and child nodes.
    updates_items = sorted(updates_items)
    for i in range(1, len(updates_items)):
      # Parent of a node would precede node in the sorted order.
      name = updates_items[i][0]  # item[0] is the name, item[1] is the value.
      prev_name = updates_items[i - 1][0]
      if name[:len(prev_name)] == prev_name:
        raise ValueError(
            '`StructuredTensor.with_updates` does not allow both parent and '
            'child nodes to be updated: parent={}, child={}. If needed you can '
            'update child nodes in the parent update value.'.format(
                prev_name, name))
    return self._with_updates_impl((), updates_items, validate)

  def _with_updates_impl(self, error_prefix: Tuple[str, ...],
                         updates: List[Tuple[FieldName, Union[_FieldValue,
                                                              _FieldFn]]],
                         validate: bool) -> 'StructuredTensor':
    """Recursive part of `with_updates` implementation."""
    # Get current fields.
    new_fields = dict(self._fields)

    # Convert field name to string with full path for error messages.
    def name_fullpath(name: Sequence[str]) -> str:
      return str(error_prefix + (name,))

    # Apply value if a function or the value itself.
    def apply_value(name: str, value: Union[_FieldValue,
                                            _FieldFn]) -> _FieldValue:
      if callable(value):
        # `value` is actually a transforming function.
        if name not in new_fields:
          raise ValueError(
              '`StructuredTensor.with_updates` cannot update the field {} '
              'because a transforming function was given, but that field '
              'does not already exist.'.format(name_fullpath(name)))
        value = value(new_fields[name])
      return value

    # Merge updates.
    for name, value in updates:
      if not name or not name[0]:
        raise ValueError(
            '`StructuredTensor.with_updates` does not allow empty names '
            '{}.'.format(name_fullpath(name)))

      if len(name) == 1:
        name = name[0]
        if value is None:
          if name not in new_fields:
            raise ValueError(
                '`StructuredTensor.with_updates` cannot delete field '
                '{} because it is not present.'.format(name_fullpath(name)))
          new_fields.pop(name)
        else:
          new_fields[name] = apply_value(name, value)
      else:
        # Recursive
        prefix = name[0]
        suffix = name[1:]
        if prefix not in new_fields:
          raise ValueError(
              '`StructuredTensor.with_updates` cannot create new sub-field '
              '{} if parent field {} is not set.'.format(
                  error_prefix + tuple(name), name_fullpath(prefix)))
        current_value = new_fields[prefix]
        if not isinstance(current_value, StructuredTensor):
          raise ValueError(
              '`StructuredTensor.with_updates` cannot create new sub-field '
              '{} if parent structure {} is not a `StructuredTensor` that '
              'can contain sub-structures -- it is a `{}`.'.format(
                  error_prefix + tuple(name), name_fullpath(prefix),
                  type(current_value)))
        one_update = [(suffix, value)]

        # Accessing protected member in recursion.
        # FutureWork: optimize by aggregating the recursions, instead of
        #   calling one at a time.
        # pylint: disable=protected-access
        value = current_value._with_updates_impl(error_prefix + (prefix,),
                                                 one_update, validate)
        # pylint: enable=protected-access
        new_fields[prefix] = value

    # TODO(edloper): When validate=True, only validate the modified fields.
    try:
      return StructuredTensor.from_fields(
          new_fields,
          shape=self.shape,
          row_partitions=self.row_partitions,
          nrows=self.nrows(),
          validate=validate)

    except ValueError as e:
      msg = '`StructuredTensor.with_updates` failed'
      if error_prefix:
        msg = '{} for field {}'.format(msg, error_prefix)
      raise ValueError(msg) from e

  def _promote_helper(self, source_path, new_parent_path):
    """Creates a promoted field without adding it to the structure.

    Args:
      source_path: the source path in the structured tensor.
      new_parent_path: the new parent path. Must be a prefix of source_path.

    Returns:
      a composite tensor of source_path promoted.
    Raises:
      ValueError: if the shape of the field is unknown and the right strategy
      cannot be determined.
    """
    current_field = self.field_value(source_path)
    new_parent_rank = self.field_value(new_parent_path).rank
    parent_rank = self.field_value(source_path[:-1]).rank
    if new_parent_rank == parent_rank:
      return current_field
    current_field_rank = current_field.shape.rank
    if current_field_rank is None:
      raise ValueError('Cannot determine if dimensions should be merged.')
    inner_dim = min(parent_rank, current_field_rank - 1)
    if inner_dim <= new_parent_rank:
      return current_field
    return _merge_dims_generic(current_field, new_parent_rank, inner_dim)

  def promote(self, source_path, new_name):
    """Promotes a field, merging dimensions between grandparent and parent.

    >>> d = [
    ...  {'docs': [{'tokens':[1, 2]}, {'tokens':[3]}]},
    ...  {'docs': [{'tokens':[7]}]}]
    >>> st = tf.experimental.StructuredTensor.from_pyval(d)
    >>> st2 =st.promote(('docs','tokens'), 'docs_tokens')
    >>> st2[0]['docs_tokens']
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
    >>> st2[1]['docs_tokens']
    <tf.Tensor: shape=(1,), dtype=int32, numpy=array([7], dtype=int32)>

    Args:
      source_path: the path of the field or substructure to promote; must have
        length at least 2.
      new_name: the name of the new field (must be a string).

    Returns:
      a modified structured tensor with the new field as a child of the
      grandparent of the source_path.

    Raises:
      ValueError: if source_path is not a list or a tuple or has a length
        less than two, or new_name is not a string, or the rank
        of source_path is unknown and it is needed.
    """
    if not isinstance(new_name, str):
      raise ValueError('new_name is not a string')
    if not isinstance(source_path, (list, tuple)):
      raise ValueError('source_path must be a list or tuple')

    if len(source_path) < 2:
      raise ValueError('source_path must have length at least two')

    grandparent_path = source_path[:-2]
    new_field = self._promote_helper(source_path, grandparent_path)
    new_path = grandparent_path + (new_name,)
    return self.with_updates({new_path: new_field})

  #=============================================================================
  # Properties
  #=============================================================================

  @property
  def rank(self):
    """The rank of this StructuredTensor.  Guaranteed not to be `None`."""
    return self._ragged_shape.rank

  @property
  def shape(self):
    """The static shape of this StructuredTensor.

    The returned `TensorShape` is guaranteed to have a known rank, but the
    individual dimension sizes may be unknown.

    Returns:
      `tf.TensorShape`
    """
    return self._ragged_shape._to_tensor_shape()  # pylint: disable=protected-access

  # TODO(martinz): for backwards compatibility
  @property
  def _row_partitions(self):
    """Deprecated form of row_partitions."""
    return self.row_partitions

  # TODO(edloper): Make this a func instead of a property?  Or make nrows
  # a property instead of a func?  Seems like these should be consistent.
  @property
  def row_partitions(self):
    """A tuple of `RowPartition`s defining the shape of this `StructuredTensor`.

    When `self.rank <= 1`, this tuple will be empty.

    When `self.rank > 1`, these `RowPartitions` define the shape of the
    `StructuredTensor` by describing how a flat (1D) list of structures can be
    repeatedly partitioned to form a higher-dimensional object.  In particular,
    the flat list is first partitioned into sublists using `row_partitions[-1]`,
    and then those sublists are further partitioned using `row_partitions[-2]`,
    etc.  The following examples show the row partitions used to describe
    several different `StructuredTensor`, each of which contains 8 copies of
    the same structure (`x`):

    >>> x = {'a': 1, 'b': ['foo', 'bar', 'baz']}       # shape = [] (scalar)

    >>> s1 = [[x, x, x, x], [x, x, x, x]]              # shape = [2, 4]
    >>> tf.experimental.StructuredTensor.from_pyval(s1).row_partitions
    (tf.RowPartition(row_splits=[0 4 8]),)

    >>> s2 = [[x, x], [x, x], [x, x], [x, x]]          # shape = [4, 2]
    >>> tf.experimental.StructuredTensor.from_pyval(s2).row_partitions
    (tf.RowPartition(row_splits=[0 2 4 6 8]),)

    >>> s3 = [[x, x, x], [], [x, x, x, x], [x]]        # shape = [2, None]
    >>> tf.experimental.StructuredTensor.from_pyval(s3).row_partitions
    (tf.RowPartition(row_splits=[0 3 3 7 8]),)

    >>> s4 = [[[x, x], [x, x]], [[x, x], [x, x]]]      # shape = [2, 2, 2]
    >>> tf.experimental.StructuredTensor.from_pyval(s4).row_partitions
    (tf.RowPartition(row_splits=[0 2 4]),
     tf.RowPartition(row_splits=[0 2 4 6 8]))


    >>> s5 = [[[x, x], [x]], [[x, x]], [[x, x], [x]]]  # shape = [3, None, None]
    >>> tf.experimental.StructuredTensor.from_pyval(s5).row_partitions
    (tf.RowPartition(row_splits=[0 2 3 5]),
     tf.RowPartition(row_splits=[0 2 3 5 7 8]))

    Note that shapes for nested fields (such as `x['b']` in the above example)
    are not considered part of the shape of a `StructuredTensor`, and are not
    included in `row_partitions`.

    If this `StructuredTensor` has a ragged shape (i.e., if any of the
    `row_partitions` is not uniform in size), then all fields will be encoded
    as either `RaggedTensor`s or `StructuredTensor`s with these `RowPartition`s
    used to define their outermost `self.rank` dimensions.

    Returns:
      A `tuple` of `RowPartition` objects with length `self.rank - 1`
      (or `0` if `self.rank < 2`)

    """
    if self.rank < 2:
      return ()
    return self._ragged_shape._as_row_partitions()  # pylint:disable=protected-access

  def nrows(self):
    """The number of rows in this StructuredTensor (if rank>0).

    This means the length of the outer-most dimension of the StructuredTensor.

    Notice that if `self.rank > 1`, then this equals the number of rows
    of the first row partition. That is,
    `self.nrows() == self.row_partitions[0].nrows()`.

    Otherwise `self.nrows()` will be the first dimension of the field values.

    Returns:
      A scalar integer `Tensor` (or `None` if `self.rank == 0`).
    """
    if self.rank == 0:
      return None
    return self._ragged_shape[0]

  def with_shape_dtype(self, dtype: dtypes.DType) -> 'StructuredTensor':
    if dtype == self._ragged_shape.dtype:
      return self
    return StructuredTensor(
        fields=_fields_with_dtype(self._fields, dtype),
        ragged_shape=self._ragged_shape.with_dtype(dtype))

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

    if self.rank == 0:
      return self._scalar_getitem(key)
    else:
      return self._tensor_getitem(key)

  def _scalar_getitem(self, key):
    if (isinstance(key[0], slice) and key[0].start is None and
        key[0].stop is None and key[0].step is None):
      fields = dict((field_name, field_value.__getitem__(key[1:]))
                    for (field_name, field_value) in self._fields.items())
      return StructuredTensor.from_fields(fields, self.shape)

    elif not isinstance(key[0], compat.bytes_or_text_types):
      raise ValueError('Key for indexing a StructuredTensor must be a '
                       "string or a full slice (':')")

    return self._fields[key[0]].__getitem__(key[1:])

  def _tensor_getitem(self, key):
    rank = self.rank
    if len(key) <= rank:
      new_fields = dict((field_name, field_value.__getitem__(key))
                        for (field_name, field_value) in self._fields.items())
      result_shape = self.shape.as_list()
      for d, k in enumerate(key):
        if isinstance(k, slice):
          if not (k.start is None and k.stop is None and k.step is None):
            # TODO(edloper): Better static shape analysis here.
            result_shape[d] = None
        elif isinstance(k, (int, tensor.Tensor)):
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
    fields = sorted(self._fields.items())
    fields = ((k, str(v).replace('\n', '\n            ')) for k, v in fields)
    fields = ('"{}": {}'.format(k, v) for k, v in fields)
    dict_repr = ',\n        '.join(fields)
    return ('<StructuredTensor(\n'
            '    fields={\n'
            '        %s},\n'
            '    shape=%s)>' % (dict_repr, self.shape))

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

    >>> tf.experimental.StructuredTensor.from_fields(
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
    if len(self.shape) > 0:  # pylint: disable=g-explicit-length-test
      if not result:  # special-case for StructuredTensors w/ no fields.
        return _empty_dict_pylist_from_row_partitions(self.row_partitions,
                                                      self.nrows())
      return _pyval_field_major_to_node_major(
          list(result.keys()), list(result.values()), self.rank)
    else:
      return result

  @classmethod
  def from_pyval(cls, pyval, typespec=None):
    """Constructs a StructuredTensor from a nested Python structure.

    >>> tf.experimental.StructuredTensor.from_pyval(
    ...     {'a': [1, 2, 3], 'b': [[4, 5], [6, 7]]})
    <StructuredTensor(
        fields={
          "a": tf.Tensor([1 2 3], shape=(3,), dtype=int32),
          "b": <tf.RaggedTensor [[4, 5], [6, 7]]>},
        shape=())>

    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.

    Args:
      pyval: The nested Python structure that should be used to create the new
        `StructuredTensor`.
      typespec: A `StructuredTensor.Spec` specifying the expected type for each
        field. If not specified, then all nested dictionaries are turned into
        StructuredTensors, and all nested lists are turned into Tensors (if
        rank<2) or RaggedTensors (if rank>=2).

    Returns:
      A `StructuredTensor`.
    """
    return cls._from_pyval(pyval, typespec, ())

  @classmethod
  def _from_pyval(cls, pyval, typespec, path_so_far):
    """Helper function for from_pyval.


    Args:
      pyval: The nested Python structure that should be used to create the new
        `StructuredTensor`.
      typespec: A `StructuredTensor.Spec` specifying the expected type for each
        field. If not specified, then all nested dictionaries are turned into
        StructuredTensors, and all nested lists are turned into Tensors (if
        rank<2) or RaggedTensors (if rank>=2).
      path_so_far: the path of fields that led here (for error messages).

    Returns:
      A `StructuredTensor`.
    """
    if isinstance(pyval, dict):
      return cls._from_pydict(pyval, typespec, path_so_far)
    elif isinstance(pyval, (list, tuple)):
      keys = set()
      rank = _pyval_find_struct_keys_and_depth(pyval, keys)
      if rank is not None:
        return cls._from_pylist_of_dict(pyval, keys, rank, typespec,
                                        path_so_far)
      else:
        return cls._from_pylist_of_value(pyval, typespec, path_so_far)
    else:
      return cls._from_pyscalar(pyval, typespec, path_so_far)

  @classmethod
  def _from_pydict(cls, pyval, typespec, path_so_far):
    """Converts python dictionary `pyval` to a StructuredTensor with rank=0."""
    if typespec is None:
      fields = dict((k, cls._from_pyval(v, None, path_so_far + (k,)))
                    for (k, v) in pyval.items())
    else:
      spec_shape = typespec._shape  # pylint: disable=protected-access
      field_specs = typespec._field_specs  # pylint: disable=protected-access
      if not (isinstance(typespec, StructuredTensor.Spec) and
              spec_shape.rank == 0 and set(pyval) == set(field_specs)):
        raise ValueError('Value at %r does not match typespec: %r vs %r' %
                         (path_so_far, pyval, typespec))
      fields = dict((k, cls._from_pyval(v, field_specs[k], path_so_far + (k,)))
                    for (k, v) in pyval.items())
    return StructuredTensor.from_fields(fields=fields, shape=(), validate=False)

  @classmethod
  def _from_pylist_of_dict(cls, pyval, keys, rank, typespec, path_so_far):
    """Converts python list `pyval` to a StructuredTensor with rank>1."""
    fields = dict((key, []) for key in keys)
    for child in pyval:
      _pyval_update_fields(child, fields, 1)
    if typespec is None:
      shape = tensor_shape.TensorShape([None] * rank)
      for (key, target) in fields.items():
        fields[key] = cls._from_pyval(target, None, path_so_far + (key,))
    else:
      field_specs = typespec._fields  # pylint: disable=protected-access
      if ((not isinstance(typespec, StructuredTensor.Spec)) or  # pylint: disable=superfluous-parens
          (set(fields) - set(field_specs))):
        raise ValueError('Value at %r does not match typespec: %r vs %r' %
                         (path_so_far, pyval, typespec))
      shape = typespec._shape
      if shape.rank < rank:
        raise ValueError('Value at %r does not match typespec (rank mismatch): '
                         '%r vs %r' % (path_so_far, pyval, typespec))
      for (key, spec) in field_specs.items():
        fields[key] = cls._from_pyval(
            fields.get(key, []), spec, path_so_far + (key,))
    try:
      if not fields and typespec is None:
        # TODO(b/183245576): handle cases where the typespec is known
        # but the dictionary is empty.
        return StructuredTensor._from_pylist_of_empty_dict(pyval, rank)
      return StructuredTensor.from_fields(
          fields=fields, shape=shape, validate=False)
    except Exception as exc:
      raise ValueError('Error parsing path %r' % (path_so_far,)) from exc

  @classmethod
  def _from_pylist_of_empty_dict(cls, pyval, rank):
    """Converts a pylist of empty dictionaries to StructuredTensors."""
    if rank == 0:
      return StructuredTensor.from_fields(fields={}, shape=(), validate=False)
    elif rank == 1:
      nrows = len(pyval)
      shape = (nrows,)
      return StructuredTensor.from_fields(fields={}, shape=shape, nrows=nrows)
    elif rank > 1:
      ragged_zeros = ragged_factory_ops.constant(_dicts_to_zeros(pyval))
      nrows = len(pyval)
      shape = tensor_shape.TensorShape([len(pyval)] + ([None] * (rank - 1)))
      return StructuredTensor.from_fields(
          fields={},
          shape=shape,
          row_partitions=ragged_zeros._nested_row_partitions,  # pylint:disable=protected-access
          nrows=nrows)

  @classmethod
  def _from_pylist_of_value(cls, pyval, typespec, path_so_far):
    """Converts python list `pyval` to a Tensor or RaggedTensor with rank>1."""
    if typespec is None:
      try:
        return ragged_factory_ops.constant(pyval)
      except Exception as exc:
        raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
    elif isinstance(typespec, tensor.TensorSpec):
      try:
        result = constant_op.constant(pyval, typespec.dtype)
      except Exception as exc:
        raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
      if not typespec.shape.is_compatible_with(result.shape):
        raise ValueError('Value at %r does not match typespec: %r vs %r' %
                         (path_so_far, typespec, pyval))
      return result
    elif isinstance(typespec, ragged_tensor.RaggedTensorSpec):
      # pylint: disable=protected-access
      try:
        return ragged_factory_ops.constant(
            pyval,
            dtype=typespec._dtype,
            ragged_rank=typespec._ragged_rank,
            row_splits_dtype=typespec._row_splits_dtype,
            inner_shape=typespec._shape[typespec._ragged_rank + 1:])
      except Exception as exc:
        raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
    elif isinstance(typespec, StructuredTensor.Spec):
      empty_rank = _pyval_empty_list_depth(pyval)
      if empty_rank is None:
        raise ValueError('Value at %r does not match typespec: %r vs %r' %
                         (path_so_far, typespec, pyval))
      else:
        return cls._from_pylist_of_dict(pyval, set(), empty_rank, typespec,
                                        path_so_far)
    else:
      raise ValueError('Value at %r does not match typespec: %r vs %r' %
                       (path_so_far, typespec, pyval))

  @classmethod
  def _from_pyscalar(cls, pyval, typespec, path_so_far):
    """Converts python scalar value `pyval` to a Tensor."""
    if typespec is None:
      try:
        return constant_op.constant(pyval)
      except Exception as exc:
        raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
    else:
      if not (isinstance(typespec, tensor.TensorSpec) and
              typespec.shape.rank == 0):
        raise ValueError('Value at %r does not match typespec: %r vs %r' %
                         (path_so_far, typespec, pyval))
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

    >>> st = tf.experimental.StructuredTensor.from_pyval(
    ...     [{'foo': 12}, {'foo': 33}, {'foo': 99}])
    >>> partition = RowPartition.from_row_lengths([2, 0, 1])
    >>> st.partition_outer_dimension(partition)
    <StructuredTensor(
      fields={
        "foo": <tf.RaggedTensor [[12, 33], [], [99]]>},
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

    >>> st = tf.experimental.StructuredTensor.from_pyval(
    ...     [[{'foo': 12}, {'foo': 33}], [], [{'foo': 99}]])
    >>> st.merge_dims(0, 1)
    <StructuredTensor(
      fields={
        "foo": tf.Tensor([12 33 99], shape=(3,), dtype=int32)},
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
    if not outer_axis <= inner_axis:
      raise ValueError('Expected outer_axis (%d) to be less than or equal to '
                       'inner_axis (%d)' % (outer_axis, inner_axis))
    return _merge_dims(self, outer_axis, inner_axis)

  class Spec:
    """A spec for StructuredTensor."""

    def __validate__(self):
      assert self._ragged_shape is not None

    @classmethod
    def _from_fields_and_rank(cls, fields, rank):
      """Creates a spec of a StructuredTensor with fields and rank."""
      shape = None
      for (k, v) in fields.items():
        field_shape_untruncated = _dynamic_ragged_shape_spec_from_spec(v)
        if field_shape_untruncated is None:
          raise ValueError(f'Cannot convert spec of {k}.')
        untruncated_rank = field_shape_untruncated.rank
        if (untruncated_rank is not None and untruncated_rank < rank):
          raise ValueError(f'Rank of field {k} is {untruncated_rank}, '
                           f'but must be at least {rank}.')
        field_shape = field_shape_untruncated._truncate(rank)  # pylint: disable=protected-access
        if shape is None:
          shape = field_shape
        else:
          shape = shape._merge_with(field_shape)
      return StructuredTensor.Spec(_ragged_shape=shape, _fields=fields)

    @classmethod
    def _from_shape(
        cls, shape: dynamic_ragged_shape.DynamicRaggedShape
    ) -> 'StructuredTensor.Spec':
      """Creates the spec of an empty StructuredTensor."""
      return StructuredTensor.Spec(_ragged_shape=shape, _fields={})

    # For backwards compatibility
    @property
    def _shape(self) -> tensor_shape.TensorShape:
      return self._ragged_shape._to_tensor_shape()  # pylint: disable=protected-access

    # For backwards compatibility
    @property
    def _field_specs(self) -> Dict[str, type_spec.TypeSpec]:
      return self._fields

    # For backwards compatibility
    @property
    def shape(self) -> tensor_shape.TensorShape:
      return self._shape

    # For backwards compatibility
    @property
    def rank(self):
      return self._ragged_shape.rank


# Regular expression used to determine whether a string is a valid field name.
# Note: we plan to relax (or possibly eliminate) this in the future; you
# should not rely on the fact that some field names are currently disallowed.
_FIELD_NAME_RE = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')

# =============================================================================
# Helper functions
# =============================================================================
# TODO(edloper): Move some of these helpers to row_partition.py?


def _convert_to_structured_field_value(value):
  """Converts `value` to a Tensor, RaggedTensor, or StructuredTensor."""
  if isinstance(value,
                (tensor.Tensor, ragged_tensor.RaggedTensor, StructuredTensor)):
    return value
  elif ragged_tensor.is_ragged(value):
    return ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
  elif isinstance(value, extension_type.ExtensionType):
    return value
  else:
    try:
      return ops.convert_to_tensor(value)
    except (ValueError, TypeError) as e:
      raise TypeError('Unexpected type for value in `fields`: %r' %
                      value) from e


def _find_shape_dtype(
    fields: Mapping[str, _FieldValue], nrows: Optional[tensor.Tensor],
    row_partitions: Optional[Sequence[RowPartition]]) -> dtypes.DType:
  """Return a consistent dtype for fields, nrows, & row_partitions.

  In the future, the default will switch from int64 to int32, but for now,
  we stick with int64.

  Args:
    fields: the fields of the StructuredTensor.
    nrows: the nrows of the StructuredTensor
    row_partitions: the row_partitions of the StructuredTensor.

  Returns:
    If anything requires int64, then return int64.
    If int32 is explicitly specified, return int32. Otherwise, return int64.
  """
  field_dtypes = [_field_shape_dtype(v) for v in fields.values()]
  nrows_dtypes = [nrows.dtype] if isinstance(nrows, tensor.Tensor) else []
  rp_dtypes = [] if row_partitions is None else [
      rp.dtype for rp in row_partitions
  ]

  all_dtypes = field_dtypes + nrows_dtypes + rp_dtypes

  if dtypes.int64 in all_dtypes:
    return dtypes.int64
  if dtypes.int32 in all_dtypes:
    return dtypes.int32

  # TODO(martinz): Eventually, shift this to tf.int32.
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
  if isinstance(value, tensor.Tensor):
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
        check_ops.assert_equal(
            nrows, value_nrows, message='fields have incompatible nrows')
    ], nrows)
  return nrows, static_nrows._merge_with(static_value_nrows)  # pylint: disable=protected-access


def _merge_row_partitions(row_partitions, value, rank, dtype, validate):
  """Merges `row_partitions` with `row_partitions(value)`."""
  if isinstance(value, tensor.Tensor):
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
        p1._merge_precomputed_encodings(p2, validate)  # pylint: disable=protected-access
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


def _pyval_field_major_to_node_major(keys, values, depth):
  """Regroup each field (k, v) from dict-of-list to list-of-dict.

  Given a "field-major" encoding of the StructuredTensor (which maps each key to
  a single nested list containing the values for all structs), return a
  corresponding "node-major" encoding, consisting of a nested list of dicts.

  Args:
    keys: The field names (list of string).  Must not be empty.
    values: The field values (list of python values).  Must have the same length
      as `keys`.
    depth: The list depth at which dictionaries should be created.

  Returns:
    A nested list of dict, with depth `depth`.
  """
  assert keys
  if depth == 0:
    return dict(zip(keys, values))
  nvals = len(values[0])
  assert all(nvals == len(values[i]) for i in range(1, len(values)))
  return [
      _pyval_field_major_to_node_major(keys, value_slice, depth - 1)
      for value_slice in zip(*values)
  ]


def _empty_dict_pylist_from_row_partitions(row_partitions, nrows):
  """Returns a python list of empty dicts from the given row partitions.

  Args:
    row_partitions: The row-partitions describing the ragged shape of the
      result.
    nrows: The number of rows in the outermost row-partition.  (Or if
      `len(row_partitions)==0`, then the number of empty dicts to return.)

  Returns:
    A nested python list whose leaves (if any) are empty python dicts.
  """
  if not row_partitions:
    return [{} for _ in range(nrows)]
  else:
    values = _empty_dict_pylist_from_row_partitions(
        row_partitions[1:], row_partitions[0].row_splits()[-1])
    splits = row_partitions[0].row_splits()
    return [values[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]


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
  if isinstance(value, tensor.Tensor) or not new_partitions:
    return value

  elif isinstance(value, ragged_tensor.RaggedTensor):
    return ragged_tensor.RaggedTensor._from_row_partition(  # pylint: disable=protected-access
        values=_replace_row_partitions(value.values, new_partitions[1:]),
        row_partition=new_partitions[0])

  else:
    assert isinstance(value, StructuredTensor)
    new_fields = dict((k, _replace_row_partitions(v, new_partitions))
                      for (k, v) in value._fields.items())
    return StructuredTensor._old_init(  # pylint: disable=protected-access
        fields=new_fields,
        shape=value.shape,
        nrows=value.nrows(),
        row_partitions=tuple(new_partitions) +
        tuple(value.row_partitions[len(new_partitions):]))


def _partition_outer_dimension(value, row_partition):
  """Partitions the outer dimension of `value` using `row_partitions`.

  Examples:

    >>> partition = RowPartition.from_row_lengths([2, 0, 1])
    >>> _partition_outer_dimension(tf.constant([1, 2, 3]), partition)
    <tf.RaggedTensor [[1, 2], [], [3]]>

    >>> struct_value = tf.experimental.StructuredTensor.from_pyval(
    ...     [{'x': 1}, {'x': 2}, {'x': 3}])
    >>> _partition_outer_dimension(struct_value, partition)
    <StructuredTensor(
      fields={
        "x": <tf.RaggedTensor [[1, 2], [], [3]]>},
      shape=(3, None))>

  Args:
    value: Tensor, RaggedTensor, or StructuredTensor
    row_partition: RowPartition

  Returns:
    A value with the same type as `value`, where
    `result.rank = value.rank + 1`.
  """
  is_ragged = row_partition.uniform_row_length() is None
  if isinstance(value, tensor.Tensor) and not is_ragged:
    new_shape = array_ops.concat(
        [[row_partition.nrows(),
          row_partition.uniform_row_length()],
         array_ops.shape(value, out_type=row_partition.dtype)[1:]],
        axis=0)
    return array_ops.reshape(value, new_shape)
  elif isinstance(value, (tensor.Tensor, ragged_tensor.RaggedTensor)):
    return ragged_tensor.RaggedTensor._from_row_partition(  # pylint: disable=protected-access
        value, row_partition)
  else:
    assert isinstance(value, StructuredTensor)
    nrows = row_partition.static_nrows
    ncols = row_partition.static_uniform_row_length
    shape = tensor_shape.TensorShape([nrows,
                                      ncols]).concatenate(value.shape[1:])
    fields = dict((k, _partition_outer_dimension(v, row_partition))
                  for (k, v) in value._fields.items())
    return StructuredTensor._old_init(  # pylint: disable=protected-access
        fields, shape, row_partition.nrows(),
        (row_partition,) + value.row_partitions)


def _merge_dims(value, outer_axis, inner_axis):
  """Merges `outer_axis...inner_axis` of `value` into a single dimension."""
  assert outer_axis < inner_axis
  if isinstance(value, (tensor.Tensor, ragged_tensor.RaggedTensor)):
    return ragged_tensor.merge_dims(value, outer_axis, inner_axis)
  else:
    assert isinstance(value, StructuredTensor)
    fields = dict((k, _merge_dims(v, outer_axis, inner_axis))
                  for (k, v) in value._fields.items())
    ragged_shape = value._ragged_shape._merge_dims(  # pylint: disable=protected-access
        outer_axis, inner_axis)
    return StructuredTensor(fields, ragged_shape)


_structured_tensor_factory_key = object()  # unique private object


def _dynamic_ragged_shape_spec_from_spec(
    spec: Union[dynamic_ragged_shape.DynamicRaggedShape.Spec,
                ragged_tensor.RaggedTensorSpec, StructuredTensor.Spec,
                tensor.TensorSpec]
) -> dynamic_ragged_shape.DynamicRaggedShape.Spec:
  if isinstance(spec, StructuredTensor.Spec):
    return spec._ragged_shape  # pylint: disable=protected-access
  else:
    return dynamic_ragged_shape.DynamicRaggedShape.Spec._from_spec(spec)  # pylint: disable=protected-access


def _normalize_field_name_to_tuple(name: 'FieldName') -> Sequence[str]:
  """FieldName can be given also as string, this normalizes it to a tuple."""
  if isinstance(name, str):
    return (name,)
  if isinstance(name, list):
    return tuple(name)
  assert isinstance(name, tuple)
  return name


def _dicts_to_zeros(pyval):
  """Replaces dictionaries zeros in a pylist."""
  if isinstance(pyval, dict):
    return 0
  return [_dicts_to_zeros(x) for x in pyval]


def _merge_dims_generic(source, outer, inner):
  """Merges outer_axis...inner_axis into a single dimension.

  If outer == inner, this is a NOOP. If inner < outer, then this fails.
  If inner >= source.shape.rank, then the behavior is undefined.

  Args:
    source: a tensor, ragged tensor, or structured tensor.
    outer: a python int, indicating the first dimension to compress (must be
      nonnegative).
    inner: a python int, indicating the first dimension to keep (of the tail)
      (must be nonnegative).

  Returns:
    source with outer_axis...inner_axis merged into a single dimension.
  """
  if isinstance(source, StructuredTensor):
    return source.merge_dims(outer, inner)
  else:
    return ragged_tensor.merge_dims(source, outer, inner)


def _dynamic_ragged_shape_from_tensor(
    field, dtype=None) -> dynamic_ragged_shape.DynamicRaggedShape:
  """Extension of DynamicRaggedShape.from_tensor to support StructuredTensor."""
  if isinstance(field, StructuredTensor):
    return field._ragged_shape  # pylint: disable=protected-access
  shape = array_ops.shape_v2(field, out_type=dtype)

  if isinstance(shape, tensor.Tensor):
    return dynamic_ragged_shape.DynamicRaggedShape(
        row_partitions=[], inner_shape=shape)
  elif isinstance(shape, dynamic_ragged_shape.DynamicRaggedShape):
    return shape
  # TODO(martinz): add a test for the following line.
  raise TypeError(f'Expected shape tf.shape({field}) to return a Tensor or a '
                  f'DynamicRaggedShape. Instead, got: {shape}.')


def _merge_with_optional(
    a: Optional[dynamic_ragged_shape.DynamicRaggedShape],
    b: Optional[dynamic_ragged_shape.DynamicRaggedShape]
) -> Optional[dynamic_ragged_shape.DynamicRaggedShape]:
  if a is None:
    return b
  if b is None:
    return a
  return a._merge_with(b)  # pylint: disable=protected-access


def _shape_from_fields(
    fields, rank: int,
    dtype: dtypes.DType) -> Optional[dynamic_ragged_shape.DynamicRaggedShape]:
  """Given fields, rank, and dtype, create a shape."""

  field_shape = None
  for (k, field) in fields.items():
    try:
      next_field_shape_raw = _dynamic_ragged_shape_from_tensor(
          field, dtype=dtype)
      next_field_shape = next_field_shape_raw[:rank]
      field_shape = _merge_with_optional(field_shape, next_field_shape)
    except Exception as err:
      raise ValueError(f'Error in shape of {k}') from err

  return field_shape


def _field_shape_dtype(field: _FieldValue) -> Optional[dtypes.DType]:
  if isinstance(field, ragged_tensor.RaggedTensor):
    return field._row_partition.dtype  # pylint: disable=protected-access
  if isinstance(field, StructuredTensor):
    return field._ragged_shape.dtype  # pylint: disable=protected-access
  return None


def _field_with_shape_dtype(field: _FieldValue,
                            dtype: dtypes.DType) -> _FieldValue:
  if isinstance(field, ragged_tensor.RaggedTensor):
    return field.with_row_splits_dtype(dtype)
  if isinstance(field, StructuredTensor):
    return field.with_shape_dtype(dtype)

  return field


def _fields_with_dtype(fields: Mapping[str, _FieldValue],
                       dtype: dtypes.DType) -> Mapping[str, _FieldValue]:
  return {k: _field_with_shape_dtype(v, dtype) for (k, v) in fields.items()}


# pylint:disable=protected-access
def _dynamic_ragged_shape_init(fields, shape, nrows, row_partitions):
  """Produce a DynamicRaggedShape for StructuredTensor."""
  assert isinstance(fields, dict), fields
  assert isinstance(shape, tensor_shape.TensorShape), shape
  assert nrows is None or isinstance(nrows, tensor.Tensor) or isinstance(
      nrows, int), nrows
  assert row_partitions is None or isinstance(row_partitions,
                                              tuple), row_partitions
  rank = shape.rank

  if rank is None:
    raise TypeError("StructuredTensor's shape must have known rank.")

  # TODO(martinz): figure out whether to validate.
  dtype = _find_shape_dtype(fields, nrows, row_partitions)

  fields = _fields_with_dtype(fields, dtype)

  result = None
  if shape.is_fully_defined():
    result = dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(
        shape.as_list(), dtype=dtype)

  if rank == 0:
    return dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(
        array_ops.zeros((0,), dtype=dtype))

  result = _merge_with_optional(result, _shape_from_fields(fields, rank, dtype))
  if rank == 1:
    alt_value = tensor_shape.dimension_value(shape[0])
    if alt_value is not None:
      nrows = alt_value
    if nrows is not None:
      result = _merge_with_optional(
          result,
          dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(
              [nrows], dtype=dtype))
    if result is None:
      raise ValueError('Must specify `nrows`, a fully specified `shape`,' +
                       ' or have `fields` if `rank=1`')

    return result

  if row_partitions:
    result = _merge_with_optional(
        result,
        dynamic_ragged_shape.DynamicRaggedShape.from_row_partitions(
            row_partitions, dtype=dtype))

  if result is None:
    raise ValueError('Must specify row_partitions, a fully specified shape, ' +
                     'or have fields if rank > 1')
  return result


# TODO(martinz): Drop this method or rename.
def StructuredTensorSpec(shape, field_specs):  # pylint:disable=invalid-name
  """A placeholder for the old StructuredTensorSpec."""
  if not isinstance(field_specs, dict):
    raise TypeError('field_specs must be a dictionary.')
  for k in field_specs.keys():
    if not isinstance(k, str):
      raise TypeError('field_specs must be a dictionary with string keys.')
  for v in field_specs.values():
    if not isinstance(v, type_spec.TypeSpec):
      raise TypeError('field_specs must be a dictionary with TypeSpec values.')

  shape = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(
      tensor_shape.as_shape(shape), 0, dtypes.int32)
  rank = shape.rank
  if rank is None:
    raise TypeError("StructuredTensor's shape must have known rank.")
  for (k, v) in field_specs.items():
    field_shape_untruncated = _dynamic_ragged_shape_spec_from_spec(v)
    if field_shape_untruncated is None:
      raise ValueError(f'Cannot convert spec of {k}.')
    untruncated_rank = field_shape_untruncated.rank
    if (untruncated_rank is not None and untruncated_rank < rank):
      raise ValueError(f'Rank of field {k} is {untruncated_rank},'
                       f' but must be at least {rank}.')
    field_shape = field_shape_untruncated._truncate(rank)
    shape = shape._merge_with(field_shape)
  return StructuredTensor.Spec(_ragged_shape=shape, _fields=field_specs)
