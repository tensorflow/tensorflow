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
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
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
    rank = shape.ndims
    if rank is None:
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

    See `tf.RaggedTensor` for more information about ragged dimensions and
    `ragged_rank`.

    Returns:
      A Python `int` indicating the number of ragged dimensions in this ragged
      tensor.  The outermost dimension is not considered ragged.
    """
    return len(self._nested_row_splits)

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

    if self._static_shape.ndims == 0:
      return self._scalar_getitem(key)
    else:
      return self._tensor_getitem(key)

  def _scalar_getitem(self, key):
    if (isinstance(key[0], slice) and slice.start is None and
        slice.stop is None and slice.step is None):
      fields = dict((field_name, field_value.__getitem__(key[1:]))
                    for (field_name, field_value) in self._fields.items())
      return StructuredTensor(self._static_shape[1:], fields)

    elif not isinstance(key[0], compat.bytes_or_text_types):
      raise ValueError('Key for indexing a StructuredTensor must be a '
                       "string or a full slice (':')")

    return self._fields[key[0]].__getitem__(key[1:])

  def _tensor_getitem(self, key):
    rank = self._static_shape.ndims
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
        else:
          # Ellipsis, tf.newaxis:
          raise ValueError('Slicing not supported for %r' % k)
      result_shape = [d for d in result_shape if d != -1]
      return StructuredTensor(result_shape, new_fields)

    else:
      if not isinstance(key[rank], compat.bytes_or_text_types):
        raise ValueError('Key for indexing a StructuredTensor must be a string')
      return self._fields[key[rank]].__getitem__(key[:rank] + key[rank + 1:])

  def __repr__(self):
    return '<StructuredTensor(shape=%s, fields=%r)>' % (self._static_shape,
                                                        self._fields)

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

    >>> print(StructuredTensor([3], {'a': [1, 2, 3]}).to_pyval())
    [{b'a': 1}, {b'a': 2}, {b'a': 3}]

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
    if len(self._static_shape) > 0:  # pylint: disable=g-explicit-length-test
      return _pyval_field_major_to_node_major(list(result.keys()),
                                              list(result.values()),
                                              self._static_shape.as_list())
    else:
      return result

  @classmethod
  def from_pyval(cls, pyval, typespec=None):
    """Constructs a StructuredTensor from a nested Python structure.

    >>> print StructuredTensor.from_pyval(
    ...     {'a': [1, 2, 3], 'b': [[4, 5], [6, 7]]})
    <StructuredTensor {'a': [1, 2, 3], 'b': [[4, 5], [6, 7]]}>

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
              spec_shape.ndims == 0 and set(pyval) == set(field_specs)):
        raise ValueError('Value does not match typespec: %r vs %r' %
                         (pyval, typespec))
      fields = dict(
          (k, cls.from_pyval(v, field_specs[k])) for (k, v) in pyval.items())
    return StructuredTensor(shape=(), fields=fields)

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
        if not spec.is_compatible_with(fields[key]):
          raise ValueError('Value does not match typespec: %r vs %r' %
                           (spec, fields[key]))
    return StructuredTensor(shape=shape, fields=fields)

  @classmethod
  def _from_pylist_of_value(cls, pyval, typespec):
    """Converts python list `pyval` to a Tensor or RaggedTensor with rank>1."""
    if typespec is None:
      return ragged_factory_ops.constant(pyval)
    elif isinstance(typespec, tensor_spec.TensorSpec):
      # TODO(edloper): Check that typespec.shape matches.
      return constant_op.constant(pyval, typespec.dtype)
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
              typespec.shape.ndims == 0):
        raise ValueError('Value does not match typespec.')
      # TODO(edloper): Check that typespec.shape matches.
      return constant_op.constant(pyval, typespec.dtype)

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
