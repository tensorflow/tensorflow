# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Lookup table operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
# pylint: disable=unused-import
from tensorflow.python.ops.lookup_ops import FastHashSpec
from tensorflow.python.ops.lookup_ops import HasherSpec
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import IdTableWithHashBuckets
from tensorflow.python.ops.lookup_ops import index_table_from_file
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow.python.ops.lookup_ops import InitializableLookupTableBase
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer
from tensorflow.python.ops.lookup_ops import LookupInterface
from tensorflow.python.ops.lookup_ops import StrongHashSpec
from tensorflow.python.ops.lookup_ops import TableInitializerBase
from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer
from tensorflow.python.ops.lookup_ops import TextFileIndex
from tensorflow.python.ops.lookup_ops import TextFileInitializer
from tensorflow.python.ops.lookup_ops import TextFileStringTableInitializer
# pylint: enable=unused-import
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util.deprecation import deprecated


@deprecated("2017-04-10", "Use `index_table_from_file`.")
def string_to_index_table_from_file(vocabulary_file=None,
                                    num_oov_buckets=0,
                                    vocab_size=None,
                                    default_value=-1,
                                    hasher_spec=FastHashSpec,
                                    name=None):
  return index_table_from_file(
      vocabulary_file, num_oov_buckets, vocab_size, default_value, hasher_spec,
      key_dtype=dtypes.string, name=name)


@deprecated("2017-04-10", "Use `index_table_from_tensor`.")
def string_to_index_table_from_tensor(mapping,
                                      num_oov_buckets=0,
                                      default_value=-1,
                                      hasher_spec=FastHashSpec,
                                      name=None):
  with ops.name_scope(name, "string_to_index") as scope:
    mapping = ops.convert_to_tensor(mapping)
  if dtypes.string != mapping.dtype.base_dtype:
    raise ValueError("string_to_index_table_from_tensor requires string.")
  return index_table_from_tensor(
      mapping, num_oov_buckets, default_value, hasher_spec, name=scope)


def index_table_from_tensor(mapping,
                            num_oov_buckets=0,
                            default_value=-1,
                            hasher_spec=FastHashSpec,
                            dtype=dtypes.string,
                            name=None):
  """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a string `mapping` 1-D tensor
  where each element is a key and corresponding index within the tensor is the
  value.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is `[mapping size, mapping size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` or `table.initializer.run()` once.

  Elements in `mapping` cannot have duplicates, otherwise when executing the
  table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  mapping_strings = tf.constant(["emerson", "lake", "palmer"])
  table = tf.contrib.lookup.index_table_from_tensor(
      mapping=mapping_strings, num_oov_buckets=1, default_value=-1)
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = table.lookup(features)
  ...
  tf.tables_initializer().run()

  ids.eval()  ==> [0, 1, 3, 2]
  ```

  Args:
    mapping: A 1-D `Tensor` that specifies the mapping of keys to indices. The
      type of this object must be castable to `dtype`.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignment of out-of-vocabulary buckets.
    dtype: The type of values passed to `lookup`. Only string and integers are
      supported.
    name: A name for this op (optional).

  Returns:
    The lookup table to map an input `Tensor` to index `int64` `Tensor`.

  Raises:
    ValueError: If `mapping` is invalid.
    ValueError: If `num_oov_buckets` is negative.
  """
  if mapping is None:
    raise ValueError("mapping must be specified.")
  return lookup_ops.index_table_from_tensor(
      vocabulary_list=mapping,
      num_oov_buckets=num_oov_buckets,
      default_value=default_value,
      hasher_spec=hasher_spec,
      dtype=dtype,
      name=name)


@deprecated(
    "2017-01-07", "This op will be removed after the deprecation date. "
    "Please switch to index_table_from_tensor and call the lookup "
    "method of the returned table.")
def string_to_index(tensor, mapping, default_value=-1, name=None):
  """Maps `tensor` of strings into `int64` indices based on `mapping`.

  This operation converts `tensor` of strings into `int64` indices.
  The mapping is initialized from a string `mapping` tensor where each element
  is a key and corresponding index within the tensor is the value.

  Any entry in the input which does not have a corresponding entry in 'mapping'
  (an out-of-vocabulary entry) is assigned the `default_value`

  Elements in `mapping` cannot be duplicated, otherwise the initialization
  will throw a FailedPreconditionError.

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` once.

  For example:

  ```python
  mapping_strings = tf.constant(["emerson", "lake", "palmer"])
  feats = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = tf.contrib.lookup.string_to_index(
      feats, mapping=mapping_strings, default_value=-1)
  ...
  tf.tables_initializer().run()

  ids.eval()  ==> [0, 1, -1, 2]
  ```

  Args:
    tensor: A 1-D input `Tensor` with the strings to map to indices.
    mapping: A 1-D string `Tensor` that specifies the mapping of strings to
      indices.
    default_value: The `int64` value to use for out-of-vocabulary strings.
      Defaults to -1.
    name: A name for this op (optional).

  Returns:
    The mapped indices. It has the same shape and tensor type (dense or sparse)
    as `tensor`.
  """
  table = index_table_from_tensor(
      mapping=mapping, default_value=default_value, name=name)
  return table.lookup(tensor)


def index_to_string_table_from_tensor(mapping, default_value="UNK", name=None):
  """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The mapping is initialized from a string `mapping` 1-D `Tensor` where
  each element is a value and the corresponding index within the tensor is the
  key.

  Any input which does not have a corresponding index in 'mapping'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` or `table.initializer.run()` once.

  Elements in `mapping` cannot have duplicates, otherwise when executing the
  table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  mapping_string = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      mapping_string, default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    mapping: A 1-D string `Tensor` that specifies the strings to map from
      indices.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `mapping` is not set.
  """

  if mapping is None:
    raise ValueError("mapping must be specified.")

  return lookup_ops.index_to_string_table_from_tensor(
      vocabulary_list=mapping, default_value=default_value, name=name)


@deprecated(
    "2017-01-07", "This op will be removed after the deprecation date. "
    "Please switch to index_to_string_table_from_tensor and call the lookup "
    "method of the returned table.")
def index_to_string(tensor, mapping, default_value="UNK", name=None):
  """Maps `tensor` of indices into string values based on `mapping`.

  This operation converts `int64` indices into string values. The mapping is
  initialized from a string `mapping` tensor where each element is a value and
  the corresponding index within the tensor is the key.

  Any input which does not have a corresponding index in 'mapping'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `tf.tables_initializer.run()` once.

  For example:

  ```python
  mapping_string = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  values = tf.contrib.lookup.index_to_string(
      indices, mapping=mapping_string, default_value="UNKNOWN")
  ...
  tf.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    tensor: A `int64` `Tensor` with the indices to map to strings.
    mapping: A 1-D string `Tensor` that specifies the strings to map from
      indices.
    default_value: The string value to use for out-of-vocabulary indices.
    name: A name for this op (optional).

  Returns:
    The strings values associated to the indices. The resultant dense
    feature value tensor has the same shape as the corresponding `indices`.
  """
  table = index_to_string_table_from_tensor(
      mapping=mapping, default_value=default_value, name=name)
  return table.lookup(tensor)


class MutableHashTable(LookupInterface):
  """A generic mutable hash table implementation.

  Data can be inserted by calling the insert method and removed by calling the
  remove method. It does not support initialization via the init method.

  Example usage:

  ```python
  table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                             value_dtype=tf.int64,
                                             default_value=-1)
  sess.run(table.insert(keys, values))
  out = table.lookup(query_keys)
  print(out.eval())
  ```
  """

  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               shared_name=None,
               name="MutableHashTable",
               checkpoint=True):
    """Creates an empty `MutableHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      shared_name: If non-empty, this table will be shared under
        the given name across multiple sessions.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.

    Returns:
      A `MutableHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(default_value,
                                                dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._name = name

    if context.executing_eagerly() and shared_name is None:
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      shared_name = "table_%d" % (ops.uid(),)
    self._shared_name = shared_name
    super(MutableHashTable, self).__init__(key_dtype, value_dtype)

    self._resource_handle = self.create_resource()
    if checkpoint:
      saveable = MutableHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def create_resource(self):
    # The table must be shared if checkpointing is requested for multi-worker
    # training to work correctly. Use the node name if no shared_name has been
    # explicitly specified.
    use_node_name_sharing = self._checkpoint and self._shared_name is None
    if self._default_value.get_shape().ndims == 0:
      table_ref = gen_lookup_ops.mutable_hash_table_v2(
          shared_name=self._shared_name,
          use_node_name_sharing=use_node_name_sharing,
          key_dtype=self._key_dtype,
          value_dtype=self._value_dtype,
          name=self._name)
    else:
      table_ref = gen_lookup_ops.mutable_hash_table_of_tensors_v2(
          shared_name=self._shared_name,
          use_node_name_sharing=use_node_name_sharing,
          key_dtype=self._key_dtype,
          value_dtype=self._value_dtype,
          value_shape=self._default_value.get_shape(),
          name=self._name)

    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref

  @property
  def name(self):
    return self._table_name

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self.name,
                        [self.resource_handle]) as name:
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(
            self.resource_handle, name=name)

  def remove(self, keys, name=None):
    """Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    if keys.dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    with ops.name_scope(
        name, "%s_lookup_table_remove" % self.name,
        (self.resource_handle, keys, self._default_value)) as name:
      # pylint: disable=protected-access
      op = gen_lookup_ops.lookup_table_remove_v2(
          self.resource_handle, keys, name=name)

    return op

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(
        name, "%s_lookup_table_find" % self.name,
        (self.resource_handle, keys, self._default_value)) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(
            self.resource_handle, keys, self._default_value, name=name)
    return values

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the
        table's key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_insert" % self.name,
                        [self.resource_handle, keys, values]) as name:
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_insert_v2(
            self.resource_handle, keys, values, name=name)
    return op

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self.name,
                        [self.resource_handle]) as name:
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self.resource_handle, self._key_dtype, self._value_dtype, name=name)
    return exported_keys, exported_values

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    return {"table": functools.partial(MutableHashTable._Saveable, table=self)}

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for MutableHashTable."""

    def __init__(self, table, name):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      # pylint: disable=protected-access
      super(MutableHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.colocate_with(self.op.resource_handle):
        return gen_lookup_ops.lookup_table_import_v2(
            self.op.resource_handle, restored_tensors[0], restored_tensors[1])


class MutableDenseHashTable(LookupInterface):
  """A generic mutable hash table implementation using tensors as backing store.

  Data can be inserted by calling the insert method and removed by calling the
  remove method. It does not support initialization via the init method.

  It uses "open addressing" with quadratic reprobing to resolve collisions.
  Compared to `MutableHashTable` the insert, remove and lookup operations in a
  `MutableDenseHashTable` are typically faster, but memory usage can be higher.
  However, `MutableDenseHashTable` does not require additional memory for
  temporary tensors created during checkpointing and restore operations.

  Example usage:

  ```python
  table = tf.contrib.lookup.MutableDenseHashTable(key_dtype=tf.int64,
                                                  value_dtype=tf.int64,
                                                  default_value=-1,
                                                  empty_key=0,
                                                  deleted_key=-1)

  sess.run(table.insert(keys, values))
  out = table.lookup(query_keys)
  print(out.eval())
  ```
  """

  # TODO(andreasst): consider extracting common code with MutableHashTable into
  # a common superclass.
  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               empty_key,
               deleted_key,
               initial_num_buckets=None,
               shared_name=None,
               name="MutableDenseHashTable",
               checkpoint=True):
    """Creates an empty `MutableDenseHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      empty_key: the key to use to represent empty buckets internally. Must not
        be used in insert, remove or lookup operations.
      initial_num_buckets: the initial number of buckets.
      shared_name: If non-empty, this table will be shared under
        the given name across multiple sessions.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.
      deleted_key: the key to use to represent deleted buckets internally. Must
        not be used in insert, remove or lookup operations and be different from
        the empty_key.

    Returns:
      A `MutableDenseHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=value_dtype, name="default_value")
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._initial_num_buckets = initial_num_buckets
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._name = name

    self._empty_key = ops.convert_to_tensor(
        empty_key, dtype=key_dtype, name="empty_key")
    self._deleted_key = ops.convert_to_tensor(
        deleted_key, dtype=key_dtype, name="deleted_key")
    if context.executing_eagerly() and shared_name is None:
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      shared_name = "table_%d" % (ops.uid(),)
    self._shared_name = shared_name
    super(MutableDenseHashTable, self).__init__(key_dtype, value_dtype)

    self._resource_handle = self.create_resource()
    if checkpoint:
      saveable = MutableDenseHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def create_resource(self):
    # The table must be shared if checkpointing is requested for multi-worker
    # training to work correctly. Use the node name if no shared_name has been
    # explicitly specified.
    use_node_name_sharing = self._checkpoint and self._shared_name is None
    table_ref = gen_lookup_ops.mutable_dense_hash_table_v2(
        empty_key=self._empty_key,
        deleted_key=self._deleted_key,
        shared_name=self._shared_name,
        use_node_name_sharing=use_node_name_sharing,
        value_dtype=self._value_dtype,
        value_shape=self._value_shape,
        initial_num_buckets=self._initial_num_buckets,
        name=self._name)
    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref

  @property
  def name(self):
    return self._table_name

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self.name,
                        [self.resource_handle]) as name:
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(
            self.resource_handle, name=name)

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(name, "%s_lookup_table_find" % self.name,
                        [self.resource_handle, keys]) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(
            self.resource_handle, keys, self._default_value, name=name)

    return values

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the
        table's key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_insert" % self.name,
                        [self.resource_handle, keys, values]) as name:
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      values = ops.convert_to_tensor(
          values, dtype=self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        op = gen_lookup_ops.lookup_table_insert_v2(
            self.resource_handle, keys, values, name=name)
      return op

  def remove(self, keys, name=None):
    """Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    if keys.dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    with ops.name_scope(
        name, "%s_lookup_table_remove" % self.name,
        (self.resource_handle, keys, self._default_value)) as name:
      # pylint: disable=protected-access
      op = gen_lookup_ops.lookup_table_remove_v2(
          self.resource_handle, keys, name=name)

    return op

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self.name,
                        [self.resource_handle]) as name:
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self.resource_handle, self._key_dtype, self._value_dtype, name=name)

    return exported_keys, exported_values

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    return {"table": functools.partial(
        MutableDenseHashTable._Saveable, table=self)}

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for MutableDenseHashTable."""

    def __init__(self, table, name):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      # pylint: disable=protected-access
      super(MutableDenseHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.colocate_with(self.op.resource_handle):
        return gen_lookup_ops.lookup_table_import_v2(
            self.op.resource_handle, restored_tensors[0], restored_tensors[1])
