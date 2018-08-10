import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.lookup import lookup_ops

class PartitionedMutableHashTable(object):
  """Partitioned mutable hash table

  Usage example:
  """

  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               shard_num=1,
               shared_name=None,
               name="PartitionedMutableHashTable",
               checkpoint=True,
               initializer=None):
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._shard_num = shard_num
    self._table_ref_list = []
    self._default_value = default_value
    for i in range(shard_num):
      self._table_ref_list.append(lookup_ops.MutableHashTable(key_dtype=key_dtype,
      											   value_dtype=value_dtype,
      											   default_value=default_value,
      											   shared_name=shared_name,
      											   name='%s_part%d' % (name, i),
      											   checkpoint=checkpoint))

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys of smaller count than the `key_count_threhold` in the
    count table. If the counter of a key is larger than `key_count_threhold`, this key and its
    new initialized value will be inserted in the table firstly if not found in the table.

    Args:
      keys: Keys to look up. Can be a tensor of list shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    original_indices = math_ops.range(array_ops.size(keys))
    if keys.dtype == tf.string:
      int_keys = tf.string_to_hash_bucket_fast(keys, tf.int64.max)
    elif keys.dtype == tf.int64 or keys.dtype == tf.int32:
      int_keys = keys
    key_assignments = int_keys % self._shard_num
    if key_assignments.dtype != tf.int32:
      key_assignments = tf.cast(key_assignments, dtype=tf.int32)
    key_partitions = data_flow_ops.dynamic_partition(keys, key_assignments, self._shard_num)
    indice_partitions = data_flow_ops.dynamic_partition(original_indices, key_assignments, self._shard_num)
    partitioned_result = []
    for i in range(self._shard_num):
      partitioned_result.append(self._table_ref_list[i].lookup(key_partitions[i]))
    ret = data_flow_ops.dynamic_stitch(indice_partitions, partitioned_result)
    return ret

  def contain(self, keys, name=None):
    """Looks up `keys` in a table, outputs true of false flags.

    Args:
      keys: Keys to look up. Can be a tensor of list shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the bool values in the same shape as `keys`.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    original_indices = math_ops.range(array_ops.size(keys))
    if keys.dtype == tf.string:
      int_keys = tf.string_to_hash_bucket_fast(keys, tf.int64.max)
    elif keys.dtype == tf.int64 or keys.dtype == tf.int32:
      int_keys = keys
    key_assignments = int_keys % self._shard_num
    if key_assignments.dtype != tf.int32:
      key_assignments = tf.cast(key_assignments, dtype=tf.int32)
    key_partitions = data_flow_ops.dynamic_partition(keys, key_assignments, self._shard_num)
    indice_partitions = data_flow_ops.dynamic_partition(original_indices, key_assignments, self._shard_num)
    partitioned_result = []
    for i in range(self._shard_num):
      partitioned_result.append(self._table_ref_list[i].contain(key_partitions[i]))
    ret = data_flow_ops.dynamic_stitch(indice_partitions, partitioned_result)
    return ret

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    # TODO. colocate_with error
    #with ops.colocate_with(self._table_ref_list[0]):
    #  return self._table_ref_list[0].size()
    size = 0
    for i in range(self._shard_num):
        size += self._table_ref_list[i].size()
    return size

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the
        table's key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The list of the created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    original_indices = math_ops.range(array_ops.size(keys))
    if keys.dtype == tf.string:
      int_keys = tf.string_to_hash_bucket_fast(keys, tf.int64.max)
    elif keys.dtype == tf.int64 or keys.dtype == tf.int32:
      int_keys = keys
    key_assignments = int_keys % self._shard_num
    if key_assignments.dtype != tf.int32:
      key_assignments = tf.cast(key_assignments, dtype=tf.int32)
    key_partitions = data_flow_ops.dynamic_partition(keys, key_assignments, self._shard_num)
    value_partitions = data_flow_ops.dynamic_partition(values, key_assignments, self._shard_num)
    partitioned_ops = []
    for i in range(self._shard_num):
      partitioned_ops.append(self._table_ref_list[i].insert(key_partitions[i], value_partitions[i]))
    return partitioned_ops
