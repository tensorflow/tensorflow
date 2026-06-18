# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""A wrapper for gen_simple_hash_table_op.py.

This defines a public API and provides a docstring for the C++ Op defined by
simple_hash_table_kernel.cc
"""

import tensorflow as tf
from tensorflow.examples.custom_ops_doc.simple_hash_table.simple_hash_table_op import gen_simple_hash_table_op


class SimpleHashTable(tf.saved_model.experimental.TrackableResource):
  """A simple mutable hash table implementation.

  Implement a simple hash table as a Resource using ref-counting.
  This demonstrates a Stateful Op for a general Create/Read/Update/Delete
  (CRUD) style use case.  To instead make an op for a specific lookup table
  case, it is preferable to follow the implementation style of
  TensorFlow's internal ops, e.g. use LookupInterface.

  Data can be inserted by calling the `insert` method and removed by calling
  the `remove` method. It does not support initialization via the init method.

  The `import` and `export` methods allow loading and restoring all of
  the key, value pairs. These methods (or their corresponding kernels)
  are intended to be used for supporting SavedModel.

  Example usage:
    hash_table = simple_hash_table_op.SimpleHashTable(key_dtype, value_dtype,
                                                      111)
    result1 = hash_table.find(1, -999)  # -999
    hash_table.insert(1, 100)
    result2 = hash_table.find(1, -999)  # 100
    hash_table.remove(1)
    result3 = hash_table.find(1, -999)  # -999
  """

  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               name="SimpleHashTable"):
    """Creates an empty `SimpleHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      name: A name for the operation (optional).

    Returns:
      A `SimpleHashTable` object.
    """
    super(SimpleHashTable, self).__init__()
    self._default_value = tf.convert_to_tensor(default_value, dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._name = name
    self._resource_handle = self._create_resource()
    # Methods that use the Resource get its handle using the
    # public self.resource_handle property (defined by TrackableResource).
    # This property calls self._create_resource() the first time
    # if private self._resource_handle is not preemptively initialized.

  def _create_resource(self):
    """Create the resource tensor handle.

    `_create_resource` is an override of a method in base class
    `TrackableResource` that is required for SavedModel support. It can be
    called by the `resource_handle` property defined by `TrackableResource`.

    Returns:
      A tensor handle to the lookup table.
    """
    assert self._default_value.get_shape().ndims == 0
    table_ref = gen_simple_hash_table_op.examples_simple_hash_table_create(
        key_dtype=self._key_dtype,
        value_dtype=self._value_dtype,
        name=self._name)
    return table_ref

  def _serialize_to_tensors(self):
    """Implements checkpointing protocols for `Trackable`."""
    tensors = self.export()
    return {"table-keys": tensors[0], "table-values": tensors[1]}

  def _restore_from_tensors(self, restored_tensors):
    """Implements checkpointing protocols for `Trackable`."""
    return gen_simple_hash_table_op.examples_simple_hash_table_import(
        self.resource_handle, restored_tensors["table-keys"],
        restored_tensors["table-values"])

  @property
  def key_dtype(self):
    """The table key dtype."""
    return self._key_dtype

  @property
  def value_dtype(self):
    """The table value dtype."""
    return self._value_dtype

  def find(self, key, dynamic_default_value=None, name=None):
    """Looks up `key` in a table, outputs the corresponding value.

    The `default_value` is used if key not present in the table.

    Args:
      key: Key to look up. Must match the table's key_dtype.
      dynamic_default_value: The value to use if the key is missing in the
        table. If None (by default), the `table.default_value` will be used.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the value in the same shape as `key` using the
        table's value type.

    Raises:
      TypeError: when `key` do not match the table data types.
    """
    with tf.name_scope(name or "%s_lookup_table_find" % self._name):
      key = tf.convert_to_tensor(key, dtype=self._key_dtype, name="key")
      if dynamic_default_value is not None:
        dynamic_default_value = tf.convert_to_tensor(
            dynamic_default_value,
            dtype=self._value_dtype,
            name="default_value")
      value = gen_simple_hash_table_op.examples_simple_hash_table_find(
          self.resource_handle, key, dynamic_default_value
          if dynamic_default_value is not None else self._default_value)
    return value

  def insert(self, key, value, name=None):
    """Associates `key` with `value`.

    Args:
      key: Scalar key to insert.
      value: Scalar value to be associated with key.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `key` or `value` doesn't match the table data
        types.
    """
    with tf.name_scope(name or "%s_lookup_table_insert" % self._name):
      key = tf.convert_to_tensor(key, self._key_dtype, name="key")
      value = tf.convert_to_tensor(value, self._value_dtype, name="value")
      # pylint: disable=protected-access
      op = gen_simple_hash_table_op.examples_simple_hash_table_insert(
          self.resource_handle, key, value)
      return op

  def remove(self, key, name=None):
    """Remove `key`.

    Args:
      key: Scalar key to remove.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `key` doesn't match the table data type.
    """
    with tf.name_scope(name or "%s_lookup_table_remove" % self._name):
      key = tf.convert_to_tensor(key, self._key_dtype, name="key")

      # For remove, just the key is used by the kernel; no value is used.
      # But the kernel is specific to key_dtype and value_dtype
      # (i.e. it uses a <key_dtype, value_dtype> template).
      # So value_dtype is passed in explicitly. (While
      # key_dtype is specified implicitly by the dtype of key.)

      # pylint: disable=protected-access
      op = gen_simple_hash_table_op.examples_simple_hash_table_remove(
          self.resource_handle, key, value_dtype=self._value_dtype)
      return op

  def export(self, name=None):
    """Export all `key` and `value` pairs.

    Args:
      name: A name for the operation (optional).

    Returns:
      A tuple of two tensors, the first with the `keys` and the second with
      the `values`.
    """
    with tf.name_scope(name or "%s_lookup_table_export" % self._name):
      # pylint: disable=protected-access
      keys, values = gen_simple_hash_table_op.examples_simple_hash_table_export(
          self.resource_handle,
          key_dtype=self._key_dtype,
          value_dtype=self._value_dtype)
      return keys, values

  def do_import(self, keys, values, name=None):
    """Import all `key` and `value` pairs.

    (Note that "import" is a python reserved word, so it cannot be the name of
    a method.)

    Args:
      keys: Tensor of all keys.
      values: Tensor of all values.
      name: A name for the operation (optional).

    Returns:
      A tuple of two tensors, the first with the `keys` and the second with
      the `values`.
    """
    with tf.name_scope(name or "%s_lookup_table_import" % self._name):
      # pylint: disable=protected-access
      op = gen_simple_hash_table_op.examples_simple_hash_table_import(
          self.resource_handle, keys, values)
      return op


tf.no_gradient("Examples>SimpleHashTableCreate")
tf.no_gradient("Examples>SimpleHashTableFind")
tf.no_gradient("Examples>SimpleHashTableInsert")
tf.no_gradient("Examples>SimpleHashTableRemove")
