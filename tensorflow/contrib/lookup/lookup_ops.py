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
"""Lookup table Operations."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops


class LookupInterface(object):
  """Represent a lookup table that persists across different steps."""

  def __init__(self, key_dtype, value_dtype, name):
    """Construct a lookup table interface.

    Args:
      key_dtype: The table key type.
      value_dtype: The table value type.
      name: A name for the operation (optional).
    """
    self._key_dtype = dtypes.as_dtype(key_dtype)
    self._value_dtype = dtypes.as_dtype(value_dtype)
    self._name = name

  @property
  def key_dtype(self):
    """The table key dtype."""
    return self._key_dtype

  @property
  def value_dtype(self):
    """The table value dtype."""
    return self._value_dtype

  @property
  def name(self):
    """The name of the table."""
    return self._name

  @property
  def init(self):
    """The table initialization op."""
    raise NotImplementedError

  def size(self, name=None):
    """Compute the number of elements in this table."""
    raise NotImplementedError

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values."""
    raise NotImplementedError

  def _check_table_dtypes(self, key_dtype, value_dtype):
    """Check that the given key_dtype and value_dtype matches the table dtypes.

    Args:
      key_dtype: The key data type to check.
      value_dtype: The value data type to check.

    Raises:
      TypeError: when 'key_dtype' or 'value_dtype' doesn't match the table data
        types.
    """
    if key_dtype != self.key_dtype:
      raise TypeError("Invalid key dtype, expected %s but got %s." %
                      (self.key_dtype, key_dtype))
    if value_dtype != self.value_dtype:
      raise TypeError("Invalid value dtype, expected %s but got %s." %
                      (self.value_dtype, value_dtype))


class InitializableLookupTableBase(LookupInterface):
  """Initializable lookup table interface.

  An initializable lookup tables persist across different steps.
  """

  def __init__(self, table_ref, default_value, initializer):
    """Construct a table object from a table reference.

    If requires a table initializer object (subclass of `TableInitializerBase`).
    It provides the table key and value types, as well as the op to initialize
    the table. The caller is responsible to execute the initialization op.

    Args:
      table_ref: The table reference, i.e. the output of the lookup table ops.
      default_value: The value to use if a key is missing in the table.
      initializer: The table initializer to use.
    """
    super(InitializableLookupTableBase, self).__init__(
        initializer.key_dtype, initializer.value_dtype,
        table_ref.op.name.split("/")[-1])
    self._table_ref = table_ref
    self._default_value = ops.convert_to_tensor(default_value,
                                                dtype=self._value_dtype)
    self._default_value.get_shape().merge_with(tensor_shape.scalar())
    self._init = initializer.initialize(self)

  @property
  def table_ref(self):
    """Get the underlying table reference."""
    return self._table_ref

  @property
  def default_value(self):
    """The default value of the table."""
    return self._default_value

  @property
  def init(self):
    """The table initialization op."""
    return self._init

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    if name is None:
      name = "%s_Size" % self._name
    # pylint: disable=protected-access
    return gen_data_flow_ops._lookup_table_size(self._table_ref, name=name)
    # pylint: enable=protected-access

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is use for keys not present in the table.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: A name for the operation (optional).

    Returns:
      A `SparseTensor` if keys are sparse, otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` or `default_value` doesn't match the table data
        types.
    """
    if name is None:
      name = "%s_lookup_table_find" % self._name

    key_tensor = keys
    if isinstance(keys, ops.SparseTensor):
      key_tensor = keys.values

    if keys.dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    # pylint: disable=protected-access
    values = gen_data_flow_ops._lookup_table_find(self._table_ref,
                                                  key_tensor,
                                                  self._default_value,
                                                  name=name)
    # pylint: enable=protected-access

    if isinstance(keys, ops.SparseTensor):
      return ops.SparseTensor(keys.indices, values, keys.shape)
    else:
      return values


class HashTable(InitializableLookupTableBase):
  """A generic hash table implementation.

  Example usage:

  ```python
  table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
  out = table.lookup(input_tensor).
  table.init.run()
  print out.eval()
  ```
  """

  def __init__(self, initializer, default_value, shared_name=None, name=None):
    """Creates a non-initialized `HashTable` object.

    Creates a table, the type of its keys and values are specified by the
    initializer.
    Before using the table you will have to initialize it. After initialization
    the table will be immutable.

    Args:
      initializer: The table initializer to use.
      default_value: The value to use if a key is missing in the table.
      shared_name: If non-empty, this table will be shared under
        the given name across multiple sessions.
      name: A name for the operation (optional).

    Returns:
      A `HashTable` object.
    """
    with ops.op_scope([initializer], name, "hash_table"):
      # pylint: disable=protected-access
      table_ref = gen_data_flow_ops._hash_table(
          shared_name=shared_name,
          key_dtype=initializer.key_dtype,
          value_dtype=initializer.value_dtype,
          name=name)
      # pylint: enable=protected-access

      super(HashTable, self).__init__(table_ref, default_value, initializer)


class TableInitializerBase(object):
  """Base class for lookup table initializers."""

  def __init__(self, key_dtype, value_dtype):
    """Construct a table initializer object.

    Args:
      key_dtype: Type of the table keys.
      value_dtype: Type of the table values.
    """
    self._key_dtype = dtypes.as_dtype(key_dtype)
    self._value_dtype = dtypes.as_dtype(value_dtype)

  @property
  def key_dtype(self):
    """The expected table key dtype."""
    return self._key_dtype

  @property
  def value_dtype(self):
    """The expected table value dtype."""
    return self._value_dtype

  def initialize(self, table):
    """Returns the table initialization op."""
    raise NotImplementedError


class KeyValueTensorInitializer(TableInitializerBase):
  """Table initializers given `keys` and `values` tensors."""

  def __init__(self, keys, values, key_dtype=None, value_dtype=None, name=None):
    """Constructs a table initializer object based on keys and values tensors.

    Args:
      keys: The tensor for the keys.
      values: The tensor for the values.
      key_dtype: The `keys` data type. Used when `keys` is a python array.
      value_dtype: The `values` data type. Used when `values` is a python array.
      name: A name for the operation (optional).
    """
    with ops.op_scope([keys, values], name, "key_value_init") as scope:
      self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name="keys")
      self._values = ops.convert_to_tensor(values,
                                           dtype=value_dtype,
                                           name="values")
      self._name = scope

    super(KeyValueTensorInitializer, self).__init__(self._keys.dtype,
                                                    self._values.dtype)

  def initialize(self, table):
    """Initializes the given `table` with `keys` and `values` tensors.

    Args:
      table: The table to initialize.

    Returns:
      The operation that initializes the table.

    Raises:
      TypeError: when the keys and values data types do not match the table
      key and value data types.
    """
    # pylint: disable=protected-access
    table._check_table_dtypes(self._keys.dtype, self._values.dtype)
    with ops.op_scope([table], self._name) as scope:
      init_op = gen_data_flow_ops._initialize_table(table.table_ref,
                                                    self._keys,
                                                    self._values,
                                                    name=scope)
    # pylint: enable=protected-access
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op


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
  `tf.initialize_all_tables.run()` once.

  For example:

  ```python
  mapping_strings = t.constant(["emerson", "lake", "palmer")
  feats = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = tf.contrib.lookup.string_to_index(
      feats, mapping=mapping_strings, default_value=-1)
  ...
  tf.initialize_all_tables().run()

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
  with ops.op_scope([tensor], name, "string_to_index") as scope:
    shared_name = ""
    keys = ops.convert_to_tensor(mapping, dtypes.string)
    vocab_size = array_ops.size(keys)
    values = math_ops.cast(math_ops.range(vocab_size), dtypes.int64)
    init = KeyValueTensorInitializer(keys,
                                     values,
                                     dtypes.string,
                                     dtypes.int64,
                                     name="table_init")

    t = HashTable(init,
                  default_value,
                  shared_name=shared_name,
                  name="hash_table")
    return t.lookup(tensor, name=scope)


def index_to_string(tensor, mapping, default_value="UNK", name=None):
  """Maps `tensor` of indices into string values based on `mapping`.

  This operation converts `int64` indices into string values. The mapping is
  initialized from a string `mapping` tensor where each element is a value and
  the corresponding index within the tensor is the key.

  Any input which does not have a corresponding index in 'mapping'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `tf.initialize_all_tables.run()` once.

  For example:

  ```python
  mapping_string = t.constant(["emerson", "lake", "palmer")
  indices = tf.constant([1, 5], tf.int64)
  values = tf.contrib.lookup.index_to_string(
      indices, mapping=mapping_string, default_value="UNKNOWN")
  ...
  tf.initialize_all_tables().run()

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
  with ops.op_scope([tensor], name, "index_to_string") as scope:
    shared_name = ""
    values = ops.convert_to_tensor(mapping, dtypes.string)
    vocab_size = array_ops.size(values)
    keys = math_ops.cast(math_ops.range(vocab_size), dtypes.int64)
    init = KeyValueTensorInitializer(keys,
                                     values,
                                     dtypes.int64,
                                     dtypes.string,
                                     name="table_init")
    t = HashTable(init,
                  default_value,
                  shared_name=shared_name,
                  name="hash_table")
    return t.lookup(tensor, name=scope)
