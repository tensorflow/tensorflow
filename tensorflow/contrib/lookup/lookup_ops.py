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

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
# pylint: disable=unused-import
from tensorflow.python.ops.lookup_ops import FastHashSpec
from tensorflow.python.ops.lookup_ops import HasherSpec
from tensorflow.python.ops.lookup_ops import IdTableWithHashBuckets
from tensorflow.python.ops.lookup_ops import index_table_from_file
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow.python.ops.lookup_ops import InitializableLookupTableBase
from tensorflow.python.ops.lookup_ops import InitializableLookupTableBaseV1
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer
from tensorflow.python.ops.lookup_ops import LookupInterface
from tensorflow.python.ops.lookup_ops import StrongHashSpec
from tensorflow.python.ops.lookup_ops import TableInitializerBase
from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer
from tensorflow.python.ops.lookup_ops import TextFileIndex
from tensorflow.python.ops.lookup_ops import TextFileInitializer
from tensorflow.python.ops.lookup_ops import TextFileStringTableInitializer
# pylint: enable=unused-import
from tensorflow.python.util.deprecation import deprecated


@deprecated("2017-04-10", "Use `index_table_from_file`.")
def string_to_index_table_from_file(vocabulary_file=None,
                                    num_oov_buckets=0,
                                    vocab_size=None,
                                    default_value=-1,
                                    hasher_spec=FastHashSpec,
                                    name=None):
  return index_table_from_file(
      vocabulary_file,
      num_oov_buckets,
      vocab_size,
      default_value,
      hasher_spec,
      key_dtype=dtypes.string,
      name=name)


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
  `session.run(tf.compat.v1.tables_initializer)` or `session.run(table.init)`
  once.

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
  tf.compat.v1.tables_initializer().run()

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


@deprecated("2017-01-07", "This op will be removed after the deprecation date. "
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
  `session.run(tf.compat.v1.tables_initializer)` once.

  For example:

  ```python
  mapping_strings = tf.constant(["emerson", "lake", "palmer"])
  feats = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = tf.contrib.lookup.string_to_index(
      feats, mapping=mapping_strings, default_value=-1)
  ...
  tf.compat.v1.tables_initializer().run()

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
  `session.run(tf.compat.v1.tables_initializer)` or `session.run(table.init)`
  once.

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
  tf.compat.v1.tables_initializer().run()

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
  `session.run(tf.compat.v1.tables_initializer)` once.

  For example:

  ```python
  mapping_string = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  values = tf.contrib.lookup.index_to_string(
      indices, mapping=mapping_string, default_value="UNKNOWN")
  ...
  tf.compat.v1.tables_initializer().run()

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


class HashTable(InitializableLookupTableBaseV1):
  """A generic hash table implementation.

  Example usage:

  ```python
  table = tf.HashTable(
      tf.KeyValueTensorInitializer(keys, values), -1)
  out = table.lookup(input_tensor)
  table.init.run()
  print(out.eval())
  ```
  """

  def __init__(self, initializer, default_value, shared_name=None, name=None):
    """Creates a non-initialized `HashTable` object.

    Creates a table, the type of its keys and values are specified by the
    initializer.
    Before using the table you will have to initialize it. After initialization
    the table will be immutable.

    Args:
      initializer: The table initializer to use. See `HashTable` kernel for
        supported key and value types.
      default_value: The value to use if a key is missing in the table.
      shared_name: If non-empty, this table will be shared under the given name
        across multiple sessions.
      name: A name for the operation (optional).

    Returns:
      A `HashTable` object.
    """
    self._initializer = initializer
    self._default_value = default_value
    self._shared_name = shared_name
    self._name = name or "hash_table"
    self._table_name = None
    super(HashTable, self).__init__(default_value, initializer)
    self._value_shape = self._default_value.get_shape()

  def _create_resource(self):
    table_ref = gen_lookup_ops.hash_table_v2(
        shared_name=self._shared_name,
        key_dtype=self._initializer.key_dtype,
        value_dtype=self._initializer.value_dtype,
        name=self._name)
    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref

  @property
  def init(self):
    return self.initializer

  @property
  def name(self):
    return self._table_name

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_Export" % self.name,
                        [self.resource_handle]) as name:
      exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
          self.resource_handle, self._key_dtype, self._value_dtype, name=name)

    exported_values.set_shape(exported_keys.get_shape().concatenate(
        self._value_shape))
    return exported_keys, exported_values


MutableHashTable = lookup_ops.MutableHashTable
MutableDenseHashTable = lookup_ops.DenseHashTable
