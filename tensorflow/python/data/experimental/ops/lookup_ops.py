# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#==============================================================================
"""Lookup operations."""

from tensorflow.python.data.experimental.ops.cardinality import assert_cardinality
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.DatasetInitializer")
class DatasetInitializer(lookup_ops.TableInitializerBase):
  """Creates a table initializer from a `tf.data.Dataset`.

  Sample usage:

  >>> keys = tf.data.Dataset.range(100)
  >>> values = tf.data.Dataset.range(100).map(
  ...     lambda x: tf.strings.as_string(x * 2))
  >>> ds = tf.data.Dataset.zip((keys, values))
  >>> init = tf.data.experimental.DatasetInitializer(ds)
  >>> table = tf.lookup.StaticHashTable(init, "")
  >>> table.lookup(tf.constant([0, 1, 2], dtype=tf.int64)).numpy()
  array([b'0', b'2', b'4'], dtype=object)

  Attributes:
    dataset: A `tf.data.Dataset` object that produces tuples of scalars. The
      first scalar is treated as a key and the second as value.
  Raises: ValueError if `dataset` doesn't conform to specifications.
  """

  def __init__(self, dataset):
    """Creates a table initializer from a `tf.data.Dataset`.

    Args:
      dataset: A `tf.data.Dataset` object that produces tuples of scalars. The
        first scalar is treated as a key and the second as value.
    Raises: ValueError if `dataset` doesn't conform to specifications.
    Returns: A `DatasetInitializer` object
    """
    # Assert that the dataset element spec is a tuple of TensorSpecs where
    # each tensor is a scalar.
    self.dataset = dataset
    elem_spec = self.dataset.element_spec
    if len(elem_spec) != 2:
      raise ValueError("element spec size should be 2")
    if not isinstance(elem_spec[0], tensor_spec.TensorSpec):
      raise ValueError("elem_spec[0] should be of type TensorSpec, got: %s" %
                       type(elem_spec[1]))
    if not isinstance(elem_spec[1], tensor_spec.TensorSpec):
      raise ValueError("elem_spec[1] should be of type TensorSpec, got: %s" %
                       type(elem_spec[1]))
    if elem_spec[0].shape.rank not in (None, 0):
      raise ValueError("key tensor should be a scalar")
    if elem_spec[1].shape.rank not in (None, 0):
      raise ValueError("value tensor should be a scalar")

    key_type = elem_spec[0].dtype
    value_type = elem_spec[1].dtype
    super(DatasetInitializer, self).__init__(key_type, value_type)

  def initialize(self, table):
    lookup_ops.check_table_dtypes(table, self._key_dtype, self._value_dtype)
    init_op = ged_ops.initialize_table_from_dataset(
        table.resource_handle, self.dataset._variant_tensor)  # pylint: disable=protected-access
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op


@tf_export("data.experimental.table_from_dataset")
def table_from_dataset(dataset=None,
                       num_oov_buckets=0,
                       vocab_size=None,
                       default_value=None,
                       hasher_spec=lookup_ops.FastHashSpec,
                       key_dtype=dtypes.string,
                       name=None):
  """Returns a lookup table based on the given dataset.

  This operation constructs a lookup table based on the given dataset of pairs
  of (key, value).

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is
  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.

  Sample Usages:

  >>> keys = tf.data.Dataset.range(100)
  >>> values = tf.data.Dataset.range(100).map(
  ...     lambda x: tf.strings.as_string(x * 2))
  >>> ds = tf.data.Dataset.zip((keys, values))
  >>> table = tf.data.experimental.table_from_dataset(
  ...                               ds, default_value='n/a', key_dtype=tf.int64)
  >>> table.lookup(tf.constant([0, 1, 2], dtype=tf.int64)).numpy()
  array([b'0', b'2', b'4'], dtype=object)

  Args:
    dataset: A dataset containing (key, value) pairs.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignation of out-of-vocabulary buckets.
    key_dtype: The `key` data type.
    name: A name for this op (optional).

  Returns:
    The lookup table based on the given dataset.

  Raises:
    ValueError: If
      * `dataset` does not contain pairs
      * The 2nd item in the `dataset` pairs has a dtype which is incompatible
        with `default_value`
      * `num_oov_buckets` is negative
      * `vocab_size` is not greater than zero
      * The `key_dtype` is not integer or string
  """
  elem_spec = dataset.element_spec
  if len(elem_spec) != 2:
    raise ValueError("The given dataset must contain pairs.")
  if default_value is None:
    default_value = -1
    if not (elem_spec[1].dtype.is_integer or elem_spec[1].dtype.is_floating):
      raise ValueError("The dtype of the values requires manually setting a "
                       "compatible default_value.")
  if num_oov_buckets < 0:
    raise ValueError(
        "num_oov_buckets must be greater or equal than 0, got %d." %
        num_oov_buckets)
  if (not isinstance(vocab_size, ops.Tensor) and vocab_size is not None and
      vocab_size < 1):
    raise ValueError("vocab_size must be greater than 0, got %d." % vocab_size)
  if (not key_dtype.is_integer) and (dtypes.string != key_dtype.base_dtype):
    raise TypeError("Only integer and string keys are supported.")
  if vocab_size is not None:
    if isinstance(vocab_size, ops.Tensor):
      vocab_size = math_ops.cast(vocab_size, dtypes.int64)
    dataset = dataset.take(vocab_size)
    dataset = dataset.apply(assert_cardinality(vocab_size))
  with ops.name_scope(name, "string_to_index"):
    initializer = DatasetInitializer(dataset)
    with ops.name_scope(None, "hash_table"):
      table = lookup_ops.StaticHashTableV1(initializer, default_value)
      if num_oov_buckets:
        table = lookup_ops.IdTableWithHashBuckets(
            table,
            num_oov_buckets=num_oov_buckets,
            hasher_spec=hasher_spec,
            key_dtype=key_dtype)
      return table


@tf_export("data.experimental.index_table_from_dataset")
def index_table_from_dataset(dataset=None,
                             num_oov_buckets=0,
                             vocab_size=None,
                             default_value=-1,
                             hasher_spec=lookup_ops.FastHashSpec,
                             key_dtype=dtypes.string,
                             name=None):
  """Returns an index lookup table based on the given dataset.

  This operation constructs a lookup table based on the given dataset of keys.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is
  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.

  Sample Usages:

  >>> ds = tf.data.Dataset.range(100).map(lambda x: tf.strings.as_string(x * 2))
  >>> table = tf.data.experimental.index_table_from_dataset(
  ...                                     ds, key_dtype=dtypes.int64)
  >>> table.lookup(tf.constant(['0', '2', '4'], dtype=tf.string)).numpy()
  array([0, 1, 2])

  Args:
    dataset: A dataset of keys.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignation of out-of-vocabulary buckets.
    key_dtype: The `key` data type.
    name: A name for this op (optional).

  Returns:
    The lookup table based on the given dataset.

  Raises:
    ValueError: If
      * `num_oov_buckets` is negative
      * `vocab_size` is not greater than zero
      * The `key_dtype` is not integer or string
  """
  return table_from_dataset(dataset.enumerate().map(lambda v, k: (k, v)),
                            num_oov_buckets, vocab_size, default_value,
                            hasher_spec, key_dtype, name)
