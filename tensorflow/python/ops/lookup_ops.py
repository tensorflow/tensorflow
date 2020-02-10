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
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import uuid

import six

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_lookup_ops import *
from tensorflow.python.training.saver import BaseSaverBuilder
# pylint: enable=wildcard-import
from tensorflow.python.training.tracking import base as trackable_base
from tensorflow.python.training.tracking import tracking as trackable
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["initialize_all_tables"])
@deprecated(None, "Use `tf.tables_initializer` instead.")
def initialize_all_tables(name="init_all_tables"):
  """Returns an Op that initializes all tables of the default graph.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.
  """
  return tables_initializer(name)


@tf_export(v1=["initializers.tables_initializer", "tables_initializer"])
def tables_initializer(name="init_all_tables"):
  """Returns an Op that initializes all tables of the default graph.

  See the [Low Level
  Intro](https://www.tensorflow.org/guide/low_level_intro#feature_columns)
  guide, for an example of usage.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.
  """
  initializers = ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)
  if initializers:
    return control_flow_ops.group(*initializers, name=name)
  return control_flow_ops.no_op(name=name)


def _check_table_dtypes(table, key_dtype, value_dtype):
  """Check that the given key_dtype and value_dtype matches the table dtypes.

  Args:
    table: The table to check types against to.
    key_dtype: The key data type to check.
    value_dtype: The value data type to check.

  Raises:
    TypeError: when 'key_dtype' or 'value_dtype' doesn't match the table data
      types.
  """
  if key_dtype.base_dtype != table.key_dtype:
    raise TypeError("Invalid key dtype, expected %s but got %s." %
                    (table.key_dtype, key_dtype))
  if value_dtype.base_dtype != table.value_dtype:
    raise TypeError("Invalid value dtype, expected %s but got %s." %
                    (table.value_dtype, value_dtype))


class LookupInterface(trackable.TrackableResource):
  """Represent a lookup table that persists across different steps."""

  def __init__(self, key_dtype, value_dtype):
    """Construct a lookup table interface.

    Args:
      key_dtype: The table key type.
      value_dtype: The table value type.
    """
    self._key_dtype = dtypes.as_dtype(key_dtype)
    self._value_dtype = dtypes.as_dtype(value_dtype)
    super(LookupInterface, self).__init__()

  def _create_resource(self):
    raise NotImplementedError

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
    return NotImplementedError

  def size(self, name=None):
    """Compute the number of elements in this table."""
    raise NotImplementedError

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values."""
    raise NotImplementedError


class InitializableLookupTableBase(LookupInterface):
  """Initializable lookup table interface.

  An initializable lookup tables persist across different steps.
  """

  def __init__(self, default_value, initializer):
    """Construct a table object from a table reference.

    If requires a table initializer object (subclass of `TableInitializerBase`).
    It provides the table key and value types, as well as the op to initialize
    the table. The caller is responsible to execute the initialization op.

    Args:
      default_value: The value to use if a key is missing in the table.
      initializer: The table initializer to use.
    """
    super(InitializableLookupTableBase, self).__init__(initializer.key_dtype,
                                                       initializer.value_dtype)
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=self._value_dtype)
    self._default_value.get_shape().merge_with(tensor_shape.TensorShape([]))
    if isinstance(initializer, trackable_base.Trackable):
      self._initializer = self._track_trackable(initializer, "_initializer")
    with ops.init_scope():
      self._resource_handle = self._create_resource()
    if (not context.executing_eagerly() and
        ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access
      with ops.init_scope():
        self._init_op = self._initialize()
    else:
      self._init_op = self._initialize()

  def _initialize(self):
    return self._initializer.initialize(self)

  @property
  def default_value(self):
    """The default value of the table."""
    return self._default_value

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: A name for the operation (optional).

    Returns:
      A `SparseTensor` if keys are sparse, otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` or `default_value` doesn't match the table data
        types.
    """
    key_tensor = keys
    if isinstance(keys, sparse_tensor.SparseTensor):
      key_tensor = keys.values

    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    with ops.name_scope(
        name, "%s_Lookup" % self.name,
        (self.resource_handle, key_tensor, self._default_value)):
      values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle,
                                                   key_tensor,
                                                   self._default_value)

    values.set_shape(key_tensor.get_shape())
    if isinstance(keys, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(keys.indices, values, keys.dense_shape)
    else:
      return values


class InitializableLookupTableBaseV1(InitializableLookupTableBase):

  @property
  def initializer(self):
    return self._init_op


@tf_export("lookup.StaticHashTable", v1=[])
class StaticHashTable(InitializableLookupTableBase):
  """A generic hash table that is immutable once initialized.

  Example usage:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  print(table.lookup(input_tensor))
  ```
  """

  def __init__(self, initializer, default_value, name=None):
    """Creates a non-initialized `HashTable` object.

    Creates a table, the type of its keys and values are specified by the
    initializer.
    Before using the table you will have to initialize it. After initialization
    the table will be immutable.

    Args:
      initializer: The table initializer to use. See `HashTable` kernel for
        supported key and value types.
      default_value: The value to use if a key is missing in the table.
      name: A name for the operation (optional).

    Returns:
      A `HashTable` object.
    """
    self._initializer = initializer
    self._default_value = default_value
    self._shared_name = self._initializer._shared_name  # pylint: disable=protected-access
    if not self._shared_name:
      # Force using a shared name so that StaticHashTable resources can be
      # shared across different kernels. If no "shared_name" is set and
      # "use_node_name_sharing" is False, then each kernel gets its own local
      # resource.
      self._shared_name = "hash_table_%s" % (str(uuid.uuid4()),)
    self._name = name or "hash_table"
    self._table_name = None
    super(StaticHashTable, self).__init__(default_value, initializer)
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
    with ops.name_scope(name, "%s_Export" % self.name, [self.resource_handle]):
      exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
          self.resource_handle, self._key_dtype, self._value_dtype)

    exported_values.set_shape(exported_keys.get_shape().concatenate(
        self._value_shape))
    return exported_keys, exported_values


@tf_export(v1=["lookup.StaticHashTable"])
class StaticHashTableV1(StaticHashTable):
  """A generic hash table that is immutable once initialized.

  When running in graph mode, you must evaluate the tensor returned by
  `tf.tables_initializer()` before evaluating the tensor returned by
  this class's `lookup()` method. Example usage in graph mode:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  out = table.lookup(input_tensor)
  with tf.Session() as sess:
      sess.run(tf.tables_initializer())
      print(sess.run(out))
  ```

  In eager mode, no special code is needed to initialize the table.
  Example usage in eager mode:

  ```python
  tf.enable_eager_execution()
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  print(table.lookup(input_tensor))
  ```
  """

  @property
  def initializer(self):
    return self._init_op


# For backwards compatibility. This will be removed in TF 2.0.
class HashTable(StaticHashTableV1):

  @property
  def init(self):
    return self.initializer


class TableInitializerBase(trackable_base.Trackable):
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

  @property
  def _shared_name(self):
    """Returns a shared name to be used by the table."""
    shared_name = ""
    if context.executing_eagerly():
      # Ensure a unique name when eager execution is enabled to avoid spurious
      # sharing issues.
      # TODO(rohanj): Use context.shared_name() instead.
      shared_name += str(ops.uid())
    return shared_name


@tf_export("lookup.KeyValueTensorInitializer")
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
    if (not context.executing_eagerly() and
        ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access
      with ops.init_scope():
        self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name="keys")
        self._values = ops.convert_to_tensor(
            values, dtype=value_dtype, name="values")
    else:
      self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name="keys")
      self._values = ops.convert_to_tensor(
          values, dtype=value_dtype, name="values")
    self._name = name if name is not None else "key_value_init"
    if context.executing_eagerly():
      # Ensure a unique name when eager execution is enabled to avoid spurious
      # sharing issues.
      # TODO(rohanj): Use context.shared_name() instead.
      self._name += str(ops.uid())

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
    _check_table_dtypes(table, self._keys.dtype, self._values.dtype)
    with ops.name_scope(
        self._name, values=(table.resource_handle, self._keys, self._values)):
      init_op = gen_lookup_ops.lookup_table_import_v2(table.resource_handle,
                                                      self._keys, self._values)
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op


@tf_export("lookup.TextFileIndex")
class TextFileIndex(object):
  """The key and value content to get from each line.

  This class defines the key and value used for tf.lookup.TextFileInitializer.

  The key and value content to get from each line is specified either
  by the following, or a value `>=0`.
  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.

  A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.
  """
  WHOLE_LINE = -2
  LINE_NUMBER = -1


@tf_export("lookup.TextFileInitializer")
class TextFileInitializer(TableInitializerBase):
  """Table initializers from a text file.

  This initializer assigns one entry in the table for each line in the file.

  The key and value type of the table to initialize is given by `key_dtype` and
  `value_dtype`.

  The key and value content to get from each line is specified by
  the `key_index` and `value_index`.

  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.
  * A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.

  For example if we have a file with the following content:

  ```
  emerson 10
  lake 20
  palmer 30
  ```

  The following snippet initializes a table with the first column as keys and
  second column as values:

  * `emerson -> 10`
  * `lake -> 20`
  * `palmer -> 30`

  ```python
  table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
      "test.txt", tf.string, 0, tf.int64, 1, delimiter=" "), -1)
  ...
  table.init.run()
  ```

  Similarly to initialize the whole line as keys and the line number as values.

  * `emerson 10 -> 0`
  * `lake 20 -> 1`
  * `palmer 30 -> 2`

  ```python
  table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
      "test.txt", tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
      tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" "), -1)
  ...
  table.init.run()
  ```
  """

  def __init__(self,
               filename,
               key_dtype,
               key_index,
               value_dtype,
               value_index,
               vocab_size=None,
               delimiter="\t",
               name=None):
    """Constructs a table initializer object to populate from a text file.

    It generates one key-value pair per line. The type of table key and
    value are specified by `key_dtype` and `value_dtype`, respectively.
    Similarly the content of the key and value are specified by the key_index
    and value_index.

    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
      expects data type int64.
    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
      type string.
    - A value >=0 means use the index (starting at zero) of the split line based
      on `delimiter`.

    Args:
      filename: The filename of the text file to be used for initialization. The
        path must be accessible from wherever the graph is initialized (eg.
        trainer or eval workers). The filename may be a scalar `Tensor`.
      key_dtype: The `key` data type.
      key_index: the index that represents information of a line to get the
        table 'key' values from.
      value_dtype: The `value` data type.
      value_index: the index that represents information of a line to get the
        table 'value' values from.'
      vocab_size: The number of elements in the file, if known.
      delimiter: The delimiter to separate fields in a line.
      name: A name for the operation (optional).

    Raises:
      ValueError: when the filename is empty, or when the table key and value
      data types do not match the expected data types.
    """
    if not isinstance(filename, ops.Tensor) and not filename:
      raise ValueError("Filename required for %s." % name)

    self._filename_arg = filename
    key_dtype = dtypes.as_dtype(key_dtype)
    value_dtype = dtypes.as_dtype(value_dtype)

    if key_index < -2:
      raise ValueError("Invalid key index %s." % (key_index))

    if key_index == TextFileIndex.LINE_NUMBER and key_dtype != dtypes.int64:
      raise ValueError("Signature mismatch. Keys must be dtype %s, got %s." %
                       (dtypes.int64, key_dtype))
    if ((key_index == TextFileIndex.WHOLE_LINE) and
        (not key_dtype.is_integer) and (key_dtype != dtypes.string)):
      raise ValueError(
          "Signature mismatch. Keys must be integer or string, got %s." %
          key_dtype)
    if value_index < -2:
      raise ValueError("Invalid value index %s." % (value_index))

    if value_index == TextFileIndex.LINE_NUMBER and value_dtype != dtypes.int64:
      raise ValueError("Signature mismatch. Values must be dtype %s, got %s." %
                       (dtypes.int64, value_dtype))
    if value_index == TextFileIndex.WHOLE_LINE and value_dtype != dtypes.string:
      raise ValueError("Signature mismatch. Values must be dtype %s, got %s." %
                       (dtypes.string, value_dtype))

    if (vocab_size is not None) and (vocab_size <= 0):
      raise ValueError("Invalid vocab_size %s." % vocab_size)

    self._key_index = key_index
    self._value_index = value_index
    self._vocab_size = vocab_size
    self._delimiter = delimiter
    self._name = name
    self._filename = self._track_trackable(
        trackable.Asset(filename), "_filename")

    super(TextFileInitializer, self).__init__(key_dtype, value_dtype)

  def initialize(self, table):
    """Initializes the table from a text file.

    Args:
      table: The table to be initialized.

    Returns:
      The operation that initializes the table.

    Raises:
      TypeError: when the keys and values data types do not match the table
      key and value data types.
    """
    _check_table_dtypes(table, self.key_dtype, self.value_dtype)
    with ops.name_scope(self._name, "text_file_init", (table.resource_handle,)):
      filename = ops.convert_to_tensor(
          self._filename, dtypes.string, name="asset_filepath")
      init_op = gen_lookup_ops.initialize_table_from_text_file_v2(
          table.resource_handle, filename, self._key_index, self._value_index,
          -1 if self._vocab_size is None else self._vocab_size, self._delimiter)
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    # If the filename tensor is anything other than a string constant (e.g.,
    # if it is a placeholder) then it does not make sense to track it as an
    # asset.
    if not context.executing_eagerly() and constant_op.is_constant(filename):
      ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)
    return init_op

  @property
  def _shared_name(self):
    if self._vocab_size:
      # Keep the shared_name:
      # <table_type>_<filename>_<vocab_size>_<key_index>_<value_index>
      shared_name = "hash_table_%s_%d_%s_%s" % (
          self._filename_arg, self._vocab_size, self._key_index,
          self._value_index)
    else:
      # Keep the shared_name
      # <table_type>_<filename>_<key_index>_<value_index>
      shared_name = "hash_table_%s_%s_%s" % (self._filename_arg,
                                             self._key_index, self._value_index)
    return shared_name


class TextFileStringTableInitializer(TextFileInitializer):
  """Table initializer for `int64` IDs to string tables from a text file."""

  def __init__(self,
               filename,
               key_column_index=TextFileIndex.LINE_NUMBER,
               value_column_index=TextFileIndex.WHOLE_LINE,
               vocab_size=None,
               delimiter="\t",
               name="text_file_string_table_init"):
    """Constructs an initializer for an id-to-string table from a text file.

    It populates a table that its key and value types are int64 and string,
    respectively. It generates one key-value pair per line.
    The content of the key and value are specified by `key_column_index`
    and `value_column_index`.

    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
      expects data type int64.
    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
      type string.
    - A value >=0 means use the index (starting at zero) of the split line based
      on `delimiter`.

    Args:
      filename: The filename of the text file to be used for initialization. The
        path must be accessible from wherever the graph is initialized (eg.
        trainer or eval workers). The filename may be a scalar `Tensor`.
      key_column_index: The column index from the text file to get the keys
        from. The default is to use the line number, starting from zero.
      value_column_index: The column index from the text file to get the values
        from. The default is to use the whole line content.
      vocab_size: The number of elements in the file, if known.
      delimiter: The delimiter to separate fields in a line.
      name: Optional name for the op.

    Raises:
      TypeError: when the filename is empty, or when the table key and value
      data types do not match the expected data types.
    """
    super(TextFileStringTableInitializer, self).__init__(
        filename,
        dtypes.int64,
        key_column_index,
        dtypes.string,
        value_column_index,
        vocab_size=vocab_size,
        delimiter=delimiter,
        name=name)


class TextFileIdTableInitializer(TextFileInitializer):
  """Table initializer for string to `int64` IDs tables from a text file."""

  def __init__(self,
               filename,
               key_column_index=TextFileIndex.WHOLE_LINE,
               value_column_index=TextFileIndex.LINE_NUMBER,
               vocab_size=None,
               delimiter="\t",
               name="text_file_id_table_init",
               key_dtype=dtypes.string):
    """Constructs an initializer for an string-to-id table from a text file.

    It populates a table that its key and value types are string and int64,
    respectively. It generates one key-value pair per line.
    The content of the key and value are specified by the key_index
    and value_index.

    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
      expects data type int64.
    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
      type string.
    - A value >=0 means use the index (starting at zero) of the split line based
      on `delimiter`.

    Args:
      filename: The filename of the text file to be used for initialization. The
        path must be accessible from wherever the graph is initialized (eg.
        trainer or eval workers). The filename may be a scalar `Tensor`.
      key_column_index: The column index from the text file to get the `key`
        values from. The default is to use the whole line content.
      value_column_index: The column index from the text file to get the `value`
        values from. The default is to use the line number, starting from zero.
      vocab_size: The number of elements in the file, if known.
      delimiter: The delimiter to separate fields in a line.
      name: Optional name for the op.
      key_dtype: The `key` data type.

    Raises:
      TypeError: when the filename is empty, or when the table key and value
      data types do not match the expected data types.
    """
    super(TextFileIdTableInitializer, self).__init__(
        filename,
        key_dtype,
        key_column_index,
        dtypes.int64,
        value_column_index,
        vocab_size=vocab_size,
        delimiter=delimiter,
        name=name)


class HasherSpec(collections.namedtuple("HasherSpec", ["hasher", "key"])):
  """A structure for the spec of the hashing function to use for hash buckets.

  `hasher` is the name of the hashing function to use (eg. "fasthash",
  "stronghash").
  `key` is optional and specify the key to use for the hash function if
  supported, currently only used by a strong hash.

  Fields:
    hasher: The hasher name to use.
    key: The key to be used by the hashing function, if required.
  """
  __slots__ = ()


FastHashSpec = HasherSpec("fasthash", None)  # pylint: disable=invalid-name


class StrongHashSpec(HasherSpec):
  """A structure to specify a key of the strong keyed hash spec.

  The strong hash requires a `key`, which is a list of 2 unsigned integer
  numbers. These should be non-zero; random numbers generated from random.org
  would be a fine choice.

  Fields:
    key: The key to be used by the keyed hashing function.
  """
  __slots__ = ()

  def __new__(cls, key):
    if len(key) != 2:
      raise ValueError("key must have size 2, got %s." % len(key))

    if not isinstance(key[0], compat.integral_types) or not isinstance(
        key[1], compat.integral_types):
      raise TypeError("Invalid key %s. Must be unsigned integer values." % key)

    return super(cls, StrongHashSpec).__new__(cls, "stronghash", key)


def _as_string(tensor):
  if dtypes.string == tensor.dtype.base_dtype:
    return tensor
  return string_ops.as_string(tensor)


class IdTableWithHashBuckets(LookupInterface):
  r"""String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

  For example, if an instance of `IdTableWithHashBuckets` is initialized with a
  string-to-id table that maps:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`

  The `IdTableWithHashBuckets` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where bucket_id will be between `3` and
  `3 + num_oov_buckets - 1`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is `["emerson", "lake", "palmer", "king", "crimson"]`,
  the lookup result is `[0, 1, 2, 4, 7]`.

  If `table` is None, only out-of-vocabulary buckets are used.

  Example usage:

  ```python
  num_oov_buckets = 3
  input_tensor = tf.constant(["emerson", "lake", "palmer", "king", "crimnson"])
  table = tf.IdTableWithHashBuckets(
      tf.StaticHashTable(
          tf.lookup.TextFileInitializer(
              filename,
              key_dtype=tf.string,
              key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
              value_dtype=tf.int64,
              value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
              delimiter="\t"),
          default_value),
      num_oov_buckets)
  out = table.lookup(input_tensor).
  table.init.run()
  print(out.eval())
  ```

  The hash function used for generating out-of-vocabulary buckets ID is handled
  by `hasher_spec`.
  """

  def __init__(self,
               table,
               num_oov_buckets,
               hasher_spec=FastHashSpec,
               name=None,
               key_dtype=None):
    """Construct a `IdTableWithHashBuckets` object.

    Args:
      table: Table that maps `tf.string` or `tf.int64` keys to `tf.int64` ids.
      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys.
      hasher_spec: A `HasherSpec` to specify the hash function to use for
        assignation of out-of-vocabulary buckets  (optional).
      name: A name for the operation (optional).
      key_dtype: Data type of keys passed to `lookup`. Defaults to
        `table.key_dtype` if `table` is specified, otherwise `tf.string`. Must
        be string or integer, and must be castable to `table.key_dtype`.

    Raises:
      ValueError: when `table` in None and `num_oov_buckets` is not positive.
      TypeError: when `hasher_spec` is invalid.
    """
    # If a name ends with a '/' it is a "name scope", remove all trailing '/'
    # characters to use as table name.
    if name:
      name = name.rstrip("/")
    if table:
      if key_dtype is None:
        key_dtype = table.key_dtype
      supported_table_key_dtypes = (dtypes.int64, dtypes.string)
      if table.key_dtype not in supported_table_key_dtypes:
        raise TypeError("Invalid key dtype, expected one of %s, but got %s." %
                        (supported_table_key_dtypes, key_dtype))
      if table.key_dtype.is_integer != key_dtype.is_integer:
        raise TypeError("Invalid key dtype, expected %s but got %s." %
                        ("integer" if key_dtype.is_integer else "non-integer",
                         table.key_dtype))
      if table.value_dtype != dtypes.int64:
        raise TypeError("Invalid value dtype, expected %s but got %s." %
                        (dtypes.int64, table.value_dtype))
      self._table = table
      name = name or self._table.name
    else:
      if num_oov_buckets <= 0:
        raise ValueError("oov_buckets must be > 0 if no table is supplied.")
      key_dtype = dtypes.string if key_dtype is None else key_dtype
      self._table = None
      name = name or "hash_bucket"
    if (not key_dtype.is_integer) and (dtypes.string != key_dtype):
      raise TypeError("Invalid key_dtype, expected integer or string, got %s." %
                      key_dtype)
    self._num_oov_buckets = num_oov_buckets

    if not isinstance(hasher_spec, HasherSpec):
      raise TypeError("hasher_spec must be of type HasherSpec, got %s" %
                      hasher_spec)
    self._hasher_spec = hasher_spec
    if name:
      self._table_name = name.split("/")[-1]
    else:
      self._table_name = None
    super(IdTableWithHashBuckets, self).__init__(key_dtype, dtypes.int64)

  def _create_resource(self):
    if self._table is not None:
      return self._table._create_resource()  # pylint: disable=protected-access
    return None

  def _initialize(self):
    if self._table is not None:
      return self._table._initialize()  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()

  @property
  def initializer(self):
    if self._table is not None:
      return self._table._init_op  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()

  @property
  @deprecated("2018-12-15", "Use `initializer` instead.")
  def init(self):
    return self.initializer

  @property
  def resource_handle(self):
    if self._table is not None:
      return self._table.resource_handle
    return None

  @property
  def name(self):
    return self._table_name

  def size(self, name=None):
    """Compute the number of elements in this table."""
    with ops.name_scope(name, "%s_Size" % self.name):
      if self._table:
        tsize = self._table.size()
      else:
        tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
      return tsize + self._num_oov_buckets

  def _get_string_to_hash_bucket_fn(self, hasher_spec):
    """Returns the string_to_hash_bucket op to use based on `hasher_spec`."""
    if not isinstance(hasher_spec, HasherSpec):
      raise TypeError("hasher_spec must be of type HasherSpec %s" % hasher_spec)
    if hasher_spec.hasher == "fasthash":
      return string_ops.string_to_hash_bucket_fast
    if hasher_spec.hasher == "legacy":
      return string_ops.string_to_hash_bucket
    if hasher_spec.hasher == "stronghash":
      return functools.partial(
          string_ops.string_to_hash_bucket_strong, key=hasher_spec.key)
    raise ValueError("Unknown hasher %s" % hasher_spec.hasher)

  def lookup(self, keys, name=None):
    """Looks up `keys` in the table, outputs the corresponding values.

    It assigns out-of-vocabulary keys to buckets based in their hashes.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.

    Returns:
      A `SparseTensor` if keys are sparse, otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    """
    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))
    values = keys
    if isinstance(keys, sparse_tensor.SparseTensor):
      values = keys.values
    if self._table and (self._table.key_dtype.base_dtype == dtypes.int64):
      values = math_ops.cast(values, dtypes.int64)

    if self._num_oov_buckets == 0:
      ids = self._table.lookup(values, name=name)
    else:
      # TODO(yleon): Consider moving this functionality to its own kernel.
      with ops.name_scope(name, "%s_Lookup" % self.name):
        str_to_hash_bucket = self._get_string_to_hash_bucket_fn(
            self._hasher_spec)
        buckets = str_to_hash_bucket(
            _as_string(values),
            num_buckets=self._num_oov_buckets,
            name="hash_bucket")
        if self._table:
          ids = self._table.lookup(values)
          buckets = math_ops.add(buckets, self._table.size())
          is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
          ids = array_ops.where_v2(is_id_non_default, ids, buckets)
        else:
          ids = buckets
    if isinstance(keys, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
    return ids


@tf_export("lookup.StaticVocabularyTable", v1=[])
class StaticVocabularyTable(LookupInterface):
  r"""String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

  For example, if an instance of `StaticVocabularyTable` is initialized with a
  string-to-id initializer that maps:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`

  The `Vocabulary` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where bucket_id will be between `3` and
  `3 + num_oov_buckets - 1`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is `["emerson", "lake", "palmer", "king", "crimson"]`,
  the lookup result is `[0, 1, 2, 4, 7]`.

  If `initializer` is None, only out-of-vocabulary buckets are used.

  Example usage:

  ```python
  num_oov_buckets = 3
  input_tensor = tf.constant(["emerson", "lake", "palmer", "king", "crimnson"])
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.TextFileInitializer(
          filename,
          key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
          value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
          delimiter="\t"),
      num_oov_buckets)
  out = table.lookup(input_tensor).
  table.init.run()
  print(out.eval())
  ```

  The hash function used for generating out-of-vocabulary buckets ID is
  Fingerprint64.
  """

  def __init__(self,
               initializer,
               num_oov_buckets,
               lookup_key_dtype=None,
               name=None):
    """Construct a `StaticVocabularyTable` object.

    Args:
      initializer: A TableInitializerBase object that contains the data used to
        initialize the table. If None, then we only use out-of-vocab buckets.
      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys. Must
        be greater than zero.
      lookup_key_dtype: Data type of keys passed to `lookup`. Defaults to
        `initializer.key_dtype` if `initializer` is specified, otherwise
        `tf.string`. Must be string or integer, and must be castable to
        `initializer.key_dtype`.
      name: A name for the operation (optional).

    Raises:
      ValueError: when `num_oov_buckets` is not positive.
      TypeError: when lookup_key_dtype or initializer.key_dtype are not
        integer or string. Also when initializer.value_dtype != int64.
    """
    if num_oov_buckets <= 0:
      raise ValueError("oov_buckets must be > 0.")
    # If a name ends with a '/' it is a "name scope", remove all trailing '/'
    # characters to use as table name.
    if name:
      name = name.rstrip("/")
    if initializer:
      if lookup_key_dtype is None:
        lookup_key_dtype = initializer.key_dtype
      supported_table_key_dtypes = (dtypes.int64, dtypes.string)
      if initializer.key_dtype not in supported_table_key_dtypes:
        raise TypeError("Invalid key dtype, expected one of %s, but got %s." %
                        (supported_table_key_dtypes, initializer.key_dtype))
      if initializer.key_dtype.is_integer != lookup_key_dtype.is_integer:
        raise TypeError(
            "Invalid key dtype, expected %s but got %s." %
            ("integer" if lookup_key_dtype.is_integer else "non-integer",
             initializer.key_dtype))
      if initializer.value_dtype != dtypes.int64:
        raise TypeError("Invalid value dtype, expected %s but got %s." %
                        (dtypes.int64, initializer.value_dtype))
      if isinstance(initializer, trackable_base.Trackable):
        self._initializer = self._track_trackable(initializer, "_initializer")
      self._table = HashTable(initializer, default_value=-1)
      name = name or self._table.name
    else:
      lookup_key_dtype = dtypes.string
      self._table = None
      name = name or "hash_bucket"
    if (not lookup_key_dtype.is_integer) and (dtypes.string !=
                                              lookup_key_dtype):
      raise TypeError("Invalid key_dtype, expected integer or string, got %s." %
                      lookup_key_dtype)
    self._num_oov_buckets = num_oov_buckets

    self._table_name = None
    if name is not None:
      self._table_name = name.split("/")[-1]
    super(StaticVocabularyTable, self).__init__(lookup_key_dtype, dtypes.int64)

  def _create_resource(self):
    if self._table is not None:
      return self._table._create_resource()  # pylint: disable=protected-access
    return None

  def _initialize(self):
    if self._table is not None:
      return self._table._initialize()  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()

  @property
  def resource_handle(self):
    if self._table is not None:
      return self._table.resource_handle
    return None

  @property
  def name(self):
    return self._table_name

  def size(self, name=None):
    """Compute the number of elements in this table."""
    with ops.name_scope(name, "%s_Size" % self.name):
      if self._table:
        tsize = self._table.size()
      else:
        tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
      return tsize + self._num_oov_buckets

  def lookup(self, keys, name=None):
    """Looks up `keys` in the table, outputs the corresponding values.

    It assigns out-of-vocabulary keys to buckets based in their hashes.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.

    Returns:
      A `SparseTensor` if keys are sparse, otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    """
    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))
    values = keys
    if isinstance(keys, sparse_tensor.SparseTensor):
      values = keys.values
    if self._table and (self._table.key_dtype.base_dtype == dtypes.int64):
      values = math_ops.cast(values, dtypes.int64)

    # TODO(yleon): Consider moving this functionality to its own kernel.
    with ops.name_scope(name, "%s_Lookup" % self.name):
      buckets = string_ops.string_to_hash_bucket_fast(
          _as_string(values),
          num_buckets=self._num_oov_buckets,
          name="hash_bucket")
      if self._table:
        ids = self._table.lookup(values)
        buckets = math_ops.add(buckets, self._table.size())
        is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
        ids = array_ops.where_v2(is_id_non_default, ids, buckets)
      else:
        ids = buckets
    if isinstance(keys, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
    return ids


@tf_export(v1=["lookup.StaticVocabularyTable"])
class StaticVocabularyTableV1(StaticVocabularyTable):

  @property
  def initializer(self):
    if self._table is not None:
      return self._table._init_op  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()


def index_table_from_file(vocabulary_file=None,
                          num_oov_buckets=0,
                          vocab_size=None,
                          default_value=-1,
                          hasher_spec=FastHashSpec,
                          key_dtype=dtypes.string,
                          name=None,
                          key_column_index=TextFileIndex.WHOLE_LINE,
                          value_column_index=TextFileIndex.LINE_NUMBER,
                          delimiter="\t"):
  """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a vocabulary file specified in
  `vocabulary_file`, where the whole line is the key and the zero-based line
  number is the ID.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is
  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  To specify multi-column vocabulary files, use key_column_index and
  value_column_index and delimiter.

  - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
    expects data type int64.
  - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
    type string.
  - A value >=0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Sample Usages:

  If we have a vocabulary file "test.txt" with the following content:

  ```
  emerson
  lake
  palmer
  ```

  ```python
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  table = tf.lookup.index_table_from_file(
      vocabulary_file="test.txt", num_oov_buckets=1)
  ids = table.lookup(features)
  ...
  tf.compat.v1.tables_initializer().run()

  ids.eval()  ==> [0, 1, 3, 2]  # where 3 is the out-of-vocabulary bucket
  ```

  Args:
    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignation of out-of-vocabulary buckets.
    key_dtype: The `key` data type.
    name: A name for this op (optional).
    key_column_index: The column index from the text file to get the `key`
      values from. The default is to use the whole line content.
    value_column_index: The column index from the text file to get the `value`
      values from. The default is to use the line number, starting from zero.
    delimiter: The delimiter to separate fields in a line.

  Returns:
    The lookup table to map a `key_dtype` `Tensor` to index `int64` `Tensor`.

  Raises:
    ValueError: If `vocabulary_file` is not set.
    ValueError: If `num_oov_buckets` is negative or `vocab_size` is not greater
      than zero.
  """
  if vocabulary_file is None or (isinstance(vocabulary_file, six.string_types)
                                 and not vocabulary_file):
    raise ValueError("vocabulary_file must be specified and must not be empty.")
  if num_oov_buckets < 0:
    raise ValueError(
        "num_oov_buckets must be greater or equal than 0, got %d." %
        num_oov_buckets)
  if vocab_size is not None and vocab_size < 1:
    vocab_file_value = vocabulary_file
    if isinstance(vocabulary_file, ops.Tensor):
      vocab_file_value = tensor_util.constant_value(vocabulary_file) or "?"
    raise ValueError("vocab_size must be greater than 0, got %d. "
                     "vocabulary_file: %s" % (vocab_size, vocab_file_value))
  if (not key_dtype.is_integer) and (dtypes.string != key_dtype.base_dtype):
    raise TypeError("Only integer and string keys are supported.")

  with ops.name_scope(name, "string_to_index"):
    table = None
    with ops.name_scope(None, "hash_table"):
      init = TextFileIdTableInitializer(
          vocabulary_file,
          vocab_size=vocab_size,
          key_dtype=dtypes.int64 if key_dtype.is_integer else key_dtype,
          name="table_init",
          key_column_index=key_column_index,
          value_column_index=value_column_index,
          delimiter=delimiter)

      table = StaticHashTableV1(init, default_value)
    if num_oov_buckets:
      table = IdTableWithHashBuckets(
          table,
          num_oov_buckets=num_oov_buckets,
          hasher_spec=hasher_spec,
          key_dtype=key_dtype)

    return table


def index_table_from_tensor(vocabulary_list,
                            num_oov_buckets=0,
                            default_value=-1,
                            hasher_spec=FastHashSpec,
                            dtype=dtypes.string,
                            name=None):
  """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a string `vocabulary_list` 1-D
  tensor where each element is a key and corresponding index within the tensor
  is the value.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`. The bucket ID range is
  `[vocabulary list size, vocabulary list size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing
  the table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])
  table = tf.lookup.index_table_from_tensor(
      vocabulary_list=vocabulary_list, num_oov_buckets=1, default_value=-1)
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = table.lookup(features)
  ...
  tf.compat.v1.tables_initializer().run()

  ids.eval()  ==> [0, 1, 4, 2]
  ```

  Args:
    vocabulary_list: A 1-D `Tensor` that specifies the mapping of keys to
      indices. The type of this object must be castable to `dtype`.
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
    ValueError: If `vocabulary_list` is invalid.
    ValueError: If `num_oov_buckets` is negative.
  """
  if vocabulary_list is None:
    raise ValueError("vocabulary_list must be specified.")

  if num_oov_buckets < 0:
    raise ValueError(
        "num_oov_buckets must be greater or equal than 0, got %d." %
        num_oov_buckets)

  if (not dtype.is_integer) and (dtypes.string != dtype.base_dtype):
    raise TypeError("Only integer and string keys are supported.")

  with ops.name_scope(name, "string_to_index"):
    keys = ops.convert_to_tensor(vocabulary_list)
    if keys.dtype.is_integer != dtype.is_integer:
      raise ValueError(
          "Expected %s, got %s." %
          ("integer" if dtype.is_integer else "non-integer", keys.dtype))
    if (not dtype.is_integer) and (keys.dtype.base_dtype != dtype):
      raise ValueError("Expected %s, got %s." % (dtype, keys.dtype))
    num_elements = array_ops.size(keys)
    values = math_ops.cast(math_ops.range(num_elements), dtypes.int64)

    with ops.name_scope(None, "hash_table"):
      table_keys = math_ops.cast(
          keys, dtypes.int64) if keys.dtype.is_integer else keys
      init = KeyValueTensorInitializer(
          table_keys,
          values,
          table_keys.dtype.base_dtype,
          dtypes.int64,
          name="table_init")
      table = StaticHashTableV1(init, default_value)
    if num_oov_buckets:
      table = IdTableWithHashBuckets(
          table,
          num_oov_buckets=num_oov_buckets,
          hasher_spec=hasher_spec,
          key_dtype=dtype)
    return table


def index_to_string_table_from_file(vocabulary_file,
                                    vocab_size=None,
                                    default_value="UNK",
                                    name=None,
                                    key_column_index=TextFileIndex.LINE_NUMBER,
                                    value_column_index=TextFileIndex.WHOLE_LINE,
                                    delimiter="\t"):
  """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The table is initialized from a vocabulary file specified in
  `vocabulary_file`, where the whole line is the value and the
  zero-based line number is the index.

  Any input which does not have a corresponding index in the vocabulary file
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  To specify multi-column vocabulary files, use key_column_index and
  value_column_index and delimiter.

  - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
    expects data type int64.
  - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
    type string.
  - A value >=0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Sample Usages:

  If we have a vocabulary file "test.txt" with the following content:

  ```
  emerson
  lake
  palmer
  ```

  ```python
  indices = tf.constant([1, 5], tf.int64)
  table = tf.lookup.index_to_string_table_from_file(
      vocabulary_file="test.txt", default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.compat.v1.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).
    key_column_index: The column index from the text file to get the `key`
      values from. The default is to use the line number, starting from zero.
    value_column_index: The column index from the text file to get the `value`
      values from. The default is to use the whole line content.
    delimiter: The delimiter to separate fields in a line.

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `vocabulary_file` is empty.
    ValueError: when `vocab_size` is invalid.
  """
  if vocabulary_file is None or (isinstance(vocabulary_file, six.string_types)
                                 and not vocabulary_file):
    raise ValueError("vocabulary_file must be specified and must not be empty.")

  if vocab_size is not None and vocab_size < 1:
    raise ValueError("vocab_size must be greater than 0, got %d." % vocab_size)

  with ops.name_scope(name, "index_to_string"):
    init = TextFileStringTableInitializer(
        vocabulary_file,
        vocab_size=vocab_size,
        name="table_init",
        key_column_index=key_column_index,
        value_column_index=value_column_index,
        delimiter=delimiter)

    # TODO(yleon): Use a more effienct structure.
    return StaticHashTableV1(init, default_value)


def index_to_string_table_from_tensor(vocabulary_list,
                                      default_value="UNK",
                                      name=None):
  """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The mapping is initialized from a string `vocabulary_list` 1-D
  `Tensor` where each element is a value and the corresponding index within the
  tensor is the key.

  Any input which does not have a corresponding index in 'vocabulary_list'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing
  the table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  table = tf.lookup.index_to_string_table_from_tensor(
      vocabulary_list, default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.compat.v1.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    vocabulary_list: A 1-D string `Tensor` that specifies the strings to map
      from indices.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `vocabulary_list` is not set.
  """

  if vocabulary_list is None:
    raise ValueError("vocabulary_list must be specified.")

  with ops.name_scope(name, "index_to_string"):
    vocabulary_list = ops.convert_to_tensor(vocabulary_list, dtypes.string)
    num_elements = array_ops.size(vocabulary_list)
    keys = math_ops.cast(math_ops.range(num_elements), dtypes.int64)

    init = KeyValueTensorInitializer(
        keys, vocabulary_list, dtypes.int64, dtypes.string, name="table_init")
    # TODO(yleon): Use a more effienct structure.
    return StaticHashTableV1(init, default_value)


class MutableHashTable(LookupInterface):
  """A generic mutable hash table implementation.

  Data can be inserted by calling the insert method and removed by calling the
  remove method. It does not support initialization via the init method.

  Example usage:

  ```python
  table = tf.lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64,
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
               name="MutableHashTable",
               checkpoint=True):
    """Creates an empty `MutableHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.

    Returns:
      A `MutableHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._name = name

    self._shared_name = None
    if context.executing_eagerly():
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      # TODO(rohanj): Use context.shared_name() instead.
      self._shared_name = "table_%d" % (ops.uid(),)
    super(MutableHashTable, self).__init__(key_dtype, value_dtype)

    self._resource_handle = self._create_resource()
    if checkpoint:
      saveable = MutableHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def _create_resource(self):
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
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

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

    with ops.name_scope(name, "%s_lookup_table_remove" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)

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
    with ops.name_scope(name, "%s_lookup_table_find" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, keys,
                                                     self._default_value)
    return values

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
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
                        [self.resource_handle, keys, values]):
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys,
                                                   values)
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
                        [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self.resource_handle, self._key_dtype, self._value_dtype)
    return exported_keys, exported_values

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    return {
        "table":
            functools.partial(
                MutableHashTable._Saveable, table=self, name=self._name)
    }

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

    def restore(self, restored_tensors, restored_shapes, name=None):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope(name, "%s_table_restore" % self.name):
        with ops.colocate_with(self.op.resource_handle):
          return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle,
                                                       restored_tensors[0],
                                                       restored_tensors[1])


@tf_export("lookup.experimental.DenseHashTable")
class DenseHashTable(LookupInterface):
  """A generic mutable hash table implementation using tensors as backing store.

  Data can be inserted by calling the insert method and removed by calling the
  remove method. It does not support initialization via the init method.

  It uses "open addressing" with quadratic reprobing to resolve collisions.
  Compared to `MutableHashTable` the insert, remove and lookup operations in a
  `DenseHashTable` are typically faster, but memory usage can be higher.
  However, `DenseHashTable` does not require additional memory for
  temporary tensors created during checkpointing and restore operations.

  Example usage:

  ```python
  table = tf.lookup.DenseHashTable(key_dtype=tf.int64,
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
               name="MutableDenseHashTable",
               checkpoint=True):
    """Creates an empty `DenseHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      empty_key: the key to use to represent empty buckets internally. Must not
        be used in insert, remove or lookup operations.
      deleted_key: the key to use to represent deleted buckets internally. Must
        not be used in insert, remove or lookup operations and be different from
        the empty_key.
      initial_num_buckets: the initial number of buckets.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.

    Returns:
      A `DenseHashTable` object.

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
    self._shared_name = None
    if context.executing_eagerly():
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      # TODO(rohanj): Use context.shared_name() instead.
      self._shared_name = "table_%d" % (ops.uid(),)
    super(DenseHashTable, self).__init__(key_dtype, value_dtype)

    self._resource_handle = self._create_resource()
    if checkpoint:
      saveable = DenseHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def _create_resource(self):
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
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

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
                        [self.resource_handle, keys]):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, keys,
                                                     self._default_value)

    return values

  def insert_or_assign(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
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
                        [self.resource_handle, keys, values]):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      values = ops.convert_to_tensor(
          values, dtype=self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys,
                                                   values)
      return op

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    return self.insert_or_assign(keys, values, name)

  def erase(self, keys, name=None):
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

    with ops.name_scope(name, "%s_lookup_table_remove" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      # pylint: disable=protected-access
      op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)

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
    return self.erase(keys, name)

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self.name,
                        [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self.resource_handle, self._key_dtype, self._value_dtype)

    return exported_keys, exported_values

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    return {
        "table":
            functools.partial(
                DenseHashTable._Saveable, table=self, name=self._name)
    }

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for DenseHashTable."""

    def __init__(self, table, name):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      # pylint: disable=protected-access
      super(DenseHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes, name=None):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope(name, "%s_table_restore" % self.name):
        with ops.colocate_with(self.op.resource_handle):
          return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle,
                                                       restored_tensors[0],
                                                       restored_tensors[1])


ops.NotDifferentiable("LookupTableFind")
ops.NotDifferentiable("LookupTableFindV2")
ops.NotDifferentiable("LookupTableInsert")
ops.NotDifferentiable("LookupTableInsertV2")
ops.NotDifferentiable("LookupTableSize")
ops.NotDifferentiable("LookupTableSizeV2")
ops.NotDifferentiable("HashTable")
ops.NotDifferentiable("HashTableV2")
ops.NotDifferentiable("InitializeTable")
ops.NotDifferentiable("InitializeTableV2")
ops.NotDifferentiable("InitializeTableFromTextFile")
ops.NotDifferentiable("InitializeTableFromTextFileV2")
ops.NotDifferentiable("MutableDenseHashTable")
ops.NotDifferentiable("MutableDenseHashTableV2")
ops.NotDifferentiable("MutableHashTable")
ops.NotDifferentiable("MutableHashTableV2")
ops.NotDifferentiable("MutableHashTableOfTensors")
ops.NotDifferentiable("MutableHashTableOfTensorsV2")
