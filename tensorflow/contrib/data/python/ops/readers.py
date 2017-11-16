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
# ==============================================================================
"""Python wrappers for reader Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import dataset_ops as contrib_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import deprecation


class TextLineDataset(contrib_dataset_ops.Dataset):
  """A `Dataset` comprising lines from one or more text files."""

  @deprecation.deprecated(None, "Use `tf.data.TextLineDataset`.")
  def __init__(self, filenames, compression_type=None, buffer_size=None):
    """Creates a `TextLineDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
    """
    dataset = readers.TextLineDataset(filenames, compression_type,
                                      buffer_size)
    super(TextLineDataset, self).__init__(dataset)


class TFRecordDataset(contrib_dataset_ops.Dataset):
  """A `Dataset` comprising records from one or more TFRecord files."""

  @deprecation.deprecated(None, "Use `tf.data.TFRecordDataset`.")
  def __init__(self, filenames, compression_type=None, buffer_size=None):
    """Creates a `TFRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. 0 means no buffering.
    """
    dataset = readers.TFRecordDataset(filenames, compression_type,
                                      buffer_size)
    super(TFRecordDataset, self).__init__(dataset)


class FixedLengthRecordDataset(contrib_dataset_ops.Dataset):
  """A `Dataset` of fixed-length records from one or more binary files."""

  @deprecation.deprecated(None, "Use `tf.data.FixedLengthRecordDataset`.")
  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None):
    """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in
        each record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
    """
    dataset = readers.FixedLengthRecordDataset(
        filenames, record_bytes, header_bytes, footer_bytes, buffer_size)
    super(FixedLengthRecordDataset, self).__init__(dataset)


def read_batch_features(file_pattern,
                        batch_size,
                        features,
                        reader,
                        reader_args=None,
                        randomize_input=True,
                        num_epochs=None,
                        capacity=10000):
  """Reads batches of Examples.

  Example:

  ```
  serialized_examples = [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "code", "art" ] } } }
    },
    features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "sports" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  features: {
    "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
    "gender": FixedLenFeature([], dtype=tf.string),
    "kws": VarLenFeature(dtype=tf.string),
  }
  ```

  And the expected output is:

  ```python
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
    "kws": SparseTensor(
      indices=[[0, 0], [0, 1], [1, 0]],
      values=["code", "art", "sports"]
      dense_shape=[2, 2]),
  }
  ```

  Args:
    file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int representing the number of consecutive elements of this
      dataset to combine in a single batch.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. See `tf.parse_example`.
    reader: A function or class that can be called with a `filenames` tensor
      and (optional) `reader_args` and returns a `Dataset` of Examples.
    reader_args: Additional arguments to pass to the reader class.
    randomize_input: Whether the input should be randomized.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever.
    capacity: Capacity of the ShuffleDataset. A large capacity ensures better
      shuffling but would increase memory usage and startup time.

  Returns:
    A dict from keys in features to Tensor or SparseTensor objects.
  """
  filenames = _get_file_names(file_pattern, randomize_input)
  if reader_args:
    dataset = reader(filenames, *reader_args)
  else:
    dataset = reader(filenames)
  if dataset.output_types == (dtypes.string, dtypes.string):
    dataset = dataset.map(lambda _, v: v)
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)
  if randomize_input:
    dataset = dataset.shuffle(capacity)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda x: parsing_ops.parse_example(x, features))
  iterator = dataset.make_one_shot_iterator()
  outputs = iterator.get_next()
  return outputs


def _get_file_names(file_pattern, randomize_input):
  """Parse list of file names from pattern, optionally shuffled.

  Args:
    file_pattern: File glob pattern, or list of glob patterns.
    randomize_input: Whether to shuffle the order of file names.

  Returns:
    List of file names matching `file_pattern`.

  Raises:
    ValueError: If `file_pattern` is empty, or pattern matches no files.
  """
  if isinstance(file_pattern, list):
    if not file_pattern:
      raise ValueError("File pattern is empty.")
    file_names = []
    for entry in file_pattern:
      file_names.extend(gfile.Glob(entry))
  else:
    file_names = list(gfile.Glob(file_pattern))

  if not file_names:
    raise ValueError("No files match %s." % file_pattern)

  # Sort files so it will be deterministic for unit tests.
  if not randomize_input:
    file_names = sorted(file_names)
  return file_names


class SqlDataset(contrib_dataset_ops.Dataset):

  def __init__(self, driver_name, data_source_name, query, output_types):
    dataset = _SqlDataset(driver_name, data_source_name, query, output_types)
    super(SqlDataset, self).__init__(dataset)


class _SqlDataset(dataset_ops.Dataset):
  """A `Dataset` consisting of the results from a SQL query."""

  def __init__(self, driver_name, data_source_name, query, output_types):
    """Creates a `SqlDataset`.

    `SqlDataset` allows a user to read data from the result set of a SQL query.
    For example:

    ```python
    dataset = tf.contrib.data.SqlDataset("sqlite", "/foo/bar.sqlite3",
                                         "SELECT name, age FROM people",
                                         (tf.string, tf.int32))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # Prints the rows of the result set of the above query.
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    ```

    Args:
      driver_name: A 0-D `tf.string` tensor containing the database type.
        Currently, the only supported value is 'sqlite'.
      data_source_name: A 0-D `tf.string` tensor containing a connection string
        to connect to the database.
      query: A 0-D `tf.string` tensor containing the SQL query to execute.
      output_types: A tuple of `tf.DType` objects representing the types of the
        columns returned by `query`.
    """
    super(_SqlDataset, self).__init__()
    self._driver_name = ops.convert_to_tensor(
        driver_name, dtype=dtypes.string, name="driver_name")
    self._data_source_name = ops.convert_to_tensor(
        data_source_name, dtype=dtypes.string, name="data_source_name")
    self._query = ops.convert_to_tensor(
        query, dtype=dtypes.string, name="query")
    self._output_types = output_types

  def _as_variant_tensor(self):
    return gen_dataset_ops.sql_dataset(self._driver_name,
                                       self._data_source_name, self._query,
                                       nest.flatten(self.output_types),
                                       nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return nest.map_structure(lambda _: tensor_shape.TensorShape([]),
                              self._output_types)

  @property
  def output_types(self):
    return self._output_types
