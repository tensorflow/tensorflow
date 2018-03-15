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

from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import shuffle_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import deprecation

_ACCEPTABLE_CSV_TYPES = (dtypes.float32, dtypes.float64, dtypes.int32,
                         dtypes.int64, dtypes.string)


def make_csv_dataset(
    file_pattern,
    batch_size,
    column_keys,
    column_defaults,
    label_key=None,
    field_delim=",",
    use_quote_delim=True,
    skip=0,
    filter_fn=None,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=1,
):
  """Reads CSV files into a dataset.

  Reads CSV files into a dataset, where each element is a (features, labels)
  tuple that corresponds to a batch of CSV rows. The features dictionary
  maps feature column names to `Tensor`s containing the corresponding
  feature data, and labels is a `Tensor` containing the batch's label data.

  Args:
    file_pattern: List of files or patterns of file paths containing CSV
      records. See @{tf.gfile.Glob} for pattern rules.
    batch_size: An int representing the number of consecutive elements of this
      dataset to combine in a single batch.
    column_keys: A list of strings that corresponds to the CSV columns, in
      order. One per column of the input record.
    column_defaults: A list of default values for the CSV fields. One item per
      column of the input record. Each item in the list is either one of the
      following dtypes: float32, float64, int32, int64, or string, or a
      `Tensor` with one of the aforementioned types. One item per column of
      the input record, with either scalar default value for that column if it
      is required, or, if the column is required, an empty `Tensor` or a dtype.
    label_key: A optional string corresponding to the label column. If provided,
      the data for this column is returned as a separate `Tensor` from the
      features dictionary, so that the dataset complies with the format expected
      by a `tf.Estimator.train` or `tf.Estimator.evaluate` input function.
    field_delim: An optional `string`. Defaults to `","`. Char delimiter to
      separate fields in a record.
    use_quote_delim: An optional bool. Defaults to `True`. If false, treats
      double quotation marks as regular characters inside of the string fields.
    skip: An integer that corresponds to the number of lines to skip at the
      head of each CSV file. Defaults to 0.
    filter_fn: A callable function that takes in a CSV string and returns a
      boolean that corresponds to whether the record should be included. If
      None, does not filter records.
    num_epochs: An int specifying the number of times this dataset is repeated.
      If None, cycles through the dataset forever.
    shuffle: A bool that indicates whether the input should be shuffled.
    shuffle_buffer_size: Buffer size to use for shuffling. A large buffer size
      ensures better shuffling, but would increase memory usage and startup
      time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: An int specifying the number of feature batches to
      prefetch for performance improvement. Recommended value is the number of
      batches consumed per training step.

  Returns:
    A dataset, where each element is a (features, labels) tuple that corresponds
    to a batch of `batch_size` CSV rows. The features dictionary maps feature
    column names to `Tensor`s containing the corresponding column data, and
    labels is a `Tensor` containing the column data for the label column
    specified by `label_key`.
  """
  filenames = _get_file_names(file_pattern, False)
  column_defaults = [
      constant_op.constant([], dtype=x) if x in _ACCEPTABLE_CSV_TYPES else x
      for x in column_defaults
  ]

  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if label_key is not None:
    assert label_key in column_keys

  def filename_to_dataset(filename):
    ds = core_readers.TextLineDataset(filename)
    if skip > 0:
      ds = ds.skip(skip)
    if filter_fn is not None:
      ds = ds.filter(filter_fn)
    return ds

  def decode_csv(line):
    """Decodes csv line into features.

    Args:
      line: String tensor corresponding to one csv record.
    Returns:
      A dictionary of feature names to values for that particular record. If
      label_key is provided, extracts the label feature to be returned as the
      second element of the tuple.
    """
    columns = parsing_ops.decode_csv(
        line,
        column_defaults,
        field_delim=field_delim,
        use_quote_delim=use_quote_delim)
    features = dict(zip(column_keys, columns))
    if label_key is not None:
      label = features.pop(label_key)
      return features, label
    return features

  # TODO(rachelim): interleave records from files for better shuffling
  dataset = dataset.flat_map(filename_to_dataset)
  # TODO(rachelim): use fused shuffle_and_repeat for perf
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)

  dataset = dataset.batch(batch_size)
  dataset = dataset.map(decode_csv)
  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset


def make_batched_features_dataset(file_pattern,
                                  batch_size,
                                  features,
                                  reader=core_readers.TFRecordDataset,
                                  reader_args=None,
                                  num_epochs=None,
                                  shuffle=True,
                                  shuffle_buffer_size=10000,
                                  shuffle_seed=None,
                                  prefetch_buffer_size=1,
                                  reader_num_threads=1,
                                  parser_num_threads=2,
                                  sloppy_ordering=False):
  """Returns a `Dataset` of feature dictionaries from `Example` protos.

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
    reader: A function or class that can be
      called with a `filenames` tensor and (optional) `reader_args` and returns
      a `Dataset` of `Example` tensors. Defaults to `tf.data.TFRecordDataset`.
    reader_args: Additional arguments to pass to the reader class.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever. Defaults to `None`.
    shuffle: A boolean, indicates whether the input should be shuffled. Defaults
      to `True`.
    shuffle_buffer_size: Buffer size of the ShuffleDataset. A large capacity
      ensures better shuffling but would increase memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: Number of feature batches to prefetch in order to
      improve performance. Recommended value is the number of batches consumed
      per training step (default is 1).
    reader_num_threads: Number of threads used to read `Example` records. If >1,
      the results will be interleaved.
    parser_num_threads: Number of threads to use for parsing `Example` tensors
      into a dictionary of `Feature` tensors.
    sloppy_ordering: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.

  Returns:
    A dataset of `dict` elements. Each `dict` maps feature keys to
    `Tensor` or `SparseTensor` objects.
  """
  # Create dataset of all matching filenames
  if shuffle:
    dataset = dataset_ops.Dataset.list_files(file_pattern, shuffle=True)
  else:
    # TODO(b/73959787): Use Dataset.list_files() once ordering is deterministic.
    filenames = _get_file_names(file_pattern, shuffle)
    dataset = dataset_ops.Dataset.from_tensor_slices(filenames)

  # Read `Example` records from files as tensor objects.
  if reader_args is None:
    reader_args = []

  # Read files sequentially (if reader_num_threads=1) or in parallel
  dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          sloppy=sloppy_ordering))

  # Extract values if the `Example` tensors are stored as key-value tuples.
  if dataset.output_types == (dtypes.string, dtypes.string):
    dataset = dataset.map(lambda _, v: v)

  # Apply dataset repeat and shuffle transformations.
  repeat_dataset = (num_epochs != 1)
  if repeat_dataset and shuffle:
    # Used fused shuffle_and_repeat operation for better performance
    dataset = dataset.apply(
        shuffle_ops.shuffle_and_repeat(shuffle_buffer_size, num_epochs,
                                       shuffle_seed))
  elif repeat_dataset:
    dataset = dataset.repeat(num_epochs)
  elif shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)

  dataset = dataset.batch(batch_size)

  # Parse `Example` tensors to a dictionary of `Feature` tensors.
  dataset = dataset.map(
      lambda x: parsing_ops.parse_example(x, features),
      num_parallel_calls=parser_num_threads)

  # TODO(rachelim): Add an optional label_key argument for extracting the label
  # from the features dictionary, to comply with the type expected by the
  # input_fn to a `tf.Estimator.train` or `tf.Estimator.evaluate` function.
  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset


@deprecation.deprecated(None,
                        "Use `tf.contrib.data.make_batched_features_dataset`")
def read_batch_features(file_pattern,
                        batch_size,
                        features,
                        reader=core_readers.TFRecordDataset,
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
    reader: A function or class that can be
      called with a `filenames` tensor and (optional) `reader_args` and returns
      a `Dataset` of `Example` tensors. Defaults to `tf.data.TFRecordDataset`.
    reader_args: Additional arguments to pass to the reader class.
    randomize_input: Whether the input should be randomized.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever.
    capacity: Buffer size of the ShuffleDataset. A large capacity ensures better
      shuffling but would increase memory usage and startup time.
  Returns:
    A dict from keys in features to `Tensor` or `SparseTensor` objects.
  """
  dataset = make_batched_features_dataset(
      file_pattern,
      batch_size,
      features,
      reader=reader,
      reader_args=reader_args,
      shuffle=randomize_input,
      num_epochs=num_epochs,
      shuffle_buffer_size=capacity)
  iterator = dataset.make_one_shot_iterator()
  outputs = iterator.get_next()
  return outputs


def _get_file_names(file_pattern, shuffle):
  """Parse list of file names from pattern, optionally shuffled.

  Args:
    file_pattern: File glob pattern, or list of glob patterns.
    shuffle: Whether to shuffle the order of file names.

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
  if not shuffle:
    file_names = sorted(file_names)
  return file_names


class SqlDataset(dataset_ops.Dataset):
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
    super(SqlDataset, self).__init__()
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
  def output_classes(self):
    return nest.map_structure(lambda _: ops.Tensor, self._output_types)

  @property
  def output_shapes(self):
    return nest.map_structure(lambda _: tensor_shape.TensorShape([]),
                              self._output_types)

  @property
  def output_types(self):
    return self._output_types
