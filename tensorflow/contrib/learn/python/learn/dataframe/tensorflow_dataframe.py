# Copyright 2016 Google Inc. All Rights Reserved.
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

"""TensorFlowDataFrame implements convenience functions using TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.dataframe import dataframe as df
from tensorflow.contrib.learn.python.learn.dataframe.transforms import batch
from tensorflow.contrib.learn.python.learn.dataframe.transforms import csv_parser
from tensorflow.contrib.learn.python.learn.dataframe.transforms import example_parser
from tensorflow.contrib.learn.python.learn.dataframe.transforms import in_memory_source
from tensorflow.contrib.learn.python.learn.dataframe.transforms import reader_source
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile


class FileFormat(object):
  TEXT = 0
  TFRECORD = 1

FILE_FORMAT_TO_READER_CLS = {
    FileFormat.TEXT: io_ops.TextLineReader,
    FileFormat.TFRECORD: io_ops.TFRecordReader
}


def _expand_file_names(filepatterns):
  """Takes a list of file patterns and returns a list of resolved file names."""
  if not isinstance(filepatterns, (list, tuple, set)):
    filepatterns = [filepatterns]
  filenames = set()
  for filepattern in filepatterns:
    names = set(gfile.Glob(filepattern))
    filenames |= names
  return list(filenames)


class TensorFlowDataFrame(df.DataFrame):
  """TensorFlowDataFrame implements convenience functions using TensorFlow."""

  def run(self, num_batches=None, graph=None, session=None, start_queues=True):
    """Builds and runs the columns of the `DataFrame` and yields batches.

    This is a generator that yields a dictionary mapping column names to
    evaluated columns.
    Args:
      num_batches: the maximum number of batches to produce. If none specified,
        the returned value will iterate through infinite batches.
      graph: the `Graph` in which the `DataFrame` should be built.
      session: the `Session` in which to run the columns of the `DataFrame`.
      start_queues: if true, queues will be started before running and halted
        after producting `n` batches.

    Yields:
      A dictionary, mapping column names to the values resulting from running
      each column for a single batch.
    """
    if graph is None:
      graph = tf.get_default_graph()
    with graph.as_default():
      if session is None:
        session = tf.Session()
      self_built = self.build()
      keys = list(self_built.keys())
      cols = list(self_built.values())
      if start_queues:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
      i = 0
      while num_batches is None or i < num_batches:
        i += 1
        values = session.run(cols)
        yield collections.OrderedDict(zip(keys, values))
      if start_queues:
        coord.request_stop()
        coord.join(threads)

  def run_once(self):
    """Creates a new 'Graph` and `Session` and runs a single batch.

    Returns:
      A dictionary mapping column names to numpy arrays that contain a single
      batch of the `DataFrame`.
    """
    return list(self.run(num_batches=1))[0]

  def batch(self,
            batch_size,
            shuffle=False,
            num_threads=1,
            queue_capacity=None,
            min_after_dequeue=None,
            seed=None):
    """Resize the batches in the `DataFrame` to the given `batch_size`.

    Args:
      batch_size: desired batch size.
      shuffle: whether records should be shuffled. Defaults to true.
      num_threads: the number of enqueueing threads.
      queue_capacity: capacity of the queue that will hold new batches.
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `DataFrame` with `batch_size` rows.
    """
    column_names = list(self._columns.keys())
    if shuffle:
      batcher = batch.ShuffleBatch(batch_size,
                                   output_names=column_names,
                                   num_threads=num_threads,
                                   queue_capacity=queue_capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   seed=seed)
    else:
      batcher = batch.Batch(batch_size,
                            output_names=column_names,
                            num_threads=num_threads,
                            queue_capacity=queue_capacity)

    batched_series = batcher(list(self._columns.values()))
    dataframe = type(self)()
    dataframe.assign(**(dict(zip(column_names, batched_series))))
    return dataframe

  @classmethod
  def from_csv(cls,
               filepatterns,
               batch_size,
               default_values,
               has_header=True,
               column_names=None,
               shuffle=True,
               num_threads=1,
               queue_capacity=None,
               min_after_dequeue=None,
               seed=None):
    """Create a `DataFrame` from `tensorflow.Example`s.

    If `has_header` is false, then `column_names` must be specified. If
    `has_header` is true and `column_names` are specified, then `column_names`
    overrides the names in the header.

    Args:
      filepatterns: a list of file patterns that resolve to CSV files.
      batch_size: desired batch size.
      default_values: a list of default values for each column.
      has_header: whether or not the CSV files have headers.
      column_names: a list of names for the columns in the CSV files.
      shuffle: whether records should be shuffled. Defaults to true.
      num_threads: the number of readers that will work in parallel.
      queue_capacity: capacity of the queue that will store parsed lines.
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `DataFrame` that has columns corresponding to `features` and is filled
      with `Example`s from `filepatterns`.

    Raises:
      ValueError: no files match `filepatterns`.
      ValueError: `features` contains the reserved name 'index'.
    """
    filenames = _expand_file_names(filepatterns)
    if not filenames:
      raise ValueError("No matching file names.")

    if column_names is None:
      if not has_header:
        raise ValueError("If column_names is None, has_header must be true.")
      with gfile.GFile(filenames[0]) as f:
        column_names = csv.DictReader(f).fieldnames

    if "index" in column_names:
      raise ValueError(
          "'index' is reserved and can not be used for a column name.")

    reader_kwargs = {"skip_header_lines": (1 if has_header else 0)}
    index, value = reader_source.TextFileSource(
        filenames,
        reader_kwargs=reader_kwargs,
        batch_size=batch_size,
        queue_capacity=queue_capacity,
        shuffle=shuffle,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads,
        seed=seed)()
    parser = csv_parser.CSVParser(column_names, default_values)
    parsed = parser(value)

    column_dict = parsed._asdict()
    column_dict["index"] = index

    dataframe = cls()
    dataframe.assign(**column_dict)
    return dataframe

  @classmethod
  def from_examples(cls,
                    filepatterns,
                    batch_size,
                    features,
                    file_format=FileFormat.TFRECORD,
                    shuffle=True,
                    num_threads=1,
                    queue_capacity=None,
                    min_after_dequeue=None,
                    seed=None):
    """Create a `DataFrame` from `tensorflow.Example`s.

    Args:
      filepatterns: a list of file patterns containing `tensorflow.Example`s.
      batch_size: desired batch size.
      features: a dict mapping feature names to `VarLenFeature` or
        `FixedLenFeature`.
      file_format: a `FileFormat` indicating the format of the files in
        `filepatterns`.
      shuffle: whether records should be shuffled. Defaults to true.
      num_threads: the number of readers that will work in parallel.
      queue_capacity: capacity of the queue that will store parsed `Example`s
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `DataFrame` that has columns corresponding to `features` and is filled
      with `Example`s from `filepatterns`.

    Raises:
      ValueError: no files match `filepatterns`.
      ValueError: `features` contains the reserved name 'index'.
    """
    filenames = _expand_file_names(filepatterns)
    if not filenames:
      raise ValueError("No matching file names.")

    if "index" in features:
      raise ValueError(
          "'index' is reserved and can not be used for a feature name.")

    index, record = reader_source.ReaderSource(
        FILE_FORMAT_TO_READER_CLS[file_format],
        filenames,
        batch_size=batch_size,
        queue_capacity=queue_capacity,
        shuffle=shuffle,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads,
        seed=seed)()
    parser = example_parser.ExampleParser(features)
    parsed = parser(record)

    column_dict = parsed._asdict()
    column_dict["index"] = index

    dataframe = cls()
    dataframe.assign(**column_dict)
    return dataframe

  @classmethod
  def from_pandas(cls,
                  pandas_dataframe,
                  batch_size=None,
                  shuffle=True,
                  queue_capacity=None,
                  min_after_dequeue=None,
                  seed=None):
    """Create a `tf.learn.DataFrame` from a `pandas.DataFrame`.

    Args:
      pandas_dataframe: `pandas.DataFrame` that serves as a data source.
      batch_size: desired batch size.
      shuffle: whether records should be shuffled. Defaults to true.
      queue_capacity: capacity of the queue that will store parsed `Example`s
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `tf.learn.DataFrame` that contains batches drawn from the given
      `pandas_dataframe`.
    """
    pandas_source = in_memory_source.PandasSource(pandas_dataframe, batch_size,
                                                  queue_capacity, shuffle,
                                                  min_after_dequeue, seed)
    dataframe = cls()
    dataframe.assign(**(pandas_source()._asdict()))
    return dataframe

  @classmethod
  def from_numpy(cls,
                 numpy_array,
                 batch_size=None,
                 shuffle=True,
                 queue_capacity=None,
                 min_after_dequeue=None,
                 seed=None):
    """Create a `tf.learn.DataFrame` from a `numpy.ndarray`.

    The returned `DataFrame` contains two columns: 'index' and 'value'. The
    'value' column contains a row from the array. The 'index' column contains
    the corresponding row number.

    Args:
      numpy_array: `numpy.ndarray` that serves as a data source.
      batch_size: desired batch size.
      shuffle: whether records should be shuffled. Defaults to true.
      queue_capacity: capacity of the queue that will store parsed `Example`s
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `tf.learn.DataFrame` that contains batches drawn from the given
      array.
    """
    numpy_source = in_memory_source.NumpySource(numpy_array, batch_size,
                                                queue_capacity, shuffle,
                                                min_after_dequeue, seed)
    dataframe = cls()
    dataframe.assign(**(numpy_source()._asdict()))
    return dataframe





