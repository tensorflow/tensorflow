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

"""TensorFlowDataFrame implements convenience functions using TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe import dataframe as df
from tensorflow.contrib.learn.python.learn.dataframe.transforms import batch
from tensorflow.contrib.learn.python.learn.dataframe.transforms import csv_parser
from tensorflow.contrib.learn.python.learn.dataframe.transforms import example_parser
from tensorflow.contrib.learn.python.learn.dataframe.transforms import in_memory_source
from tensorflow.contrib.learn.python.learn.dataframe.transforms import reader_source
from tensorflow.contrib.learn.python.learn.dataframe.transforms import sparsify
from tensorflow.contrib.learn.python.learn.dataframe.transforms import split_mask
from tensorflow.python.client import session as sess
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner as qr


def _expand_file_names(filepatterns):
  """Takes a list of file patterns and returns a list of resolved file names."""
  if not isinstance(filepatterns, (list, tuple, set)):
    filepatterns = [filepatterns]
  filenames = set()
  for filepattern in filepatterns:
    names = set(gfile.Glob(filepattern))
    filenames |= names
  return list(filenames)


def _dtype_to_nan(dtype):
  if dtype is dtypes.string:
    return b""
  elif dtype.is_integer:
    return np.nan
  elif dtype.is_floating:
    return np.nan
  elif dtype is dtypes.bool:
    return np.nan
  else:
    raise ValueError("Can't parse type without NaN into sparse tensor: %s" %
                     dtype)


def _get_default_value(feature_spec):
  if isinstance(feature_spec, parsing_ops.FixedLenFeature):
    return feature_spec.default_value
  else:
    return _dtype_to_nan(feature_spec.dtype)


class TensorFlowDataFrame(df.DataFrame):
  """TensorFlowDataFrame implements convenience functions using TensorFlow."""

  def run(self,
          num_batches=None,
          graph=None,
          session=None,
          start_queues=True,
          initialize_variables=True,
          **kwargs):
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
      initialize_variables: if true, variables will be initialized.
      **kwargs: Additional keyword arguments e.g. `num_epochs`.

    Yields:
      A dictionary, mapping column names to the values resulting from running
      each column for a single batch.
    """
    if graph is None:
      graph = ops.get_default_graph()
    with graph.as_default():
      if session is None:
        session = sess.Session()
      self_built = self.build(**kwargs)
      keys = list(self_built.keys())
      cols = list(self_built.values())
      if initialize_variables:
        if variables.local_variables():
          session.run(variables.initialize_local_variables())
        if variables.all_variables():
          session.run(variables.initialize_all_variables())
      if start_queues:
        coord = coordinator.Coordinator()
        threads = qr.start_queue_runners(sess=session, coord=coord)
      i = 0
      while num_batches is None or i < num_batches:
        i += 1
        try:
          values = session.run(cols)
          yield collections.OrderedDict(zip(keys, values))
        except errors.OutOfRangeError:
          break
      if start_queues:
        coord.request_stop()
        coord.join(threads)

  def select_rows(self, boolean_series):
    """Returns a `DataFrame` with only the rows indicated by `boolean_series`.

    Note that batches may no longer have consistent size after calling
    `select_rows`, so the new `DataFrame` may need to be rebatched.
    For example:
    '''
    filtered_df = df.select_rows(df["country"] == "jp").batch(64)
    '''

    Args:
      boolean_series: a `Series` that evaluates to a boolean `Tensor`.

    Returns:
      A new `DataFrame` with the same columns as `self`, but selecting only the
      rows where `boolean_series` evaluated to `True`.
    """
    result = type(self)()
    for key, col in self._columns.items():
      try:
        result[key] = col.select_rows(boolean_series)
      except AttributeError as e:
        raise NotImplementedError((
            "The select_rows method is not implemented for Series type {}. "
            "Original error: {}").format(type(col), e))
    return result

  def split(self, index_series, proportion, batch_size=None):
    """Deterministically split a `DataFrame` into two `DataFrame`s.

    Note this split is only as deterministic as the underlying hash function;
    see `tf.string_to_hash_bucket_fast`.  The hash function is deterministic
    for a given binary, but may change occasionally.  The only way to achieve
    an absolute guarantee that the split `DataFrame`s do not change across runs
    is to materialize them.

    Note too that the allocation of a row to one partition or the
    other is evaluated independently for each row, so the exact number of rows
    in each partition is binomially distributed.

    Args:
      index_series: a `Series` of unique strings, whose hash will determine the
        partitioning; or the name in this `DataFrame` of such a `Series`.
        (This `Series` must contain strings because TensorFlow provides hash
        ops only for strings, and there are no number-to-string converter ops.)
      proportion: The proportion of the rows to select for the 'left'
        partition; the remaining (1 - proportion) rows form the 'right'
        partition.
      batch_size: the batch size to use when rebatching the left and right
        `DataFrame`s.  If None (default), the `DataFrame`s are not rebatched;
        thus their batches will have variable sizes, according to which rows
        are selected from each batch of the original `DataFrame`.

    Returns:
      Two `DataFrame`s containing the partitioned rows.
    """
    if isinstance(index_series, str):
      index_series = self[index_series]
    left_mask, = split_mask.SplitMask(proportion)(index_series)
    right_mask = ~left_mask
    left_rows = self.select_rows(left_mask)
    right_rows = self.select_rows(right_mask)

    if batch_size:
      left_rows = left_rows.batch(batch_size=batch_size, shuffle=False)
      right_rows = right_rows.batch(batch_size=batch_size, shuffle=False)

    return left_rows, right_rows

  def split_fast(self, index_series, proportion, batch_size,
                 base_batch_size=1000):
    """Deterministically split a `DataFrame` into two `DataFrame`s.

    Note this split is only as deterministic as the underlying hash function;
    see `tf.string_to_hash_bucket_fast`.  The hash function is deterministic
    for a given binary, but may change occasionally.  The only way to achieve
    an absolute guarantee that the split `DataFrame`s do not change across runs
    is to materialize them.

    Note too that the allocation of a row to one partition or the
    other is evaluated independently for each row, so the exact number of rows
    in each partition is binomially distributed.

    Args:
      index_series: a `Series` of unique strings, whose hash will determine the
        partitioning; or the name in this `DataFrame` of such a `Series`.
        (This `Series` must contain strings because TensorFlow provides hash
        ops only for strings, and there are no number-to-string converter ops.)
      proportion: The proportion of the rows to select for the 'left'
        partition; the remaining (1 - proportion) rows form the 'right'
        partition.
      batch_size: the batch size to use when rebatching the left and right
        `DataFrame`s.  If None (default), the `DataFrame`s are not rebatched;
        thus their batches will have variable sizes, according to which rows
        are selected from each batch of the original `DataFrame`.
      base_batch_size: the batch size to use for materialized data, prior to the
        split.

    Returns:
      Two `DataFrame`s containing the partitioned rows.
    """
    if isinstance(index_series, str):
      index_series = self[index_series]
    left_mask, = split_mask.SplitMask(proportion)(index_series)
    right_mask = ~left_mask
    self["left_mask__"] = left_mask
    self["right_mask__"] = right_mask
    # TODO(soergel): instead of base_batch_size can we just do one big batch?
    # avoid computing the hashes twice
    m = self.materialize_to_memory(batch_size=base_batch_size)
    left_rows_df = m.select_rows(m["left_mask__"])
    right_rows_df = m.select_rows(m["right_mask__"])
    del left_rows_df[["left_mask__", "right_mask__"]]
    del right_rows_df[["left_mask__", "right_mask__"]]

    # avoid recomputing the split repeatedly
    left_rows_df = left_rows_df.materialize_to_memory(batch_size=batch_size)
    right_rows_df = right_rows_df.materialize_to_memory(batch_size=batch_size)
    return left_rows_df, right_rows_df

  def run_one_batch(self):
    """Creates a new 'Graph` and `Session` and runs a single batch.

    Returns:
      A dictionary mapping column names to numpy arrays that contain a single
      batch of the `DataFrame`.
    """
    return list(self.run(num_batches=1))[0]

  def run_one_epoch(self):
    """Creates a new 'Graph` and `Session` and runs a single epoch.

    Naturally this makes sense only for DataFrames that fit in memory.

    Returns:
      A dictionary mapping column names to numpy arrays that contain a single
      epoch of the `DataFrame`.
    """
    # batches is a list of dicts of numpy arrays
    batches = [b for b in self.run(num_epochs=1)]

    # first invert that to make a dict of lists of numpy arrays
    pivoted_batches = {}
    for k in batches[0].keys():
      pivoted_batches[k] = []
    for b in batches:
      for k, v in b.items():
        pivoted_batches[k].append(v)

    # then concat the arrays in each column
    result = {k: np.concatenate(column_batches)
              for k, column_batches in pivoted_batches.items()}
    return result

  def materialize_to_memory(self, batch_size):
    unordered_dict_of_arrays = self.run_one_epoch()

    # there may already be an 'index' column, in which case from_ordereddict)
    # below will complain because it wants to generate a new one.
    # for now, just remove it.
    # TODO(soergel): preserve index history, potentially many levels deep
    del unordered_dict_of_arrays["index"]

    # the order of the columns in this dict is arbitrary; we just need it to
    # remain consistent.
    ordered_dict_of_arrays = collections.OrderedDict(unordered_dict_of_arrays)
    return TensorFlowDataFrame.from_ordereddict(ordered_dict_of_arrays,
                                                batch_size=batch_size)

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
  def _from_csv_base(cls, filepatterns, get_default_values, has_header,
                     column_names, num_threads, enqueue_size,
                     batch_size, queue_capacity, min_after_dequeue, shuffle,
                     seed):
    """Create a `DataFrame` from CSV files.

    If `has_header` is false, then `column_names` must be specified. If
    `has_header` is true and `column_names` are specified, then `column_names`
    overrides the names in the header.

    Args:
      filepatterns: a list of file patterns that resolve to CSV files.
      get_default_values: a function that produces a list of default values for
        each column, given the column names.
      has_header: whether or not the CSV files have headers.
      column_names: a list of names for the columns in the CSV files.
      num_threads: the number of readers that will work in parallel.
      enqueue_size: block size for each read operation.
      batch_size: desired batch size.
      queue_capacity: capacity of the queue that will store parsed lines.
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      shuffle: whether records should be shuffled. Defaults to true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `DataFrame` that has columns corresponding to `features` and is filled
      with examples from `filepatterns`.

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

    default_values = get_default_values(column_names)

    reader_kwargs = {"skip_header_lines": (1 if has_header else 0)}
    index, value = reader_source.TextFileSource(
        filenames,
        reader_kwargs=reader_kwargs,
        enqueue_size=enqueue_size,
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
  def from_csv(cls,
               filepatterns,
               default_values,
               has_header=True,
               column_names=None,
               num_threads=1,
               enqueue_size=None,
               batch_size=32,
               queue_capacity=None,
               min_after_dequeue=None,
               shuffle=True,
               seed=None):
    """Create a `DataFrame` from CSV files.

    If `has_header` is false, then `column_names` must be specified. If
    `has_header` is true and `column_names` are specified, then `column_names`
    overrides the names in the header.

    Args:
      filepatterns: a list of file patterns that resolve to CSV files.
      default_values: a list of default values for each column.
      has_header: whether or not the CSV files have headers.
      column_names: a list of names for the columns in the CSV files.
      num_threads: the number of readers that will work in parallel.
      enqueue_size: block size for each read operation.
      batch_size: desired batch size.
      queue_capacity: capacity of the queue that will store parsed lines.
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      shuffle: whether records should be shuffled. Defaults to true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `DataFrame` that has columns corresponding to `features` and is filled
      with examples from `filepatterns`.

    Raises:
      ValueError: no files match `filepatterns`.
      ValueError: `features` contains the reserved name 'index'.
    """

    def get_default_values(column_names):
      # pylint: disable=unused-argument
      return default_values

    return cls._from_csv_base(filepatterns, get_default_values, has_header,
                              column_names, num_threads,
                              enqueue_size, batch_size, queue_capacity,
                              min_after_dequeue, shuffle, seed)

  @classmethod
  def from_csv_with_feature_spec(cls,
                                 filepatterns,
                                 feature_spec,
                                 has_header=True,
                                 column_names=None,
                                 num_threads=1,
                                 enqueue_size=None,
                                 batch_size=32,
                                 queue_capacity=None,
                                 min_after_dequeue=None,
                                 shuffle=True,
                                 seed=None):
    """Create a `DataFrame` from CSV files, given a feature_spec.

    If `has_header` is false, then `column_names` must be specified. If
    `has_header` is true and `column_names` are specified, then `column_names`
    overrides the names in the header.

    Args:
      filepatterns: a list of file patterns that resolve to CSV files.
      feature_spec: a dict mapping column names to `FixedLenFeature` or
          `VarLenFeature`.
      has_header: whether or not the CSV files have headers.
      column_names: a list of names for the columns in the CSV files.
      num_threads: the number of readers that will work in parallel.
      enqueue_size: block size for each read operation.
      batch_size: desired batch size.
      queue_capacity: capacity of the queue that will store parsed lines.
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      shuffle: whether records should be shuffled. Defaults to true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.

    Returns:
      A `DataFrame` that has columns corresponding to `features` and is filled
      with examples from `filepatterns`.

    Raises:
      ValueError: no files match `filepatterns`.
      ValueError: `features` contains the reserved name 'index'.
    """

    def get_default_values(column_names):
      return [_get_default_value(feature_spec[name]) for name in column_names]

    dataframe = cls._from_csv_base(filepatterns, get_default_values, has_header,
                                   column_names, num_threads,
                                   enqueue_size, batch_size, queue_capacity,
                                   min_after_dequeue, shuffle, seed)

    # replace the dense columns with sparse ones in place in the dataframe
    for name in dataframe.columns():
      if name != "index" and isinstance(feature_spec[name],
                                        parsing_ops.VarLenFeature):
        strip_value = _get_default_value(feature_spec[name])
        (dataframe[name],) = sparsify.Sparsify(strip_value)(dataframe[name])

    return dataframe

  @classmethod
  def from_examples(cls,
                    filepatterns,
                    features,
                    reader_cls=io_ops.TFRecordReader,
                    num_threads=1,
                    enqueue_size=None,
                    batch_size=32,
                    queue_capacity=None,
                    min_after_dequeue=None,
                    shuffle=True,
                    seed=None):
    """Create a `DataFrame` from `tensorflow.Example`s.

    Args:
      filepatterns: a list of file patterns containing `tensorflow.Example`s.
      features: a dict mapping feature names to `VarLenFeature` or
        `FixedLenFeature`.
      reader_cls: a subclass of `tensorflow.ReaderBase` that will be used to
        read the `Example`s.
      num_threads: the number of readers that will work in parallel.
      enqueue_size: block size for each read operation.
      batch_size: desired batch size.
      queue_capacity: capacity of the queue that will store parsed `Example`s
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      shuffle: whether records should be shuffled. Defaults to true.
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
        reader_cls,
        filenames,
        enqueue_size=enqueue_size,
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
                  num_threads=None,
                  enqueue_size=None,
                  batch_size=None,
                  queue_capacity=None,
                  min_after_dequeue=None,
                  shuffle=True,
                  seed=None,
                  data_name="pandas_data"):
    """Create a `tf.learn.DataFrame` from a `pandas.DataFrame`.

    Args:
      pandas_dataframe: `pandas.DataFrame` that serves as a data source.
      num_threads: the number of threads to use for enqueueing.
      enqueue_size: the number of rows to enqueue per step.
      batch_size: desired batch size.
      queue_capacity: capacity of the queue that will store parsed `Example`s
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      shuffle: whether records should be shuffled. Defaults to true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.
      data_name: a scope name identifying the data.

    Returns:
      A `tf.learn.DataFrame` that contains batches drawn from the given
      `pandas_dataframe`.
    """
    pandas_source = in_memory_source.PandasSource(
        pandas_dataframe,
        num_threads=num_threads,
        enqueue_size=enqueue_size,
        batch_size=batch_size,
        queue_capacity=queue_capacity,
        shuffle=shuffle,
        min_after_dequeue=min_after_dequeue,
        seed=seed,
        data_name=data_name)
    dataframe = cls()
    dataframe.assign(**(pandas_source()._asdict()))
    return dataframe

  @classmethod
  def from_numpy(cls,
                 numpy_array,
                 num_threads=None,
                 enqueue_size=None,
                 batch_size=None,
                 queue_capacity=None,
                 min_after_dequeue=None,
                 shuffle=True,
                 seed=None,
                 data_name="numpy_data"):
    """Creates a `tf.learn.DataFrame` from a `numpy.ndarray`.

    The returned `DataFrame` contains two columns: 'index' and 'value'. The
    'value' column contains a row from the array. The 'index' column contains
    the corresponding row number.

    Args:
      numpy_array: `numpy.ndarray` that serves as a data source.
      num_threads: the number of threads to use for enqueueing.
      enqueue_size: the number of rows to enqueue per step.
      batch_size: desired batch size.
      queue_capacity: capacity of the queue that will store parsed `Example`s
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      shuffle: whether records should be shuffled. Defaults to true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.
      data_name: a scope name identifying the data.

    Returns:
      A `tf.learn.DataFrame` that contains batches drawn from the given
      array.
    """
    numpy_source = in_memory_source.NumpySource(
        numpy_array,
        num_threads=num_threads,
        enqueue_size=enqueue_size,
        batch_size=batch_size,
        queue_capacity=queue_capacity,
        shuffle=shuffle,
        min_after_dequeue=min_after_dequeue,
        seed=seed,
        data_name=data_name)
    dataframe = cls()
    dataframe.assign(**(numpy_source()._asdict()))
    return dataframe

  @classmethod
  def from_ordereddict(cls,
                       ordered_dict_of_arrays,
                       num_threads=None,
                       enqueue_size=None,
                       batch_size=None,
                       queue_capacity=None,
                       min_after_dequeue=None,
                       shuffle=True,
                       seed=None,
                       data_name="numpy_data"):
    """Creates a `tf.learn.DataFrame` from an `OrderedDict` of `numpy.ndarray`.

    The returned `DataFrame` contains a column for each key of the dict plus an
    extra 'index' column. The 'index' column contains the row number. Each of
    the other columns contains a row from the corresponding array.

    Args:
      ordered_dict_of_arrays: `OrderedDict` of `numpy.ndarray` that serves as a
          data source.
      num_threads: the number of threads to use for enqueueing.
      enqueue_size: the number of rows to enqueue per step.
      batch_size: desired batch size.
      queue_capacity: capacity of the queue that will store parsed `Example`s
      min_after_dequeue: minimum number of elements that can be left by a
        dequeue operation. Only used if `shuffle` is true.
      shuffle: whether records should be shuffled. Defaults to true.
      seed: passed to random shuffle operations. Only used if `shuffle` is true.
      data_name: a scope name identifying the data.

    Returns:
      A `tf.learn.DataFrame` that contains batches drawn from the given arrays.

    Raises:
      ValueError: `ordered_dict_of_arrays` contains the reserved name 'index'.
    """
    numpy_source = in_memory_source.OrderedDictNumpySource(
        ordered_dict_of_arrays,
        num_threads=num_threads,
        enqueue_size=enqueue_size,
        batch_size=batch_size,
        queue_capacity=queue_capacity,
        shuffle=shuffle,
        min_after_dequeue=min_after_dequeue,
        seed=seed,
        data_name=data_name)
    dataframe = cls()
    dataframe.assign(**(numpy_source()._asdict()))
    return dataframe
