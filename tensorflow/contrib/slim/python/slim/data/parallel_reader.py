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
"""Implements a parallel data reader with queues and optional shuffling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes as tf_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary
from tensorflow.python.training import input as tf_input
from tensorflow.python.training import queue_runner


class ParallelReader(io_ops.ReaderBase):
  """Reader class that uses multiple readers in parallel to improve speed.

  See ReaderBase for supported methods.
  """

  def __init__(self,
               reader_class,
               common_queue,
               num_readers=4,
               reader_kwargs=None):
    """ParallelReader creates num_readers instances of the reader_class.

    Each instance is created by calling the `reader_class` function passing
    the arguments specified in `reader_kwargs` as in:
      reader_class(**read_kwargs)

    When you read from a ParallelReader, with its `read()` method,
    you just dequeue examples from the `common_queue`.

    The readers will read different files in parallel, asynchronously enqueueing
    their output into `common_queue`. The `common_queue.dtypes` must be
    [tf.string, tf.string]

    Because each reader can read from a different file, the examples in the
    `common_queue` could be from different files. Due to the asynchronous
    reading there is no guarantee that all the readers will read the same
    number of examples.

    If the `common_queue` is a shuffling queue, then the examples are shuffled.

    Usage:
      common_queue = tf.RandomShuffleQueue(
          capacity=256,
          min_after_dequeue=128,
          dtypes=[tf.string, tf.string])
      p_reader = ParallelReader(tf.TFRecordReader, common_queue)

      common_queue = tf.FIFOQueue(
          capacity=256,
          dtypes=[tf.string, tf.string])
      p_reader = ParallelReader(readers, common_queue, num_readers=2)


    Args:
      reader_class: one of the io_ops.ReaderBase subclasses ex: TFRecordReader
      common_queue: a Queue to hold (key, value pairs) with `dtypes` equal to
        [tf.string, tf.string]. Must be one of the data_flow_ops.Queues
        instances, ex. `tf.FIFOQueue()`, `tf.RandomShuffleQueue()`, ...
      num_readers: a integer, number of instances of reader_class to create.
      reader_kwargs: an optional dict of kwargs to create the readers.

    Raises:
      TypeError: if `common_queue.dtypes` is not [tf.string, tf.string].
    """
    if len(common_queue.dtypes) != 2:
      raise TypeError('common_queue.dtypes must be [tf.string, tf.string]')
    for dtype in common_queue.dtypes:
      if not dtype.is_compatible_with(tf_dtypes.string):
        raise TypeError('common_queue.dtypes must be [tf.string, tf.string]')

    reader_kwargs = reader_kwargs or {}
    self._readers = [reader_class(**reader_kwargs) for _ in range(num_readers)]
    self._common_queue = common_queue

  @property
  def num_readers(self):
    return len(self._readers)

  @property
  def common_queue(self):
    return self._common_queue

  def read(self, queue, name=None):
    """Returns the next record (key, value pair) produced by the reader.

    The multiple reader instances are all configured to `read()` from the
    filenames listed in `queue` and enqueue their output into the `common_queue`
    passed to the constructor, and this method returns the next record dequeued
    from that `common_queue`.


    Readers dequeue a work unit from `queue` if necessary (e.g. when a
    reader needs to start reading from a new file since it has finished with
    the previous file).

    A queue runner for enqueing in the `common_queue` is automatically added to
    the TF QueueRunners collection.

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      name: A name for the operation (optional).

    Returns:
      The next record (i.e. (key, value pair)) from the common_queue.
    """

    enqueue_ops = []
    for reader in self._readers:
      enqueue_ops.append(self._common_queue.enqueue(reader.read(queue)))

    queue_runner.add_queue_runner(
        queue_runner.QueueRunner(self._common_queue, enqueue_ops))

    return self._common_queue.dequeue(name=name)

  def num_records_produced(self, name=None):
    """Returns the number of records this reader has produced.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.

    """
    num_records = [r.num_records_produced() for r in self._readers]
    return math_ops.add_n(num_records, name=name)

  def num_work_units_completed(self, name=None):
    """Returns the number of work units this reader has finished processing.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.
    """
    num_work_units = [r.num_work_units_completed() for r in self._readers]
    return math_ops.add_n(num_work_units, name=name)


def parallel_read(data_sources,
                  reader_class,
                  num_epochs=None,
                  num_readers=4,
                  reader_kwargs=None,
                  shuffle=True,
                  dtypes=None,
                  capacity=256,
                  min_after_dequeue=128,
                  seed=None,
                  scope=None):
  """Reads multiple records in parallel from data_sources using n readers.

  It uses a ParallelReader to read from multiple files in  parallel using
  multiple readers created using `reader_class` with `reader_kwargs'.

  If shuffle is True the common_queue would be a RandomShuffleQueue otherwise
  it would be a FIFOQueue.

  Usage:
      data_sources = ['path_to/train*']
      key, value = parallel_read(data_sources, tf.CSVReader, num_readers=4)

  Args:
    data_sources: a list/tuple of files or the location of the data, i.e.
      /path/to/train@128, /path/to/train* or /tmp/.../train*
    reader_class: one of the io_ops.ReaderBase subclasses ex: TFRecordReader
    num_epochs: The number of times each data source is read. If left as None,
        the data will be cycled through indefinitely.
    num_readers: a integer, number of Readers to create.
    reader_kwargs: an optional dict, of kwargs for the reader.
    shuffle: boolean, wether should shuffle the files and the records by using
      RandomShuffleQueue as common_queue.
    dtypes:  A list of types.  The length of dtypes must equal the number
        of elements in each record. If it is None it will default to
        [tf.string, tf.string] for (key, value).
    capacity: integer, capacity of the common_queue.
    min_after_dequeue: integer, minimum number of records in the common_queue
      after dequeue. Needed for a good shuffle.
    seed: A seed for RandomShuffleQueue.
    scope: Optional name scope for the ops.

  Returns:
    key, value: a tuple of keys and values from the data_source.
  """
  data_files = get_data_files(data_sources)
  with ops.name_scope(scope, 'parallel_read'):
    filename_queue = tf_input.string_input_producer(
        data_files, num_epochs=num_epochs, shuffle=shuffle, name='filenames')
    dtypes = dtypes or [tf_dtypes.string, tf_dtypes.string]
    if shuffle:
      common_queue = data_flow_ops.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=dtypes,
          seed=seed,
          name='common_queue')
    else:
      common_queue = data_flow_ops.FIFOQueue(
          capacity=capacity, dtypes=dtypes, name='common_queue')

    summary.scalar('fraction_of_%d_full' % capacity,
                   math_ops.to_float(common_queue.size()) * (1. / capacity))

    return ParallelReader(
        reader_class,
        common_queue,
        num_readers=num_readers,
        reader_kwargs=reader_kwargs).read(filename_queue)


def single_pass_read(data_sources, reader_class, reader_kwargs=None,
                     scope=None):
  """Reads sequentially the data_sources using the reader, doing a single pass.

  Args:
    data_sources: a list/tuple of files or the location of the data, i.e.
      /path/to/train@128, /path/to/train* or /tmp/.../train*
    reader_class: one of the io_ops.ReaderBase subclasses ex: TFRecordReader.
    reader_kwargs: an optional dict, of kwargs for the reader.
    scope: Optional name scope for the ops.

  Returns:
    key, value: a tuple of keys and values from the data_source.
  """
  data_files = get_data_files(data_sources)
  with ops.name_scope(scope, 'single_pass_read'):
    filename_queue = tf_input.string_input_producer(
        data_files, num_epochs=1, shuffle=False, capacity=1, name='filenames')
    reader_kwargs = reader_kwargs or {}
    return reader_class(**reader_kwargs).read(filename_queue)


def get_data_files(data_sources):
  """Get data_files from data_sources.

  Args:
    data_sources: a list/tuple of files or the location of the data, i.e.
      /path/to/train@128, /path/to/train* or /tmp/.../train*

  Returns:
    a list of data_files.

  Raises:
    ValueError: if not data files are not found

  """
  if isinstance(data_sources, (list, tuple)):
    data_files = []
    for source in data_sources:
      data_files += get_data_files(source)
  else:
    if '*' in data_sources or '?' in data_sources or '[' in data_sources:
      data_files = gfile.Glob(data_sources)
    else:
      data_files = [data_sources]
  if not data_files:
    raise ValueError('No data files found in %s' % (data_sources,))
  return data_files
