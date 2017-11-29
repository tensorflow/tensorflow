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

"""`ReaderSource` produces `Tensor`s of keys and values using a `tf.Reader`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.ops import io_ops
from tensorflow.python.training import input as input_ops


class ReaderSource(transform.TensorFlowTransform):
  """Produces `Tensor`s of keys and values using a `tf.Reader`."""

  def __init__(self,
               reader_cls,
               work_units,
               reader_kwargs=None,
               enqueue_size=None,
               batch_size=1,
               queue_capacity=None,
               shuffle=False,
               min_after_dequeue=None,
               num_threads=1,
               seed=None):
    """Initializes a ReaderSource.

    Args:
      reader_cls: A subclass of `tensorflow.ReaderBase` that will be used to
        read from `work_units`.
      work_units: A list that describes the source(s) of data to read.
        Typically, this is a list of filenames.
      reader_kwargs: A dictionary of kwargs to be passed to `reader_cls` when it
        is constructed.
      enqueue_size: block size for each read operation.
      batch_size: The desired batch size of output. Defaults to 1.
      queue_capacity: Capacity of the queue. Defaults to 10 * `batch_size`.
      shuffle: Whether records will be shuffled before returning. Defaults to
        false.
      min_after_dequeue: Minimum number of elements in the queue to allow a
        dequeue operation. Only used when `shuffle` is true. Defaults to
        `queue_capacity` / 4.
      num_threads: Number of threads that will be used for reading. Each thread
        has its own instance of `reader_cls`.
      seed: A seed used for shuffling. Only used if `shuffle` is true.
    """
    super(ReaderSource, self).__init__()
    self._reader_cls = reader_cls
    self._reader_kwargs = reader_kwargs
    self._work_units = work_units
    self._reader_kwargs = {} if reader_kwargs is None else reader_kwargs
    if enqueue_size is None:
      enqueue_size = max(1, int(batch_size / num_threads))
    self._enqueue_size = enqueue_size
    self._batch_size = batch_size
    self._queue_capacity = (batch_size * 10 if queue_capacity is None else
                            queue_capacity)
    self._shuffle = shuffle
    self._min_after_dequeue = int(self.queue_capacity / 4 if min_after_dequeue
                                  is None else min_after_dequeue)
    self._num_threads = num_threads
    self._seed = seed

  @transform._parameter  # pylint: disable=protected-access
  def reader_cls(self):
    return self._reader_cls

  @transform._parameter  # pylint: disable=protected-access
  def work_units(self):
    return self._work_units

  @transform._parameter  # pylint: disable=protected-access
  def reader_kwargs(self):
    return self._reader_kwargs

  @transform._parameter  # pylint: disable=protected-access
  def enqueue_size(self):
    return self._enqueue_size

  @transform._parameter  # pylint: disable=protected-access
  def batch_size(self):
    return self._batch_size

  @transform._parameter  # pylint: disable=protected-access
  def queue_capacity(self):
    return self._queue_capacity

  @transform._parameter  # pylint: disable=protected-access
  def shuffle(self):
    return self._shuffle

  @transform._parameter  # pylint: disable=protected-access
  def min_after_dequeue(self):
    return self._min_after_dequeue

  @transform._parameter  # pylint: disable=protected-access
  def num_threads(self):
    return self._num_threads

  @transform._parameter  # pylint: disable=protected-access
  def seed(self):
    return self._seed

  @property
  def name(self):
    return "ReaderSource"

  @property
  def input_valency(self):
    return 0

  @property
  def _output_names(self):
    return ("index", "value")

  def _apply_transform(self, transform_input, **kwargs):
    filename_queue = input_ops.string_input_producer(
        self.work_units,
        num_epochs=kwargs.get("num_epochs"),
        shuffle=self.shuffle,
        seed=self.seed)
    reader_ops = []
    for _ in range(self.num_threads):
      reader = self._reader_cls(**self._reader_kwargs)
      reader_ops.append(reader.read_up_to(filename_queue, self.enqueue_size))

    if self.shuffle:
      dequeued = input_ops.shuffle_batch_join(
          reader_ops,
          self.batch_size,
          capacity=self.queue_capacity,
          min_after_dequeue=self.min_after_dequeue,
          seed=self.seed,
          enqueue_many=True,
          shared_name=None,
          name=None)
    else:
      dequeued = input_ops.batch_join(reader_ops,
                                      self.batch_size,
                                      capacity=self.queue_capacity,
                                      enqueue_many=True,
                                      dynamic_pad=False,
                                      shared_name=None,
                                      name=None)

    # pylint: disable=not-callable
    return self.return_type(*dequeued)


# `ReaderSource`s for common `tf.ReaderBase` types.
def TextFileSource(file_names,
                   reader_kwargs=None,
                   enqueue_size=1,
                   batch_size=1,
                   queue_capacity=None,
                   shuffle=False,
                   min_after_dequeue=None,
                   num_threads=1,
                   seed=None):
  return ReaderSource(io_ops.TextLineReader,
                      work_units=file_names,
                      reader_kwargs=reader_kwargs,
                      enqueue_size=enqueue_size,
                      batch_size=batch_size,
                      queue_capacity=queue_capacity,
                      shuffle=shuffle,
                      min_after_dequeue=min_after_dequeue,
                      num_threads=num_threads,
                      seed=seed)


def TFRecordSource(file_names,
                   reader_kwargs=None,
                   enqueue_size=1,
                   batch_size=1,
                   queue_capacity=None,
                   shuffle=False,
                   min_after_dequeue=None,
                   num_threads=1,
                   seed=None):
  return ReaderSource(io_ops.TFRecordReader,
                      work_units=file_names,
                      reader_kwargs=reader_kwargs,
                      enqueue_size=enqueue_size,
                      batch_size=batch_size,
                      queue_capacity=queue_capacity,
                      shuffle=shuffle,
                      min_after_dequeue=min_after_dequeue,
                      num_threads=num_threads,
                      seed=seed)
