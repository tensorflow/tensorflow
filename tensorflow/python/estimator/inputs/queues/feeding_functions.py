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
"""Helper functions for enqueuing data from arrays and pandas `DataFrame`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np

from tensorflow.python.estimator.inputs.queues import feeding_queue_runner as fqr
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner

try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
except ImportError:
  HAS_PANDAS = False


def _get_integer_indices_for_next_batch(
    batch_indices_start, batch_size, epoch_end, array_length,
    current_epoch, total_epochs):
  """Returns the integer indices for next batch.

  If total epochs is not None and current epoch is the final epoch, the end
  index of the next batch should not exceed the `epoch_end` (i.e., the final
  batch might not have size `batch_size` to avoid overshooting the last epoch).

  Args:
    batch_indices_start: Integer, the index to start next batch.
    batch_size: Integer, size of batches to return.
    epoch_end: Integer, the end index of the epoch. The epoch could start from a
      random position, so `epoch_end` provides the end index for that.
    array_length: Integer, the length of the array.
    current_epoch: Integer, the epoch number has been emitted.
    total_epochs: Integer or `None`, the total number of epochs to emit. If
      `None` will run forever.

  Returns:
    A tuple of a list with integer indices for next batch and `current_epoch`
    value after the next batch.

  Raises:
    OutOfRangeError if `current_epoch` is not less than `total_epochs`.

  """
  if total_epochs is not None and current_epoch >= total_epochs:
    raise errors.OutOfRangeError(None, None,
                                 "Already emitted %s epochs." % current_epoch)

  batch_indices_end = batch_indices_start + batch_size
  batch_indices = [j % array_length for j in
                   range(batch_indices_start, batch_indices_end)]
  epoch_end_indices = [i for i, x in enumerate(batch_indices) if x == epoch_end]
  current_epoch += len(epoch_end_indices)

  if total_epochs is None or current_epoch < total_epochs:
    return (batch_indices, current_epoch)

  # Now we might have emitted more data for expected epochs. Need to trim.
  final_epoch_end_inclusive = epoch_end_indices[
      -(current_epoch - total_epochs + 1)]
  batch_indices = batch_indices[:final_epoch_end_inclusive + 1]

  return (batch_indices, total_epochs)


class _ArrayFeedFn(object):
  """Creates feed dictionaries from numpy arrays."""

  def __init__(self,
               placeholders,
               array,
               batch_size,
               random_start=False,
               seed=None,
               num_epochs=None):
    if len(placeholders) != 2:
      raise ValueError("_array_feed_fn expects 2 placeholders; got {}.".format(
          len(placeholders)))
    self._placeholders = placeholders
    self._array = array
    self._max = len(array)
    self._batch_size = batch_size
    self._num_epochs = num_epochs
    self._epoch = 0
    random.seed(seed)
    self._trav = random.randrange(self._max) if random_start else 0
    self._epoch_end = (self._trav - 1) % self._max

  def __call__(self):
    integer_indexes, self._epoch = _get_integer_indices_for_next_batch(
        batch_indices_start=self._trav,
        batch_size=self._batch_size,
        epoch_end=self._epoch_end,
        array_length=self._max,
        current_epoch=self._epoch,
        total_epochs=self._num_epochs)

    self._trav = (integer_indexes[-1] + 1) % self._max
    return {
        self._placeholders[0]: integer_indexes,
        self._placeholders[1]: self._array[integer_indexes]
    }


class _OrderedDictNumpyFeedFn(object):
  """Creates feed dictionaries from `OrderedDict`s of numpy arrays."""

  def __init__(self,
               placeholders,
               ordered_dict_of_arrays,
               batch_size,
               random_start=False,
               seed=None,
               num_epochs=None):
    if len(placeholders) != len(ordered_dict_of_arrays) + 1:
      raise ValueError("Expected {} placeholders; got {}.".format(
          len(ordered_dict_of_arrays), len(placeholders)))
    self._index_placeholder = placeholders[0]
    self._col_placeholders = placeholders[1:]
    self._ordered_dict_of_arrays = ordered_dict_of_arrays
    self._max = len(next(iter(ordered_dict_of_arrays.values())))
    for _, v in ordered_dict_of_arrays.items():
      if len(v) != self._max:
        raise ValueError("Array lengths must match.")
    self._batch_size = batch_size
    self._num_epochs = num_epochs
    self._epoch = 0
    random.seed(seed)
    self._trav = random.randrange(self._max) if random_start else 0
    self._epoch_end = (self._trav - 1) % self._max

  def __call__(self):
    integer_indexes, self._epoch = _get_integer_indices_for_next_batch(
        batch_indices_start=self._trav,
        batch_size=self._batch_size,
        epoch_end=self._epoch_end,
        array_length=self._max,
        current_epoch=self._epoch,
        total_epochs=self._num_epochs)

    self._trav = (integer_indexes[-1] + 1) % self._max
    feed_dict = {self._index_placeholder: integer_indexes}
    cols = [
        column[integer_indexes]
        for column in self._ordered_dict_of_arrays.values()
    ]
    feed_dict.update(dict(zip(self._col_placeholders, cols)))
    return feed_dict


class _PandasFeedFn(object):
  """Creates feed dictionaries from pandas `DataFrames`."""

  def __init__(self,
               placeholders,
               dataframe,
               batch_size,
               random_start=False,
               seed=None,
               num_epochs=None):
    if len(placeholders) != len(dataframe.columns) + 1:
      raise ValueError("Expected {} placeholders; got {}.".format(
          len(dataframe.columns), len(placeholders)))
    self._index_placeholder = placeholders[0]
    self._col_placeholders = placeholders[1:]
    self._dataframe = dataframe
    self._max = len(dataframe)
    self._batch_size = batch_size
    self._num_epochs = num_epochs
    self._epoch = 0
    random.seed(seed)
    self._trav = random.randrange(self._max) if random_start else 0
    self._epoch_end = (self._trav - 1) % self._max

  def __call__(self):
    integer_indexes, self._epoch = _get_integer_indices_for_next_batch(
        batch_indices_start=self._trav,
        batch_size=self._batch_size,
        epoch_end=self._epoch_end,
        array_length=self._max,
        current_epoch=self._epoch,
        total_epochs=self._num_epochs)

    self._trav = (integer_indexes[-1] + 1) % self._max
    result = self._dataframe.iloc[integer_indexes]
    cols = [result[col].values for col in result.columns]
    feed_dict = dict(zip(self._col_placeholders, cols))
    feed_dict[self._index_placeholder] = result.index.values
    return feed_dict


def _enqueue_data(data,
                  capacity,
                  shuffle=False,
                  min_after_dequeue=None,
                  num_threads=1,
                  seed=None,
                  name="enqueue_input",
                  enqueue_size=1,
                  num_epochs=None):
  """Creates a queue filled from a numpy array or pandas `DataFrame`.

    Returns a queue filled with the rows of the given (`OrderedDict` of) array
    or `DataFrame`. In the case of a pandas `DataFrame`, the first enqueued
    `Tensor` corresponds to the index of the `DataFrame`. For (`OrderedDict` of)
    numpy arrays, the first enqueued `Tensor` contains the row number.

  Args:
    data: a numpy `ndarray`, `OrderedDict` of numpy arrays, or pandas
      `DataFrame` that will be read into the queue.
    capacity: the capacity of the queue.
    shuffle: whether or not to shuffle the rows of the array.
    min_after_dequeue: minimum number of elements that can remain in the queue
    after a dequeue operation. Only used when `shuffle` is true. If not set,
    defaults to `capacity` / 4.
    num_threads: number of threads used for reading and enqueueing.
    seed: used to seed shuffling and reader starting points.
    name: a scope name identifying the data.
    enqueue_size: the number of rows to enqueue per step.
    num_epochs: limit enqueuing to a specified number of epochs, if provided.

  Returns:
    A queue filled with the rows of the given (`OrderedDict` of) array or
      `DataFrame`.

  Raises:
    TypeError: `data` is not a Pandas `DataFrame`, an `OrderedDict` of numpy
      arrays  or a numpy `ndarray`.
  """
  with ops.name_scope(name):
    if isinstance(data, np.ndarray):
      types = [dtypes.int64, dtypes.as_dtype(data.dtype)]
      queue_shapes = [(), data.shape[1:]]
      get_feed_fn = _ArrayFeedFn
    elif isinstance(data, collections.OrderedDict):
      types = [dtypes.int64] + [
          dtypes.as_dtype(col.dtype) for col in data.values()
      ]
      queue_shapes = [()] + [col.shape[1:] for col in data.values()]
      get_feed_fn = _OrderedDictNumpyFeedFn
    elif HAS_PANDAS and isinstance(data, pd.DataFrame):
      types = [
          dtypes.as_dtype(dt) for dt in [data.index.dtype] + list(data.dtypes)
      ]
      queue_shapes = [() for _ in types]
      get_feed_fn = _PandasFeedFn
    else:
      raise TypeError(
          "data must be either a numpy array or pandas DataFrame if pandas is "
          "installed; got {}".format(type(data).__name__))

    # TODO(jamieas): TensorBoard warnings for all warnings below once available.

    if num_threads > 1 and num_epochs is not None:
      logging.warning(
          "enqueue_data was called with num_epochs and num_threads > 1. "
          "num_epochs is applied per thread, so this will produce more "
          "epochs than you probably intend. "
          "If you want to limit epochs, use one thread.")

    if shuffle and num_threads > 1 and num_epochs is not None:
      logging.warning(
          "enqueue_data was called with shuffle=True, num_threads > 1, and "
          "num_epochs. This will create multiple threads, all reading the "
          "array/dataframe in order adding to the same shuffling queue; the "
          "results will likely not be sufficiently shuffled.")

    if not shuffle and num_threads > 1:
      logging.warning(
          "enqueue_data was called with shuffle=False and num_threads > 1. "
          "This will create multiple threads, all reading the "
          "array/dataframe in order. If you want examples read in order, use"
          " one thread; if you want multiple threads, enable shuffling.")

    if shuffle:
      min_after_dequeue = int(capacity / 4 if min_after_dequeue is None else
                              min_after_dequeue)
      queue = data_flow_ops.RandomShuffleQueue(
          capacity,
          min_after_dequeue,
          dtypes=types,
          shapes=queue_shapes,
          seed=seed)
    else:
      min_after_dequeue = 0  # just for the summary text
      queue = data_flow_ops.FIFOQueue(
          capacity, dtypes=types, shapes=queue_shapes)

    enqueue_ops = []
    feed_fns = []

    for i in range(num_threads):
      # Note the placeholders have no shapes, so they will accept any
      # enqueue_size.  enqueue_many below will break them up.
      placeholders = [array_ops.placeholder(t) for t in types]

      enqueue_ops.append(queue.enqueue_many(placeholders))
      seed_i = None if seed is None else (i + 1) * seed
      feed_fns.append(
          get_feed_fn(
              placeholders,
              data,
              enqueue_size,
              random_start=shuffle,
              seed=seed_i,
              num_epochs=num_epochs))

    runner = fqr._FeedingQueueRunner(  # pylint: disable=protected-access
        queue=queue, enqueue_ops=enqueue_ops, feed_fns=feed_fns)
    queue_runner.add_queue_runner(runner)

    full = (math_ops.cast(
        math_ops.maximum(0, queue.size() - min_after_dequeue),
        dtypes.float32) * (1. / (capacity - min_after_dequeue)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = ("queue/%sfraction_over_%d_of_%d_full" %
                    (queue.name, min_after_dequeue,
                     capacity - min_after_dequeue))
    summary.scalar(summary_name, full)
    return queue
