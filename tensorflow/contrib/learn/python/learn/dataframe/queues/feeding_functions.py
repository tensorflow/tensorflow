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

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_queue_runner as fqr
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import queue_runner

# pylint: disable=g-import-not-at-top
try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False


class _ArrayFeedFn(object):
  """Creates feed dictionaries from numpy arrays."""

  def __init__(self, placeholders, array):
    if len(placeholders) != 2:
      raise ValueError("_array_feed_fn expects 2 placeholders; got {}.".format(
          len(placeholders)))
    self._placeholders = placeholders
    self._array = array
    self._reset()

  def _reset(self):
    self._row_iterator = enumerate(self._array)

  def __call__(self):
    try:
      index, row = next(self._row_iterator)
    except StopIteration:
      self._reset()
      index, row = next(self._row_iterator)
    return {self._placeholders[0]: index, self._placeholders[1]: row}


class _PandasFeedFn(object):
  """Creates feed dictionaries from pandas `DataFrames`."""

  def __init__(self, placeholders, dataframe):
    if len(placeholders) != len(dataframe.columns) + 1:
      raise ValueError("Expected {} placeholders; got {}.".format(
          len(dataframe.columns), len(placeholders)))
    self._index_placeholder = placeholders[0]
    self._row_placeholders = placeholders[1:]
    self._dataframe = dataframe
    self._reset()

  def _reset(self):
    self._row_iterator = self._dataframe.iterrows()

  def __call__(self):
    try:
      index, row = next(self._row_iterator)
    except StopIteration:
      self._reset()
      index, row = next(self._row_iterator)
    feed_dict = dict(zip(self._row_placeholders, row))
    feed_dict[self._index_placeholder] = index
    return feed_dict


def enqueue_data(data,
                 capacity,
                 shuffle=False,
                 min_after_dequeue=None,
                 seed=None):
  """Creates a queue filled from a numpy array or pandas `DataFrame`.

    Returns a queue filled with the rows of the given array or `DataFrame`. In
    the case of a pandas `DataFrame`, the first enqueued `Tensor` corresponds to
    the index of the `DataFrame`. For numpy arrays, the first enqueued `Tensor`
    contains the row number.

  Args:
    data: a numpy `ndarray or` pandas `DataFrame` that will be read into the
      queue.
    capacity: the capacity of the queue.
    shuffle: whether or not to shuffle the rows of the array.
    min_after_dequeue: minimum number of elements that can remain in the queue
    after a dequeue operation. Only used when `shuffle` is true. If not set,
    defaults to `capacity` / 4.
    seed: used to seed RandomShuffleQueue. Only used when `shuffle` is True.

  Returns:
    A queue filled with the rows of the given array or `DataFrame`.

  Raises:
    TypeError: `data` is not a Pandas `DataFrame` or a numpy `ndarray`.
  """
  # TODO(jamieas): create multithreaded version of enqueue_data.
  if isinstance(data, np.ndarray):
    types = [dtypes.int64, dtypes.as_dtype(data.dtype)]
    shapes = [(), data.shape[1:]]
    get_feed_fn = _ArrayFeedFn
  elif HAS_PANDAS and isinstance(data, pd.DataFrame):
    types = [dtypes.as_dtype(dt)
             for dt in [data.index.dtype] + list(data.dtypes)]
    shapes = [() for _ in types]
    get_feed_fn = _PandasFeedFn
  else:
    raise TypeError(
        "data must be either a numpy array or pandas DataFrame if pandas is "
        "installed; got {}".format(
            type(data).__name__))

  placeholders = [array_ops.placeholder(*type_and_shape)
                  for type_and_shape in zip(types, shapes)]
  if shuffle:
    min_after_dequeue = (capacity / 4 if min_after_dequeue is None else
                         min_after_dequeue)
    queue = data_flow_ops.RandomShuffleQueue(capacity,
                                             min_after_dequeue,
                                             dtypes=types,
                                             shapes=shapes,
                                             seed=seed)
  else:
    queue = data_flow_ops.FIFOQueue(capacity, dtypes=types, shapes=shapes)
  enqueue_op = queue.enqueue(placeholders)
  feed_fn = get_feed_fn(placeholders, data)
  runner = fqr.FeedingQueueRunner(queue=queue,
                                  enqueue_ops=[enqueue_op],
                                  feed_fn=feed_fn)
  queue_runner.add_queue_runner(runner)
  return queue
