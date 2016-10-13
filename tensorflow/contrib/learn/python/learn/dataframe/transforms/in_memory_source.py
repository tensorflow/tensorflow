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
"""Sources for numpy arrays and pandas DataFrames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_functions


class BaseInMemorySource(transform.TensorFlowTransform):
  """Abstract parent class for NumpySource and PandasSource."""

  def __init__(self,
               data,
               num_threads=None,
               enqueue_size=None,
               batch_size=None,
               queue_capacity=None,
               shuffle=False,
               min_after_dequeue=None,
               seed=None,
               data_name="in_memory_data"):
    super(BaseInMemorySource, self).__init__()
    self._data = data
    self._num_threads = 1 if num_threads is None else num_threads
    self._batch_size = (32 if batch_size is None else batch_size)
    self._enqueue_size = max(1, int(self._batch_size / self._num_threads)
                            ) if enqueue_size is None else enqueue_size
    self._queue_capacity = (self._batch_size * 10 if queue_capacity is None else
                            queue_capacity)
    self._shuffle = shuffle
    self._min_after_dequeue = (batch_size if min_after_dequeue is None else
                               min_after_dequeue)
    self._seed = seed
    self._data_name = data_name

  @transform.parameter
  def data(self):
    return self._data

  @transform.parameter
  def num_threads(self):
    return self._num_threads

  @transform.parameter
  def enqueue_size(self):
    return self._enqueue_size

  @transform.parameter
  def batch_size(self):
    return self._batch_size

  @transform.parameter
  def queue_capacity(self):
    return self._queue_capacity

  @transform.parameter
  def shuffle(self):
    return self._shuffle

  @transform.parameter
  def min_after_dequeue(self):
    return self._min_after_dequeue

  @transform.parameter
  def seed(self):
    return self._seed

  @transform.parameter
  def data_name(self):
    return self._data_name

  @property
  def input_valency(self):
    return 0

  def _apply_transform(self, transform_input, **kwargs):
    queue = feeding_functions.enqueue_data(self.data,
                                           self.queue_capacity,
                                           self.shuffle,
                                           self.min_after_dequeue,
                                           num_threads=self.num_threads,
                                           seed=self.seed,
                                           name=self.data_name,
                                           enqueue_size=self.enqueue_size,
                                           num_epochs=kwargs.get("num_epochs"))

    dequeued = queue.dequeue_many(self.batch_size)

    # TODO(jamieas): dequeue and dequeue_many will soon return a list regardless
    # of the number of enqueued tensors. Remove the following once that change
    # is in place.
    if not isinstance(dequeued, (tuple, list)):
      dequeued = (dequeued,)
    # pylint: disable=not-callable
    return self.return_type(*dequeued)


class NumpySource(BaseInMemorySource):
  """A zero-input Transform that produces a single column from a numpy array."""

  @property
  def name(self):
    return "NumpySource"

  @property
  def _output_names(self):
    return ("index", "value")


class OrderedDictNumpySource(BaseInMemorySource):
  """A zero-input Transform that produces Series from a dict of numpy arrays."""

  def __init__(self,
               ordered_dict_of_arrays,
               num_threads=None,
               enqueue_size=None,
               batch_size=None,
               queue_capacity=None,
               shuffle=False,
               min_after_dequeue=None,
               seed=None,
               data_name="pandas_data"):
    if "index" in ordered_dict_of_arrays.keys():
      raise ValueError("Column name `index` is reserved.")
    super(OrderedDictNumpySource, self).__init__(ordered_dict_of_arrays,
                                                 num_threads, enqueue_size,
                                                 batch_size, queue_capacity,
                                                 shuffle, min_after_dequeue,
                                                 seed, data_name)

  @property
  def name(self):
    return "OrderedDictNumpySource"

  @property
  def _output_names(self):
    return tuple(["index"] + list(self._data.keys()))


class PandasSource(BaseInMemorySource):
  """A zero-input Transform that produces Series from a DataFrame."""

  def __init__(self,
               dataframe,
               num_threads=None,
               enqueue_size=None,
               batch_size=None,
               queue_capacity=None,
               shuffle=False,
               min_after_dequeue=None,
               seed=None,
               data_name="pandas_data"):
    if "index" in dataframe.columns:
      raise ValueError("Column name `index` is reserved.")
    super(PandasSource, self).__init__(dataframe, num_threads, enqueue_size,
                                       batch_size, queue_capacity, shuffle,
                                       min_after_dequeue, seed, data_name)

  @property
  def name(self):
    return "PandasSource"

  @property
  def _output_names(self):
    return tuple(["index"] + self._data.columns.tolist())
