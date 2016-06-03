# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


class BaseInMemorySource(transform.Transform):
  """Abstract parent class for NumpySource and PandasSource."""

  def __init__(self,
               data,
               batch_size=None,
               queue_capacity=None,
               shuffle=False,
               min_after_dequeue=None,
               seed=None):
    super(BaseInMemorySource, self).__init__()
    self._data = data
    self._batch_size = (1 if batch_size is None else batch_size)
    self._queue_capacity = (self._batch_size * 10 if batch_size is None
                            else batch_size)
    self._shuffle = shuffle
    self._min_after_dequeue = (batch_size if min_after_dequeue is None
                               else min_after_dequeue)
    self._seed = seed

  @transform.parameter
  def data(self):
    return self._data

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

  @property
  def input_valency(self):
    return 0

  def _apply_transform(self, transform_input):
    queue = feeding_functions.enqueue_data(
        self.data, self.queue_capacity, self.shuffle, self.min_after_dequeue)

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


class PandasSource(BaseInMemorySource):
  """A zero-input Transform that produces Series from a DataFrame."""

  def __init__(self,
               dataframe,
               batch_size=None,
               queue_capacity=None,
               shuffle=False,
               min_after_dequeue=None,
               seed=None):
    if "index" in dataframe.columns:
      raise ValueError("Column name `index` is reserved.")
    super(PandasSource, self).__init__(dataframe, batch_size, queue_capacity,
                                       shuffle, min_after_dequeue, seed)

  @property
  def name(self):
    return "PandasSource"

  @property
  def _output_names(self):
    return tuple(["index"] + self._data.columns.tolist())
