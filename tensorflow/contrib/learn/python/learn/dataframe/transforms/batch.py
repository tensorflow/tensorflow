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

"""Batches `Series` objects. For internal use, not part of the public API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.training import input as input_ops


class AbstractBatchTransform(transform.TensorFlowTransform):
  """Abstract parent class for batching Transforms."""

  def __init__(self,
               batch_size,
               output_names,
               num_threads=1,
               queue_capacity=None):
    super(AbstractBatchTransform, self).__init__()
    self._batch_size = batch_size
    self._output_name_list = output_names
    self._num_threads = num_threads
    self._queue_capacity = (self.batch_size * 10 if queue_capacity is None
                            else queue_capacity)

  @transform.parameter
  def batch_size(self):
    return self._batch_size

  @transform.parameter
  def num_threads(self):
    return self._num_threads

  @transform.parameter
  def queue_capacity(self):
    return self._queue_capacity

  @property
  def input_valency(self):
    return len(self.output_names)

  @property
  def _output_names(self):
    return self._output_name_list


class Batch(AbstractBatchTransform):
  """Batches Columns to specified size.

  Note that dimension 0 is assumed to correspond to "example number" so
  `Batch` does not prepend an additional dimension to incoming `Series`.
  For example, if a `Tensor` in `transform_input` has shape [x, y], the
  corresponding output will have shape [batch_size, y].
  """

  @property
  def name(self):
    return "Batch"

  def _apply_transform(self, transform_input, **kwargs):
    batched = input_ops.batch(transform_input,
                              batch_size=self.batch_size,
                              num_threads=self.num_threads,
                              capacity=self.queue_capacity,
                              enqueue_many=True)
    # TODO(jamieas): batch will soon return a list regardless of the number of
    # enqueued tensors. Remove the following once that change is in place.
    if not isinstance(batched, (tuple, list)):
      batched = (batched,)
    # pylint: disable=not-callable
    return self.return_type(*batched)


class ShuffleBatch(AbstractBatchTransform):
  """Creates shuffled batches from `Series` containing a single row.

  Note that dimension 0 is assumed to correspond to "example number" so
  `ShuffleBatch` does not prepend an additional dimension to incoming `Series`.
  For example, if a `Tensor` in `transform_input` has shape [x, y], the
  corresponding output will have shape [batch_size, y].
  """

  @property
  def name(self):
    return "ShuffleBatch"

  def __init__(self,
               batch_size,
               output_names,
               num_threads=1,
               queue_capacity=None,
               min_after_dequeue=None,
               seed=None):
    super(ShuffleBatch, self).__init__(batch_size, output_names, num_threads,
                                       queue_capacity)
    self._min_after_dequeue = int(self.queue_capacity / 4
                                  if min_after_dequeue is None
                                  else min_after_dequeue)
    self._seed = seed

  @transform.parameter
  def min_after_dequeue(self):
    return self._min_after_dequeue

  @transform.parameter
  def seed(self):
    return self._seed

  def _apply_transform(self, transform_input, **kwargs):
    batched = input_ops.shuffle_batch(transform_input,
                                      batch_size=self.batch_size,
                                      capacity=self.queue_capacity,
                                      min_after_dequeue=self.min_after_dequeue,
                                      num_threads=self.num_threads,
                                      seed=self.seed,
                                      enqueue_many=True)
    # TODO(jamieas): batch will soon return a list regardless of the number of
    # enqueued tensors. Remove the following once that change is in place.
    if not isinstance(batched, (tuple, list)):
      batched = (batched,)
    # pylint: disable=not-callable
    return self.return_type(*batched)
