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
"""Experimental API for gathering statistics from `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


class StatsAggregator(object):
  """A stateful resource that aggregates statistics from one or more iterators.

  To record statistics, use one of the custom transformation functions defined
  in this module when defining your @{tf.data.Dataset}. All statistics will be
  aggregated by the `StatsAggregator` that is associated with a particular
  iterator (see below). For example, to record the total number of bytes
  produced by iterating over a dataset:

  ```python
  dataset = ...
  dataset = dataset.apply(stats_ops.bytes_produced_stats("total_bytes"))
  ```

  To associate a `StatsAggregator` with a @{tf.data.Iterator} object, use
  the following pattern:

  ```python
  dataset = ...
  iterator = dataset.make_one_shot_iterator()
  stats_aggregator = stats_ops.StatsAggregator()
  set_op = stats_op.set_stats_aggregator_op(iterator, stats_aggregator)

  with tf.Session() as sess:
    # Running `set_op` will associate `iterator` with `stats_aggregator`.
    sess.run(set_op)
  ```

  To get a protocol buffer summary of the currently aggregated statistics,
  use the `StatsAggregator.get_summary()` tensor. The easiest way to do this
  is to add the returned tensor to the @{tf.GraphKeys.SUMMARIES} collection,
  so that the summaries will be included with any existing summaries.

  ```python
  stats_aggregator = stats_ops.StatsAggregator()
  stats_summary = stats_aggregator.get_summary()
  tf.add_to_collection(tf.GraphKeys.SUMMARIES, stats_summary)
  ```

  Note: This interface is experimental and expected to change. In particular,
  we expect to add other implementations of `StatsAggregator` that provide
  different ways of exporting statistics, and add more types of statistics.
  """

  def __init__(self):
    """Creates a `StatsAggregator`."""
    self._resource = gen_dataset_ops.stats_aggregator_handle()

  def get_summary(self):
    """Returns a string @{tf.Tensor} that summarizes the aggregated statistics.

    The returned tensor will contain a serialized @{tf.summary.Summary} protocol
    buffer, which can be used with the standard TensorBoard logging facilities.

    Returns:
      A scalar string @{tf.Tensor} that summarizes the aggregated statistics.
    """
    return gen_dataset_ops.stats_aggregator_summary(self._resource)

  def subscribe(self, iterator):
    """Returns a @{tf.Operation} to associate this aggregator with `iterator`.

    Note: Each @{tf.data.Iterator} can be associated with at most one
    `StatsAggregator`. After running the operation that this function
    returns, all statistics recorded in the iteration of `iterator`
    will be stored in `stats_aggregator`.

    Args:
      iterator: A @{tf.data.Iterator} object.

    Returns:
      A @{tf.Operation} that, when run, associates this aggregator with
      `iterator`.
    """
    if not isinstance(iterator, iterator_ops.Iterator):
      raise TypeError("`iterator` must be a `tf.data.Iterator` object.")
    return gen_dataset_ops.iterator_set_stats_aggregator(
        iterator._iterator_resource, self._resource)  # pylint: disable=protected-access


def bytes_produced_stats(tag):
  """Records the number of bytes produced by each element of the input dataset.

  To consume the statistics, associate a `StatsAggregator` with an iterator
  over the output dataset.

  Args:
    tag: String. All statistics recorded by the returned transformation will
      be associated with the given `tag`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    return _StatsDataset(dataset, gen_dataset_ops.bytes_produced_stats_dataset,
                         tag)

  return _apply_fn


def latency_stats(tag):
  """Records the latency of producing each element of the input dataset.

  To consume the statistics, associate a `StatsAggregator` with an iterator
  over the output dataset.

  Args:
    tag: String. All statistics recorded by the returned transformation will
      be associated with the given `tag`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    return _StatsDataset(dataset, gen_dataset_ops.latency_stats_dataset, tag)

  return _apply_fn


class _StatsDataset(dataset_ops.Dataset):
  """A `Dataset` that acts as an identity, and also records statistics."""

  def __init__(self, input_dataset, op_function, tag):
    super(_StatsDataset, self).__init__()
    self._input_dataset = input_dataset
    self._op_function = op_function
    self._tag = ops.convert_to_tensor(tag, dtype=dtypes.string)

  def _as_variant_tensor(self):
    return self._op_function(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._tag,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_classes(self):
    return self._input_dataset.output_classes
