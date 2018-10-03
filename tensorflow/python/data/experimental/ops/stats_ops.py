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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.StatsAggregator")
class StatsAggregator(object):
  """A stateful resource that aggregates statistics from one or more iterators.

  To record statistics, use one of the custom transformation functions defined
  in this module when defining your `tf.data.Dataset`. All statistics will be
  aggregated by the `StatsAggregator` that is associated with a particular
  iterator (see below). For example, to record the latency of producing each
  element by iterating over a dataset:

  ```python
  dataset = ...
  dataset = dataset.apply(tf.data.experimental.latency_stats("total_bytes"))
  ```

  To associate a `StatsAggregator` with a `tf.data.Dataset` object, use
  the following pattern:

  ```python
  stats_aggregator = stats_ops.StatsAggregator()
  dataset = ...

  # Apply `set_stats_aggregator` to associate `dataset` with `stats_aggregator`.
  dataset = dataset.apply(
      tf.data.experimental.set_stats_aggregator(stats_aggregator))
  iterator = dataset.make_one_shot_iterator()
  ```

  To get a protocol buffer summary of the currently aggregated statistics,
  use the `StatsAggregator.get_summary()` tensor. The easiest way to do this
  is to add the returned tensor to the `tf.GraphKeys.SUMMARIES` collection,
  so that the summaries will be included with any existing summaries.

  ```python
  stats_aggregator = stats_ops.StatsAggregator()
  # ...
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

  # TODO(b/116314787): Update this/add support for V2 summary API.
  def get_summary(self):
    """Returns a string `tf.Tensor` that summarizes the aggregated statistics.

    The returned tensor will contain a serialized `tf.summary.Summary` protocol
    buffer, which can be used with the standard TensorBoard logging facilities.

    Returns:
      A scalar string `tf.Tensor` that summarizes the aggregated statistics.
    """
    return gen_dataset_ops.stats_aggregator_summary(self._resource)


class _SetStatsAggregatorDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that acts as an identity, and sets given stats_aggregator."""

  def __init__(self, input_dataset, stats_aggregator):
    super(_SetStatsAggregatorDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._stats_aggregator = stats_aggregator

  def _as_variant_tensor(self):
    return gen_dataset_ops.set_stats_aggregator_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._stats_aggregator._resource,  # pylint: disable=protected-access
        **dataset_ops.flat_structure(self))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_classes(self):
    return self._input_dataset.output_classes


@tf_export("data.experimental.set_stats_aggregator")
def set_stats_aggregator(stats_aggregator):
  """Set the given `stats_aggregator` for aggregating the input dataset stats.

  Args:
    stats_aggregator: A `tf.data.experimental.StatsAggregator` object.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _SetStatsAggregatorDataset(dataset, stats_aggregator)

  return _apply_fn


# TODO(b/38416882): Properly export in the `tf.data.experimental` API when
# stable or make private / remove.
def bytes_produced_stats(tag):
  """Records the number of bytes produced by each element of the input dataset.

  To consume the statistics, associate a `StatsAggregator` with the output
  dataset.

  Args:
    tag: String. All statistics recorded by the returned transformation will
      be associated with the given `tag`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _StatsDataset(dataset, gen_dataset_ops.bytes_produced_stats_dataset,
                         tag)

  return _apply_fn


@tf_export("data.experimental.latency_stats")
def latency_stats(tag):
  """Records the latency of producing each element of the input dataset.

  To consume the statistics, associate a `StatsAggregator` with the output
  dataset.

  Args:
    tag: String. All statistics recorded by the returned transformation will
      be associated with the given `tag`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _StatsDataset(dataset, gen_dataset_ops.latency_stats_dataset, tag)

  return _apply_fn


class _StatsDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that acts as an identity, and also records statistics."""

  def __init__(self, input_dataset, op_function, tag):
    super(_StatsDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._op_function = op_function
    self._tag = ops.convert_to_tensor(tag, dtype=dtypes.string)

  def _as_variant_tensor(self):
    return self._op_function(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._tag,
        **dataset_ops.flat_structure(self))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_classes(self):
    return self._input_dataset.output_classes
