# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""StatsAggregator for aggregating statistics from `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
  aggregator = tf.data.experimental.StatsAggregator()
  dataset = ...

  # Apply `StatsOptions` to associate `dataset` with `aggregator`.
  options = dataset_ops.Options()
  options.experimental_stats = tf.data.experimental.StatsOptions(aggregator)
  dataset = dataset.with_options(options)
  iterator = dataset.make_one_shot_iterator()
  ```

  To get a protocol buffer summary of the currently aggregated statistics,
  use the `StatsAggregator.get_summary()` tensor. The easiest way to do this
  is to add the returned tensor to the `tf.GraphKeys.SUMMARIES` collection,
  so that the summaries will be included with any existing summaries.

  ```python
  aggregator = tf.data.experimental.StatsAggregator()
  # ...
  stats_summary = aggregator.get_summary()
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
