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

import tempfile

from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.util.tf_export import tf_export


_DEFAULT_MAX_QUEUE = 10


@tf_export("data.experimental.StatsAggregator", v1=[])
class StatsAggregatorV2(object):
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
  options = tf.data.Options()
  options.experimental_stats.aggregator = aggregator
  dataset = dataset.with_options(options)
  ```

  Note: This interface is experimental and expected to change. In particular,
  we expect to add other implementations of `StatsAggregator` that provide
  different ways of exporting statistics, and add more types of statistics.
  """

  def __init__(self):
    self._resource = ged_ops.stats_aggregator_handle_v2()
    # There could be a conflict with multiple file writer in the same logdir,
    # (b/37351340). Possible workarounds till this bug is resolved are a) having
    # multiple dataset stats specific file inside log_dir and b) get default
    # summary writer, getting default summary writer quite doesn't solve the
    # problem as there might be summary writers in log dir not set as default
    # e.g. in Keras calback.
    # Creating a summary_writer here could potentially be replaced with getting
    # the default summary_writer if any, creating it otherwise or a public
    # method to associate summary writer.
    self._logdir = tempfile.mkdtemp()
    self._summary_writer = summary_ops_v2.create_file_writer_v2(
        self._logdir, max_queue=_DEFAULT_MAX_QUEUE)
    ged_ops.stats_aggregator_set_summary_writer(self._resource,
                                                self._summary_writer._resource)  # pylint: disable=protected-access


@tf_export(v1=["data.experimental.StatsAggregator"])
class StatsAggregatorV1(object):
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
  options = tf.data.Options()
  options.experimental_stats.aggregator = aggregator
  dataset = dataset.with_options(options)
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
    self._resource = ged_ops.experimental_stats_aggregator_handle()

  def get_summary(self):
    """Returns a string `tf.Tensor` that summarizes the aggregated statistics.

    The returned tensor will contain a serialized `tf.summary.Summary` protocol
    buffer, which can be used with the standard TensorBoard logging facilities.

    Returns:
      A scalar string `tf.Tensor` that summarizes the aggregated statistics.
    """
    return ged_ops.experimental_stats_aggregator_summary(self._resource)


# TODO(b/116314787): Change this to StatsAggregatorV2 when we have stable
# SummaryWriterInterface, and do not break any users.
StatsAggregator = StatsAggregatorV1
