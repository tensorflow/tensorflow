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
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(None, "Use `tf.data.experimental.StatsOptions`.")
def set_stats_aggregator(stats_aggregator, prefix="", counter_prefix=""):
  """Set the given `stats_aggregator` for aggregating the input dataset stats.

  Args:
    stats_aggregator: A `tf.data.experimental.StatsAggregator` object.
    prefix: (Optional) String, all statistics recorded for the input `dataset`
      will have given `prefix` prepend with the name.
    counter_prefix: (Optional) String, all statistics recorded as `counters`
      will have the given `prefix` for the counter. Defaults to "/tensorflow".

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return dataset_ops._SetStatsAggregatorDataset(  # pylint: disable=protected-access
        dataset, stats_aggregator, prefix, counter_prefix)

  return _apply_fn


@tf_export("data.experimental.bytes_produced_stats")
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
    return _StatsDataset(
        dataset, gen_experimental_dataset_ops.bytes_produced_stats_dataset, tag)

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
    return _StatsDataset(
        dataset, gen_experimental_dataset_ops.latency_stats_dataset, tag)

  return _apply_fn


class _StatsDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, and also records statistics."""

  def __init__(self, input_dataset, op_function, tag):
    self._input_dataset = input_dataset
    self._op_function = op_function
    self._tag = ops.convert_to_tensor(tag, dtype=dtypes.string)
    variant_tensor = self._op_function(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._tag,
        **self._flat_structure)
    super(_StatsDataset, self).__init__(input_dataset, variant_tensor)
