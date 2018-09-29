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
"""Tests for the StatsDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import stats_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# TODO(shivaniagrawal): Can not checkpoint input_pipeline with the
# transformation `stats_ops.set_stats_aggregator`, since we don't support
# serializing StatsAggregator yet.
class StatsDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_dataset_bytes_stats(self, num_elements):
    return dataset_ops.Dataset.range(num_elements).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).apply(
            stats_ops.bytes_produced_stats("bytes_produced"))

  def test_bytes_produced_stats_invalid_tag_shape(self):
    with self.assertRaisesRegexp(
        ValueError, "Shape must be rank 0 but is rank 1"):
      # pylint: disable=g-long-lambda
      self.run_core_tests(
          lambda: dataset_ops.Dataset.range(100).apply(
              stats_ops.bytes_produced_stats(["bytes_produced"])),
          None, 100)
      # pylint: enable=g-long-lambda

  def testBytesStatsDatasetSaveableCore(self):
    num_outputs = 100
    self.run_core_tests(
        lambda: self._build_dataset_bytes_stats(num_outputs),
        lambda: self._build_dataset_bytes_stats(num_outputs // 10), num_outputs)

  def _build_dataset_latency_stats(self, num_elements, tag="record_latency"):
    return dataset_ops.Dataset.range(num_elements).apply(
        stats_ops.latency_stats(tag))

  def _build_dataset_multiple_tags(self,
                                   num_elements,
                                   tag1="record_latency",
                                   tag2="record_latency_2"):
    return dataset_ops.Dataset.range(num_elements).apply(
        stats_ops.latency_stats(tag1)).apply(stats_ops.latency_stats(tag2))

  def test_latency_stats_invalid_tag_shape(self):
    with self.assertRaisesRegexp(
        ValueError, "Shape must be rank 0 but is rank 1"):
      # pylint: disable=g-long-lambda
      self.run_core_tests(
          lambda: dataset_ops.Dataset.range(100).apply(
              stats_ops.latency_stats(["record_latency", "record_latency_2"])),
          None, 100)
      # pylint: enable=g-long-lambda

  def testLatencyStatsDatasetSaveableCore(self):
    num_outputs = 100

    self.run_core_tests(
        lambda: self._build_dataset_latency_stats(num_outputs),
        lambda: self._build_dataset_latency_stats(num_outputs // 10),
        num_outputs)

    self.run_core_tests(lambda: self._build_dataset_multiple_tags(num_outputs),
                        None, num_outputs)

    tag1 = "record_latency"
    tag2 = "record_latency"
    self.run_core_tests(
        lambda: self._build_dataset_multiple_tags(num_outputs, tag1, tag2),
        None, num_outputs)

  def _build_dataset_stats_aggregator(self):
    stats_aggregator = stats_ops.StatsAggregator()
    return dataset_ops.Dataset.range(10).apply(
        stats_ops.set_stats_aggregator(stats_aggregator))

  def test_set_stats_aggregator_not_support_checkpointing(self):
    with self.assertRaisesRegexp(errors.UnimplementedError,
                                 "does not support checkpointing"):
      self.run_core_tests(self._build_dataset_stats_aggregator, None, 10)


if __name__ == "__main__":
  test.main()
