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
"""Tests for the LatencyAllEdges optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.kernel_tests import stats_dataset_test_base
from tensorflow.contrib.data.python.ops import optimization
from tensorflow.contrib.data.python.ops import stats_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class OptimizeStatsDatasetTest(stats_dataset_test_base.StatsDatasetTestBase):

  def testLatencyStatsOptimization(self):

    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.from_tensors(1).apply(
        optimization.assert_next(
            ["LatencyStats", "Map", "LatencyStats", "Prefetch",
             "LatencyStats"])).map(lambda x: x * x).prefetch(1).apply(
                 optimization.optimize(["latency_all_edges"])).apply(
                     stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      self.assertEqual(1 * 1, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      summary_str = sess.run(summary_t)
      self._assertSummaryHasCount(summary_str,
                                  "record_latency_TensorDataset/_1", 1)
      self._assertSummaryHasCount(summary_str, "record_latency_MapDataset/_4",
                                  1)
      self._assertSummaryHasCount(summary_str,
                                  "record_latency_PrefetchDataset/_6", 1)


if __name__ == "__main__":
  test.main()
