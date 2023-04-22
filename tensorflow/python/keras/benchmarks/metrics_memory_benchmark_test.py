# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark tests for Keras metrics memory consumption."""

import numpy as np
import tensorflow as tf

from tensorflow.python.compat import compat

try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


class KerasMetricMemoryBenchmark(tf.test.Benchmark):

  # This test is added to measure the memory footprint for
  # metrics_utils._update_confusion_matrix_variables_optimized().

  def benchmark_auc_memory_usage(self):
    if memory_profiler is None:
      self.skipTest('Skip test since memory_profiler is not available.')

    with compat.forward_compatibility_horizon(2021, 6, 9):
      self.y_true = np.random.randint(2, size=(1024, 1024))
      self.y_pred = np.random.rand(1024, 1024)

      memory_usage_1 = memory_profiler.memory_usage((self.even_thresholds_auc))
      memory_usage_2 = memory_profiler.memory_usage(
          (self.uneven_thresholds_auc))
      # memory usage is a list of number which sampled when running the function
      # The pure memory consumption is approximately max(usage) - min(usage)
      memory_usage_1 = max(memory_usage_1) - min(memory_usage_1)
      memory_usage_2 = max(memory_usage_2) - min(memory_usage_2)

      metrics = {'even_threshold_memory_usage': memory_usage_1,
                 'uneven_threshold_memory_usage': memory_usage_2}
      self.report_benchmark(iters=1, metrics=metrics)

  def even_thresholds_auc(self):
    auc = tf.keras.metrics.AUC(num_thresholds=200)
    self.assertTrue(auc._thresholds_distributed_evenly)

    auc(self.y_true, self.y_pred)

  def uneven_thresholds_auc(self):
    num_thresholds = 200
    thresholds = [x / (num_thresholds - 1) for x in range(num_thresholds)]
    thresholds[100] += 1 / 200
    thresholds = thresholds[1:-1]

    auc = tf.keras.metrics.AUC(thresholds=thresholds)
    self.assertFalse(auc._thresholds_distributed_evenly)
    self.assertEqual(auc.num_thresholds, num_thresholds)

    auc(self.y_true, self.y_pred)


if __name__ == '__main__':
  tf.test.main()
