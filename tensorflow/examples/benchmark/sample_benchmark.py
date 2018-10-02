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
"""Sample TensorFlow benchmark."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf


# Define a class that extends from tf.test.Benchmark.
class SampleBenchmark(tf.test.Benchmark):

  # Note: benchmark method name must start with `benchmark`.
  def benchmarkSum(self):
    with tf.Session() as sess:
      x = tf.constant(10)
      y = tf.constant(5)
      result = tf.add(x, y)

      iters = 100
      start_time = time.time()
      for _ in range(iters):
        sess.run(result)
      total_wall_time = time.time() - start_time

      # Call report_benchmark to report a metric value.
      self.report_benchmark(
          name="sum_wall_time",
          # This value should always be per iteration.
          wall_time=total_wall_time/iters,
          iters=iters)


if __name__ == "__main__":
  tf.test.main()
