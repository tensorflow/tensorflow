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
"""Unit test for tf.scan under eager execution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf


class ScanBenchmark(tf.test.Benchmark):

  def runScan(self, n):
    elems = np.arange(n)
    start_time = time.time()
    _ = tf.scan(lambda a, x: a + x, elems, parallel_iterations=1)
    wall_time = time.time() - start_time

    self.report_benchmark(
        name='scan',
        iters=n,
        wall_time=wall_time)

  def benchmarkScan16000(self):
    self.runScan(16000)

  def benchmarkScan32000(self):
    self.runScan(32000)

  def benchmarkScan64000(self):
    self.runScan(64000)

  def benchmarkScan128000(self):
    self.runScan(128000)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
