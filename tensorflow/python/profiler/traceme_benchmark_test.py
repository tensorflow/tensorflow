# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark TraceMe performance.

To run benchmark:
  bazel run -c opt traceme_benchmark_test -- --benchmarks=.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import test
from tensorflow.python.profiler import traceme

_NUM_ITERS = 1000000


class TracemeBenchmarkTest(test.Benchmark):

  def benchmarkScopedTraceMe(self):
    start_time = time.time()
    for _ in range(_NUM_ITERS):
      with traceme.TraceMe('test'):
        pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    scoped_traceme_time = 1.0 * elapsed_time / _NUM_ITERS

    self.report_benchmark(iters=_NUM_ITERS, wall_time=scoped_traceme_time)

  def benchmarkDirectTraceMe(self):
    start_time = time.time()
    for _ in range(_NUM_ITERS):
      tm = pywrap_tensorflow.PythonTraceMe('test')
      tm.Enter()
      tm.Exit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    direct_traceme_time = 1.0 * elapsed_time / _NUM_ITERS

    self.report_benchmark(iters=_NUM_ITERS, wall_time=direct_traceme_time)

  def benchmarkEmptyLoop(self):
    start_time = time.time()
    for _ in range(_NUM_ITERS):
      pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    empty_loop_time = 1.0 * elapsed_time / _NUM_ITERS

    self.report_benchmark(iters=_NUM_ITERS, wall_time=empty_loop_time)


if __name__ == '__main__':
  test.main()
