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
"""Common benchmarking code.

See https://www.tensorflow.org/community/benchmarks for usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

import tensorflow as tf


class ReportingBenchmark(tf.test.Benchmark):
  """Base class for a benchmark that reports general performance metrics.

  Subclasses only need to call one of the _profile methods, and optionally
  report_results.
  """

  def time_execution(self, name, target, iters, warm_up_iters=5):
    for _ in range(warm_up_iters):
      target()

    all_times = []
    for _ in range(iters):
      iter_time = time.time()
      target()
      all_times.append(time.time() - iter_time)

    avg_time = np.average(all_times)

    extras = {}
    extras['all_times'] = all_times

    if isinstance(name, tuple):
      extras['name'] = name
      name = '_'.join(str(piece) for piece in name)

    self.report_benchmark(
        iters=iters, wall_time=avg_time, name=name, extras=extras)


if __name__ == '__main__':
  tf.test.main()
