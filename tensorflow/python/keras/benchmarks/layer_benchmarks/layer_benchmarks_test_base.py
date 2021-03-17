# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Benchmark base to run and report Keras layers benchmark results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from tensorflow.python.keras.benchmarks.layer_benchmarks import run_xprof


class LayerBenchmarksBase(tf.test.Benchmark):
  """Run and report benchmark results.

  The first run is without any profiling to purly measure running time.
  Second run is with xprof but no python trace.
  Third run is with xprof and python trace.
  Note: xprof runs fewer iterations, and the maximum iterations is 100.
  """

  def run_report(self, func, num_iters, metadata=None):
    """Run and report benchmark results for different settings."""

    # 0. Warm up.
    func()

    # 1. Run without profiling.
    start = time.time()
    for _ in range(num_iters):
      func()
    total_time = time.time() - start
    us_mean_time = total_time * 1e6 / num_iters

    metrics = [
        {"name": "examples_per_sec",
         "value": float("{0:.3f}".format(num_iters / total_time))},
        {"name": "us_per_example",
         "value": float("{0:.3f}".format(us_mean_time))}]

    # 2. Run with xprof with no python trace.
    num_iters_xprof = min(100, num_iters)
    xprof_link, us_per_example = run_xprof.run_with_xprof(
        func, num_iters_xprof, False)
    # This xprof link will appear in the benchmark dashboard.
    extras = {
        "xprof_link": xprof_link,
        "us_per_example_with_xprof": us_per_example
    }

    # 3. Run with xprof and python trace.
    xprof_link, us_per_example = run_xprof.run_with_xprof(
        func, num_iters_xprof, True)
    extras["python_trace_xprof_link"] = xprof_link
    extras["us_per_example_with_xprof_and_python"] = us_per_example

    if metadata:
      extras.update(metadata)
    self.report_benchmark(
        iters=num_iters, wall_time=us_mean_time, extras=extras, metrics=metrics)
