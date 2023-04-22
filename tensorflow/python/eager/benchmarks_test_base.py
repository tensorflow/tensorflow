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
r"""Benchmark base to run and report benchmark results."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os
import uuid

from tensorflow.python.eager import test
from tensorflow.python.platform import flags
from tensorflow.python.profiler import profiler_v2 as profiler

flags.DEFINE_bool("xprof", False, "Run and report benchmarks with xprof on")
flags.DEFINE_string("logdir", "/tmp/xprof/", "Directory to store xprof data")


class MicroBenchmarksBase(test.Benchmark):
  """Run and report benchmark results.

  The first run is without any profilng.
  Second run is with xprof and python trace. Third run is with xprof without
  python trace. Note: xprof runs are with fewer iterations.
  """

  def run_with_xprof(self, enable_python_trace, run_benchmark, func,
                     num_iters_xprof, execution_mode, suid):
    if enable_python_trace:
      options = profiler.ProfilerOptions(python_tracer_level=1)
      logdir = os.path.join(flags.FLAGS.logdir, suid + "_with_python")
    else:
      options = profiler.ProfilerOptions(python_tracer_level=0)
      logdir = os.path.join(flags.FLAGS.logdir, suid)
    with profiler.Profile(logdir, options):
      total_time = run_benchmark(func, num_iters_xprof, execution_mode)
    us_per_example = float("{0:.3f}".format(total_time * 1e6 / num_iters_xprof))
    return logdir, us_per_example

  def run_report(self, run_benchmark, func, num_iters, execution_mode=None):
    """Run and report benchmark results."""
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    extras = {
        "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
        "us_per_example": float("{0:.3f}".format(total_time * 1e6 / num_iters))
    }

    if flags.FLAGS.xprof:
      suid = str(uuid.uuid4())
      # Re-run with xprof and python trace.
      num_iters_xprof = min(100, num_iters)
      xprof_link, us_per_example = self.run_with_xprof(True, run_benchmark,
                                                       func, num_iters_xprof,
                                                       execution_mode, suid)
      extras["xprof link with python trace"] = xprof_link
      extras["us_per_example with xprof and python"] = us_per_example

      # Re-run with xprof but no python trace.
      xprof_link, us_per_example = self.run_with_xprof(False, run_benchmark,
                                                       func, num_iters_xprof,
                                                       execution_mode, suid)
      extras["xprof link"] = xprof_link
      extras["us_per_example with xprof"] = us_per_example

    benchmark_name = self._get_benchmark_name()
    self.report_benchmark(
        iters=num_iters, wall_time=mean_us, extras=extras, name=benchmark_name)
