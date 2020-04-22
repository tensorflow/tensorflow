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

from tensorflow.python.eager import test


class MicroBenchmarksBase(test.Benchmark):
  """Run and report benchmark results."""

  def run_report(self, run_benchmark, func, num_iters, execution_mode=None):
    """Run and report benchmark results."""
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    extras = {
        "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
        "us_per_example": float("{0:.3f}".format(total_time * 1e6 / num_iters))
    }
    self.report_benchmark(iters=num_iters, wall_time=mean_us, extras=extras)
