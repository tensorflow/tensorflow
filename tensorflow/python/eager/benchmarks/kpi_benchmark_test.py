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
r"""KPI Benchmarks for low-level eager execution primitives.

This is a suite of full end-to-end integration benchmakr for low-level eager
execution APIs. Also tracks them as KPI Traceme.

To run CPU benchmarks:
  bazel run -c opt kpi_benchmarks_test -- --benchmarks=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" kpi_benchmarks_test -- \
    --benchmarks=.

To run a subset of benchmarks using --benchmarks flag.
--benchmarks: the list of benchmarks to run. The specified value is interpreted
as a regular expression and any benchmark whose name contains a partial match
to the regular expression is executed.
e.g. --benchmarks=".*matmul*." will run all matmul related benchmarks.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import time

import tensorflow as tf

from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.profiler import trace

NUM_ITERATIONS = 30000


def _run_benchmark(func, num_iters, execution_mode=None):
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in range(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start


class KpiBenchmarks(benchmarks_test_base.MicroBenchmarksBase):
  """A Collection of KPI benchmarks."""

  def _get_benchmark_name(self):
    return self._get_name()

  def _run(self, func, num_iters):
    gc.disable()
    gc.collect()
    self.run_report(_run_benchmark, func, num_iters)
    gc.enable()

  def benchmark_tf_constant_2x2(self):
    x = [[1., 2.], [3., 4.]]

    def fn():
      with trace.Trace("tf.constant-2x2"):
        tf.constant(x)

    self._run(fn, NUM_ITERATIONS)

  def benchmark_tf_convert_to_tensor_2x2(self):
    x = [[1., 2.], [3., 4.]]

    def fn():
      with trace.Trace("tf.convert_to_tensor-2x2"):
        tf.convert_to_tensor(x)

    self._run(fn, NUM_ITERATIONS)

  def benchmark_tf_nn_relu_2x2(self):
    x = tf.constant([[1., 2.], [3., 4.]])

    def fn():
      with trace.Trace("tf.nn.relu-2x2"):
        tf.nn.relu(x)

    self._run(fn, NUM_ITERATIONS)

  def benchmark_tf_function_invocation_identity(self):
    x = tf.constant([[1., 2.], [3., 4.]])

    @tf.function
    def identity(x):
      return x

    def fn():
      with trace.Trace("tf.function-identity"):
        identity(x)

    self._run(fn, NUM_ITERATIONS)


if __name__ == "__main__":
  tf.test.main()
