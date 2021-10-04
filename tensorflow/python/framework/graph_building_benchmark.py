# Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
r"""Benchmarks for low-level graph building primitives.

To run CPU benchmarks:
  bazel run -c opt graph_building_benchmarks -- --benchmarks=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" graph_building_benchmarks -- \
    --benchmarks=.

To run a subset of benchmarks using --benchmarks flag.
--benchmarks: the list of benchmarks to run. The specified value is interpreted
as a regular expression and any benchmark whose name contains a partial match
to the regular expression is executed.
e.g. --benchmarks=".*MatMul.*" will run all matmul related benchmarks.

"""
import time

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import test


def run_benchmark(func, num_iters):
  start = time.time()
  for _ in range(num_iters):
    func()
  end = time.time()
  return end - start


class SingleOpBenchmarks(test.Benchmark):
  """Benchmark for graph building time of ops."""

  def _run_and_report(self, func, num_iters):
    total_time = run_benchmark(func, num_iters)
    mean_us = total_time * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={
            "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
        })

  def benchmarkAddScalars(self):
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(shape=[], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(shape=[], dtype=dtypes.float32, name="y")

      def bench():
        return gen_math_ops.add(x, y)

      self._run_and_report(bench, 1000)

  def benchmarkAddBatchedMatrices(self):
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(
          shape=[32, 784, 1000], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(
          shape=[32, 784, 1000], dtype=dtypes.float32, name="y")

      def bench():
        return gen_math_ops.add(x, y)

      self._run_and_report(bench, 1000)

  def benchmarkMatMul(self):
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(
          shape=[784, 1000], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(
          shape=[1000, 1000], dtype=dtypes.float32, name="y")

      def bench():
        return gen_math_ops.mat_mul(x, y)

      self._run_and_report(bench, 1000)


if __name__ == "__main__":
  test.main()
