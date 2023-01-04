# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for adding ops to a graph."""

import timeit

from tensorflow.core.config import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


@test_util.add_graph_building_optimization_tests
class GraphBuildingBenchmark(test.Benchmark):

  def _computeAddOpDuration(self, num_ops, num_iters):
    def add_op_to_graph(num_ops):
      with func_graph.FuncGraph("add").as_default():
        a = gen_array_ops.placeholder(dtypes.float32)
        b = gen_array_ops.placeholder(dtypes.float32)
        for _ in range(num_ops):
          gen_math_ops.add(a, b)

    runtimes = timeit.repeat(
        lambda: add_op_to_graph(num_ops), repeat=10, number=num_iters)
    return min(runtimes) / num_iters

  def _computeReadVariableOpDuration(self, num_ops, num_iters):
    def add_op_to_graph(num_ops):
      with func_graph.FuncGraph("resource").as_default():
        handle = resource_variable_ops.var_handle_op(
            dtype=dtypes.int32, shape=[])
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant(1, dtype=dtypes.int32))
        for _ in range(num_ops):
          gen_resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)

    runtimes = timeit.repeat(
        lambda: add_op_to_graph(num_ops), repeat=10, number=num_iters)
    return min(runtimes) / num_iters

  def benchmarkAddOp(self):
    num_ops = 100
    num_iters = 10
    duration = self._computeAddOpDuration(num_ops, num_iters)
    name = "BenchmarkAddOp"
    if flags.config().graph_building_optimization.value():
      name += "WithGraphBuildingOptimization"
    self.report_benchmark(
        name=name,
        iters=num_iters,
        wall_time=duration,
        extras={"num_ops": num_ops})

  def benchmarkResourceVariableOp(self):
    num_ops = 100
    num_iters = 10
    duration = self._computeReadVariableOpDuration(num_ops, num_iters)
    name = "BenchmarkReadVariableOp"
    if flags.config().graph_building_optimization.value():
      name += "WithGraphBuildingOptimization"
    self.report_benchmark(
        name=name,
        iters=num_iters,
        wall_time=duration,
        extras={"num_ops": num_ops})

if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
