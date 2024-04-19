# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark for Matmul operator."""

import itertools
import time

import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def build_graph(device, n, m, k, transpose_a, transpose_b, dtype):
  """Build a graph containing a sequence of matmul operations.

  Args:
    device: String, the device to run on.
    n: tensor A's first dimension size.
    m: tensor A's second dimension size.
    k: tensor B's second dimension size.
    transpose_a: boolean value to show if tensor A is transposed.
    transpose_b: boolean value to show if tensor B is transposed.
    dtype: numpy data type of the input tensor.

  Returns:
    A matmul operation to run()
  """
  with ops.device('%s' % device):
    if not transpose_a:
      x = variable_v1.VariableV1(
          random_ops.random_uniform([n, m], dtype=dtype), use_resource=False)
    else:
      x = variable_v1.VariableV1(
          random_ops.random_uniform([m, n], dtype=dtype), use_resource=False)
    if not transpose_b:
      y = variable_v1.VariableV1(
          random_ops.random_uniform([m, k], dtype=dtype), use_resource=False)
    else:
      y = variable_v1.VariableV1(
          random_ops.random_uniform([k, m], dtype=dtype), use_resource=False)

    z = math_ops.matmul(x, y, transpose_a=transpose_a, transpose_b=transpose_b)
    return control_flow_ops.group(z)


class MatmulBenchmark(test.Benchmark):
  """Benchmark matmul!"""

  def run_graph(self, device, n, m, k, transpose_a, transpose_b, num_iters,
                dtype):
    """Run the graph and print its execution time.

    Args:
      device: String, the device to run on.
      n: tensor A's first dimension size.
      m: tensor A's second dimension size.
      k: tensor B's second dimension size.
      transpose_a: boolean value to show if tensor A is transposed.
      transpose_b: boolean value to show if tensor B is transposed.
      num_iters: number of iterations to run the benchmark.
      dtype: numpy data type of the input tensor.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      output = build_graph(device, n, m, k, transpose_a, transpose_b, dtype)
      with session_lib.Session(graph=graph) as session:
        variables.global_variables_initializer().run()
        for _ in range(500):
          session.run(output)
        start_time = time.time()
        for _ in range(num_iters):
          session.run(output)
        duration = (time.time() - start_time)
        num_items = n * m * k * 2
        throughput = num_items * num_iters / duration / 1e9
        print('%s %s input_info:%s %d %.4fsec, %.4fGitems/s.' %
              (device, str(dtype), str(n) + 'x' + str(m) + 'x' + str(k) +
               ',ta:' + str(transpose_a) + '.tb:' + str(transpose_b), num_iters,
               duration, throughput))

    name_template = ('matmul_{device}_{dtype}_input_info_{inputinfo}')

    self.report_benchmark(
        name=name_template.format(
            device=device,
            dtype=str(dtype).replace(' ', ''),
            inputinfo=str(n) + 'x' + str(m) + 'x' + str(k) + ',ta:' +
            str(transpose_a) + ',tb:' + str(transpose_b)).replace(' ', ''),
        iters=num_iters,
        wall_time=duration)
    return duration

  def run_test_gpu(self, n, m, k, transpose_a, transpose_b, dtype, num_iters):
    self.run_graph(test.gpu_device_name(), n, m, k, transpose_a, transpose_b,
                   num_iters, dtype)

  def test_round(self, num_iters):
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      for n, m, (transpose_a, transpose_b) in itertools.product(
          [512, 1024], [1, 8, 16, 128], [(False, False), (True, False),
                                         (False, True)]):
        k = n
        self.run_test_gpu(n, m, k, transpose_a, transpose_b, dtype, num_iters)

      for n, m, k, (transpose_a, transpose_b) in itertools.product(
          [200], [1, 8, 20], [10000], [(False, False), (True, False),
                                       (False, True)]):
        self.run_test_gpu(n, m, k, transpose_a, transpose_b, dtype, num_iters)

      for (n, m, k), (transpose_a, transpose_b) in itertools.product(
          [(200, 20, 20000), (1, 10000, 200)], [(False, False), (True, False),
                                                (False, True)]):
        self.run_test_gpu(n, m, k, transpose_a, transpose_b, dtype, num_iters)

  def benchmark_matmul(self):
    self.test_round(num_iters=200)


if __name__ == '__main__':
  test.main()
