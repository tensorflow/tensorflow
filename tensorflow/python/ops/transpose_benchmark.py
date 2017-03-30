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
"""Benchmark for Transpose op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def build_graph(device, input_shape, perm, datatype, num_iters):
  """Build a graph containing a sequence of conv2d operations.

  Args:
    device: String, the device to run on.
    input_shape: Shape of the input tensor.
    perm: A list of ints with the same length as input tensor's dimension.
    datatype: numpy data type of the input tensor.
    num_iters: number of iterations to run transpose.

  Returns:
    An array of tensors to run()
  """
  with ops.device("/%s:0" % device):
    total_size = np.prod(input_shape)
    inp = np.arange(1, total_size + 1, dtype=datatype).reshape(input_shape)
    t = constant_op.constant(inp, shape=input_shape)

    outputs = []
    outputs.append(array_ops.transpose(t, perm))
    for i in range(1, num_iters):
      with ops.control_dependencies([outputs[i - 1]]):
        outputs.append(array_ops.transpose(t, perm))
    return control_flow_ops.group(*outputs)


class TransposeBenchmark(test.Benchmark):
  """Benchmark transpose!"""

  def _run_graph(self, device, input_shape, perm, num_iters, datatype):
    """Run the graph and print its execution time.

    Args:
      device: String, the device to run on.
      input_shape: Shape of the input tensor.
      perm: A list of ints with the same length as input tensor's dimension.
      num_iters: Number of iterations to run the benchmark.
      datatype: numpy data type of the input tensor.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      outputs = build_graph(device, input_shape, perm, datatype, num_iters)
      with session_lib.Session(graph=graph) as session:
        variables.global_variables_initializer().run()
        # warmup runs
        session.run(outputs)
        start_time = time.time()
        session.run(outputs)
        duration = (time.time() - start_time) / num_iters
        throughput = np.prod(np.array(
            input_shape)) * datatype().itemsize * 2 / duration / 1e9
        print("%s %s inputshape:%s perm:%s %d %.6fsec, %.4fGB/s." %
              (device, str(datatype), str(input_shape).replace(" ", ""),
               str(perm).replace(" ", ""), num_iters, duration, throughput))

    name_template = (
        "transpose_{device}_{dtype}_input_shape_{inputshape}_perm_{perm}")

    self.report_benchmark(
        name=name_template.format(
            device=device,
            dtype=str(datatype).replace(" ", ""),
            inputshape=str(input_shape).replace(" ", ""),
            perm=str(perm).replace(" ", "")).replace(" ", ""),
        iters=num_iters,
        wall_time=duration)

    return duration

  def benchmark_transpose(self):
    print("transpose benchmark:")

    datatypes = [np.complex128, np.float64, np.float32, np.float16, np.int8]

    small_shapes = [[2, 20, 20, 20, 16], [2, 16, 20, 20, 20]] * 2 + [[
        2, 100, 100, 16
    ], [2, 16, 100, 100]] * 2 + [[2, 5000, 16], [2, 16, 5000]] * 2
    small_perms = [[0, 4, 1, 2, 3], [0, 2, 3, 4, 1]] + [[4, 1, 2, 3, 0]] * 2 + [
        [0, 3, 1, 2], [0, 2, 3, 1]
    ] + [[3, 1, 2, 0]] * 2 + [[0, 2, 1]] * 2 + [[2, 1, 0]] * 2

    large_shapes = [[2, 100, 100, 100, 32], [2, 100, 100, 100, 64]] * 2 + [[
        2, 1000, 1000, 32
    ], [2, 1000, 1000, 64]] * 2 + [[2, 1000000, 32], [2, 1000000, 64]] * 2
    large_perms = [[0, 4, 1, 2, 3], [0, 2, 3, 4, 1]] + [[4, 1, 2, 3, 0]] * 2 + [
        [0, 3, 1, 2], [0, 2, 3, 1]
    ] + [[3, 1, 2, 0]] * 2 + [[0, 2, 1]] * 2 + [[2, 1, 0]] * 2

    huge_shapes = [[2, 100, 100, 100, 128], [2, 1000, 1000, 128],
                   [2, 1000000, 128]] * 2
    huge_perms = [[0, 4, 1, 2, 3], [0, 3, 1, 2], [0, 2, 1], [4, 1, 2, 3, 0],
                  [3, 1, 2, 0], [2, 1, 0]]

    num_iters = 40
    for datatype in datatypes:
      for ishape, perm in zip(small_shapes, small_perms):
        self._run_graph("gpu", ishape, perm, num_iters, datatype)

      if datatype is not np.complex128:
        if datatype is not np.float16:
          for ishape, perm in zip(large_shapes, large_perms):
            self._run_graph("gpu", ishape, perm, num_iters, datatype)

      if datatype is not np.complex128:
        if datatype is not np.float64:
          if datatype is not np.float16:
            for ishape, perm in zip(huge_shapes, huge_perms):
              self._run_graph("gpu", ishape, perm, num_iters, datatype)

if __name__ == "__main__":
  test.main()
