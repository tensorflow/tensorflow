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
"""Benchmark for split and grad of split."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging


def build_graph(device, input_shape, output_sizes, axis):
  """Build a graph containing a sequence of split operations.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensor.
    output_sizes: size of each output along axis.
    axis: axis to be split along.

  Returns:
    An array of tensors to run()
  """
  with tf.device("/%s:0" % device):
    inp = tf.zeros(input_shape)

    outputs = []
    for _ in range(100):
      outputs.extend(tf.split(inp, output_sizes, axis))
    return tf.group(*outputs)


class SplitBenchmark(tf.test.Benchmark):
  """Benchmark split!"""

  def _run_graph(self, device, output_shape, variable, num_outputs, axis):
    """Run the graph and print its execution time.

    Args:
      device: string, the device to run on.
      output_shape: shape of each output tensors.
      variable: whether or not the output shape should be fixed
      num_outputs: the number of outputs to split the input into
      axis: axis to be split

    Returns:
      The duration of the run in seconds.
    """
    graph = tf.Graph()
    with graph.as_default():
      if not variable:
        if axis == 0:
          input_shape = [output_shape[0] * num_outputs, output_shape[1]]
          sizes = [output_shape[0] for _ in range(num_outputs)]
        else:
          input_shape = [output_shape[0], output_shape[1] * num_outputs]
          sizes = [output_shape[1] for _ in range(num_outputs)]
      else:
        sizes = np.random.randint(
            low=max(1, output_shape[axis] - 2),
            high=output_shape[axis] + 2,
            size=num_outputs)
        total_size = np.sum(sizes)
        if axis == 0:
          input_shape = [total_size, output_shape[1]]
        else:
          input_shape = [output_shape[0], total_size]

      outputs = build_graph(device, input_shape, sizes, axis)
    config = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L0)))
    with tf.Session(graph=graph, config=config) as session:
      logging.set_verbosity("info")
      tf.global_variables_initializer().run()
      bench = benchmark.TensorFlowBenchmark()
      bench.run_op_benchmark(
          session,
          outputs,
          mbs=input_shape[0] * input_shape[1] * 4 * 2 * 100 / 1e6,
          extras={
              "input_shape": input_shape,
              "variable": variable,
              "axis": axis
          })

  def benchmark_split(self):
    print("Forward vs backward concat")
    shapes = [[2000, 8], [8, 2000], [100, 18], [1000, 18], [10000, 18],
              [100, 97], [1000, 97], [10000, 1], [1, 10000]]
    axis_ = [1]  # 0 is very fast because it doesn't actually do any copying
    num_outputs = 100
    variable = [False, True]  # fixed input size or not
    for shape in shapes:
      for axis in axis_:
        for v in variable:
          self._run_graph("gpu", shape, v, num_outputs, axis)


if __name__ == "__main__":
  tf.test.main()
