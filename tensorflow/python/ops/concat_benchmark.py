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

import itertools
import random
import time

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("use_gpu", True, """Run GPU benchmarks.""")


def build_graph(device, input_shape, variable, num_inputs, axis, grad):
  """Build a graph containing a sequence of concat operations.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensors.
    variable: whether or not to randomize the input shape
    num_inputs: the number of inputs to concat
    axis: axis to be concat'ed
    grad: if True compute the gradient

  Returns:
    An array of tensors to run()
  """
  with tf.device("/%s:0" % device):
    if not variable:
      inputs = [tf.zeros(input_shape) for _ in range(num_inputs)]
    else:
      if axis == 1:
        inputs = [
            tf.zeros([
                input_shape[0],
                random.randint(max(1, input_shape[1] - 5), input_shape[1] + 5)
            ]) for _ in range(num_inputs)
        ]
      else:
        inputs = [
            tf.zeros([
                random.randint(max(1, input_shape[0] - 5), input_shape[0] + 5),
                input_shape[1]
            ]) for _ in range(num_inputs)
        ]

    outputs = [tf.concat_v2(inputs, axis) for _ in range(100)]
    if grad:
      return tf.group(*list(
          itertools.chain.from_iterable(
              [tf.gradients(output, inputs) for output in outputs])))
    else:
      return tf.group(*outputs)


class ConcatBenchmark(tf.test.Benchmark):
  """Benchmark concat."""

  def _run_graph(self, device, input_shape, variable, num_inputs, axis, grad,
                 num_iters):
    """Run the graph and print its execution time.

    Args:
      device: string, the device to run on.
      input_shape: shape of the input tensors.
      variable: whether or not the input shape should be fixed
      num_inputs: the number of inputs to concat
      axis: axis to be concat'ed
      grad: if True compute the gradient
      num_iters: number of steps to run.

    Returns:
      The duration of the run in seconds.
    """
    graph = tf.Graph()
    with graph.as_default():
      outputs = build_graph(device, input_shape, variable, num_inputs, axis,
                            grad)
    config = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L0)))
    with tf.Session(graph=graph, config=config) as session:
      tf.global_variables_initializer().run()
      _ = session.run(outputs)  # warm up.
      start_time = time.time()
      for _ in range(num_iters):
        _ = session.run(outputs)
      duration = time.time() - start_time
      print("%s shape:%d/%d var: %r #inputs:%d axis:%d grad:%r - %f secs - %f "
            "GB/sec" % (device, input_shape[0], input_shape[1], variable,
                        num_inputs, axis, grad, duration / num_iters,
                        num_inputs * input_shape[0] * input_shape[1] * 4 * 2 *
                        100 / (duration / num_iters) / 1e9))

    name_template = (
        "concat_bench_{device}_input_shape_{shape}_variable_{variable}"
        "_num_inputs_{num_inputs}_axis_{axis}_grad_{grad}")

    self.report_benchmark(name=name_template.format(
        device=device,
        num_inputs=num_inputs,
        variable=variable,
        grad=grad,
        shape=str(input_shape).replace(" ", ""),
        axis=str(axis),
        iters=num_iters))

    return duration

  def benchmark_concat(self):
    print("Forward vs backward concat")
    shapes = [[2000, 8], [8, 2000], [100, 18], [1000, 18], [100, 97],
              [1000, 97], [10000, 1], [1, 10000]]
    axis_ = [0, 1]
    num_inputs = 20
    num_iters = [10] * len(shapes)
    variable = [False, True]  # fixed input size or not
    for shape, iters in zip(shapes, num_iters):
      for axis in axis_:
        for v in variable:
          self._run_graph("cpu", shape, v, num_inputs, axis, True, iters)


if __name__ == "__main__":
  tf.test.main()
