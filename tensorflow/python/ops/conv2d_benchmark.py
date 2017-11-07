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
"""Benchmark for Conv2D op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import time

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def build_graph(device, input_shape, filter_shape, strides, padding, num_iters):
  """builds a graph containing a sequence of conv2d operations.

  Args:
    device: String, the device to run on.
    input_shape: Shape of the input tensor.
    filter_shape: Shape of the filter tensor.
    strides: A list of ints. 1-D of length 4. The stride of sliding
             window for each dimension of input.
    padding: A string from: "SAME", "VALID". The type of padding
             algorithm to use.
    num_iters: number of iterations to run conv2d.

  Returns:
    An array of tensors to run()
  """
  with ops.device("/%s:0" % device):
    inp = variables.Variable(random_ops.truncated_normal(input_shape))
    filt = variables.Variable(random_ops.truncated_normal(filter_shape))

    outputs = []
    conv2d_op = nn_ops.conv2d(inp, filt, strides, padding, data_format="NHWC")
    outputs.append(conv2d_op)
    for _ in range(1, num_iters):
      with ops.control_dependencies([conv2d_op]):
        conv2d_op = nn_ops.conv2d(
            inp, filt, strides, padding, data_format="NHWC")
        outputs.append(conv2d_op)
    return control_flow_ops.group(*outputs)


class Conv2DBenchmark(test.Benchmark):
  """Benchmark conv2d!"""

  def _run_graph(self, device, input_shape, filter_shape, strides, padding,
                 num_iters):
    """runs the graph and print its execution time.

    Args:
      device: String, the device to run on.
      input_shape: Shape of the input tensor.
      filter_shape: Shape of the filter tensor.
      strides: A list of ints. 1-D of length 4. The stride of sliding
               window for each dimension of input.
      padding: A string from: "SAME", "VALID". The type of padding
               algorithm to use.  num_iters: Number of iterations to run the
                 benchmark.
      num_iters: number of iterations to run conv2d.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      outputs = build_graph(device, input_shape, filter_shape, strides, padding,
                            num_iters)
      with session_lib.Session(graph=graph) as session:
        variables.global_variables_initializer().run()
        # warmup runs
        session.run(outputs)

        start_time = time.time()
        session.run(outputs)
        duration = (time.time() - start_time) / num_iters

        print("%s inputshape:%s filtershape:%s strides:%s padding:%s "
              "%d iters: %.8f sec" %
              (device, str(input_shape).replace(" ", ""),
               str(filter_shape).replace(" ", ""),
               str(strides).replace(" ", ""), padding, num_iters, duration))

    name_template = (
        "conv2d_{device}_input_shape_{inputshape}_filter_shape_{filtershape}_"
        "strides_{strides}_padding_{padding}")

    self.report_benchmark(
        name=name_template.format(
            device=device,
            inputshape=str(input_shape).replace(" ", ""),
            filtershape=str(filter_shape).replace(" ", ""),
            strides=str(strides).replace(" ", ""),
            padding=padding).replace(" ", ""),
        iters=num_iters,
        wall_time=duration / num_iters)

    return duration

  def benchmark_conv2d(self):
    print("conv2d benchmark:")

    h = 500
    w = 500
    fh = 3
    fw = 3
    input_shapes = []
    filter_shapes = []
    for b, c in itertools.product([4, 16, 32], [i for i in range(3, 16)]):
      input_shapes += [[b, h, w, c]]
      filter_shapes += [[fh, fw, c, b]]
    strides = [[1, 2, 2, 1]]
    paddings = ["VALID", "SAME"]
    for ishape, fshape in zip(input_shapes, filter_shapes):
      for stride in strides:
        for padding in paddings:
          self._run_graph("gpu", ishape, fshape, stride, padding, 80)


if __name__ == "__main__":
  test.main()
