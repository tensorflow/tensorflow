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
"""Benchmark for fused conv2d bias and activation op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

from tensorflow.contrib.fused_conv.python.ops import fused_conv2d_bias_activation_op
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def build_conv_bias_relu_graph(device, input_shape, filter_shape, strides,
                               padding, num_iters, data_format):
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
    data_format: data format string of input, 'NHWC' and 'NCHW' are
    supported.

  Returns:
    An array of tensors to run()
  """
  if data_format == "NCHW":
    input_shape = [
        input_shape[0], input_shape[3], input_shape[1], input_shape[2]
    ]
  with ops.device("/%s:0" % device):
    inp = variables.Variable(random_ops.truncated_normal(input_shape))
    filt = variables.Variable(random_ops.truncated_normal(filter_shape))
    bias_shape = [filter_shape[-1]]
    bias = variables.Variable(random_ops.truncated_normal(bias_shape))

    outputs = []
    conv2d_out = nn_ops.conv2d(
        inp, filt, strides, padding, data_format=data_format)
    bias_out = nn_ops.bias_add(conv2d_out, bias, data_format=data_format)
    relu_out = nn_ops.relu(bias_out)
    outputs.append(relu_out)
    for _ in range(1, num_iters):
      with ops.control_dependencies([relu_out]):
        conv2d_out = nn_ops.conv2d(
            inp, filt, strides, padding, data_format=data_format)
        bias_out = nn_ops.bias_add(conv2d_out, bias, data_format=data_format)
        relu_out = nn_ops.relu(bias_out)
        outputs.append(relu_out)
    return control_flow_ops.group(*outputs)


def build_fused_conv_bias_relu_graph(device, input_shape, filter_shape, strides,
                                     padding, num_iters, data_format):
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
    data_format: data format string of input, 'NHWC' and 'NCHW' are
    supported.

  Returns:
    An array of tensors to run()
  """
  if data_format == "NCHW":
    input_shape = [
        input_shape[0], input_shape[3], input_shape[1], input_shape[2]
    ]
  with ops.device("/%s:0" % device):
    inp = variables.Variable(random_ops.truncated_normal(input_shape))
    filt = variables.Variable(random_ops.truncated_normal(filter_shape))
    bias_shape = [filter_shape[-1]]
    bias = variables.Variable(random_ops.truncated_normal(bias_shape))

    outputs = []
    fused_out = fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
        inp,
        filt,
        bias,
        strides,
        padding,
        data_format=data_format,
        activation_mode="Relu")
    outputs.append(fused_out)
    for _ in range(1, num_iters):
      with ops.control_dependencies([fused_out]):
        # pylint: disable=g-line-too-long
        fused_out = fused_conv2d_bias_activation_op.fused_conv2d_bias_activation(
            inp,
            filt,
            bias,
            strides,
            padding,
            data_format=data_format,
            activation_mode="Relu")
        outputs.append(fused_out)
    return control_flow_ops.group(*outputs)


class FusedConv2DBiasActivationBenchmark(test.Benchmark):
  """Benchmark conv2d!"""

  def _run_graph(self, device, input_shape, filter_shape, strides, padding,
                 num_iters, data_format):
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
      data_format: data format string of input, 'NHWC' and 'NCHW' are
      supported.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      outputs = build_fused_conv_bias_relu_graph(device, input_shape,
                                                 filter_shape, strides, padding,
                                                 num_iters, data_format)
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
        wall_time=duration)

    return duration

  def benchmark_fused_conv2d_bias_activation(self):

    stride = [1, 1, 1, 1]
    paddings = ["VALID", "SAME"]
    data_formats = ["NHWC", "NCHW"]

    resnet50_input_shapes = [[64, 14, 14, 256], [64, 14, 14, 256], [
        64, 14, 14, 1024
    ], [64, 55, 55, 64], [64, 28, 28, 128], [64, 28, 28, 128], [64, 55, 55, 64],
                             [64, 7, 7, 512], [64, 7, 7, 512],
                             [64, 28, 28, 512], [64, 55, 55,
                                                 256], [64, 7, 7, 2048]]

    resnet50_filter_shapes = [[1, 1, 256, 1024], [3, 3, 256, 256], [
        1, 1, 1024, 256
    ], [1, 1, 64, 256], [1, 1, 128, 512], [3, 3, 128, 128], [3, 3, 64, 64], [
        3, 3, 512, 512
    ], [1, 1, 512, 2048], [1, 1, 512, 128], [1, 1, 256, 64], [1, 1, 2048, 512]]

    inception3_input_shapes = [[64, 17, 17, 768], [64, 35, 35, 96], [
        64, 35, 35, 288
    ], [64, 8, 8, 384], [64, 8, 8, 384], [64, 17, 17, 192], [64, 35, 35, 64], [
        64, 17, 17, 192
    ], [64, 17, 17, 160], [64, 17, 17, 160], [64, 17, 17, 768], [
        64, 35, 35, 256
    ], [64, 35, 35, 48], [64, 35, 35, 192], [64, 17, 17, 128], [
        64, 17, 17, 160
    ], [64, 8, 8, 448], [64, 17, 17, 128], [64, 17, 17, 768], [64, 17, 17, 160]]
    inception3_filter_shapes = [[1, 1, 768, 192], [3, 3, 96, 96], [
        1, 1, 288, 64
    ], [1, 3, 384, 384], [3, 1, 384, 384], [7, 1, 192, 192], [3, 3, 64, 96], [
        1, 7, 192, 192
    ], [7, 1, 160, 160], [1, 7, 160, 160], [1, 1, 768, 160], [1, 1, 256, 64], [
        5, 5, 48, 64
    ], [1, 1, 192, 64], [1, 7, 128, 128], [1, 7, 160, 192], [3, 3, 448, 384],
                                [7, 1, 128, 128], [1, 1, 768,
                                                   128], [7, 1, 160, 192]]

    print("fused conv2d bias activation benchmark using resnet50's shapes:")
    for ishape, fshape in zip(resnet50_input_shapes, resnet50_filter_shapes):
      for padding in paddings:
        for data_format in data_formats:
          self._run_graph("gpu", ishape, fshape, stride, padding, 80,
                          data_format)
    print("fused conv2d bias activation benchmark using inception3's shapes:")
    for ishape, fshape in zip(inception3_input_shapes,
                              inception3_filter_shapes):
      for padding in paddings:
        for data_format in data_formats:
          self._run_graph("gpu", ishape, fshape, stride, padding, 80,
                          data_format)


if __name__ == "__main__":
  test.main()
