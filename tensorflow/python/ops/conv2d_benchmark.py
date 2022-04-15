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

import itertools
import time

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "enable_layout_optimizer", False,
    "If true, enables layout optimizer to update input data format for faster "
    "execution of convolution ops.")


def build_graph(device, dtype, data_format, input_shape, filter_shape, strides,
                padding, num_iters, warmup_iters):
  """builds a graph containing a sequence of conv2d operations.

  Args:
    device: String, the device to run on.
    dtype: Data type for the convolution.
    data_format: A string from: "NHWC" or "NCHW". Data format for input and
                 output data.
    input_shape: Shape of the input tensor.
    filter_shape: Shape of the filter tensor.
    strides: A list of ints. 1-D of length 4. The stride of sliding
             window for each dimension of input.
    padding: A string from: "SAME", "VALID". The type of padding
             algorithm to use.
    num_iters: number of iterations to run conv2d.
    warmup_iters: number of iterations for warmup runs.

  Returns:
    An array of tensors to run()
  """
  with ops.device("/%s:0" % device):
    inp = variables.VariableV1(
        random_ops.truncated_normal(input_shape, dtype=dtype))
    filt = variables.VariableV1(
        random_ops.truncated_normal(filter_shape, dtype=dtype))

    outputs = []
    conv2d_op = nn_ops.conv2d(
        inp, filt, strides, padding, data_format=data_format)
    outputs.append(conv2d_op)
    for _ in range(1, num_iters):
      with ops.control_dependencies([conv2d_op]):
        conv2d_op = nn_ops.conv2d(
            inp, filt, strides, padding, data_format=data_format)
        outputs.append(conv2d_op)

    warmup_groups = []
    warmup_conv2d_op = nn_ops.conv2d(
        inp, filt, strides, padding, data_format=data_format)
    warmup_groups.append(warmup_conv2d_op)
    for _ in range(1, warmup_iters):
      with ops.control_dependencies([warmup_conv2d_op]):
        warmup_conv2d_op = nn_ops.conv2d(
            inp, filt, strides, padding, data_format=data_format)
        warmup_groups.append(warmup_conv2d_op)
    return control_flow_ops.group(*warmup_groups), control_flow_ops.group(
        *outputs)


class Conv2DBenchmark(test.Benchmark):
  """Benchmark conv2d!"""

  def _run_graph(self, device, dtype, data_format, input_shape, filter_shape,
                 strides, padding, num_iters, warmup_iters):
    """runs the graph and print its execution time.

    Args:
      device: String, the device to run on.
      dtype: Data type for the convolution.
      data_format: A string from: "NHWC" or "NCHW". Data format for input and
                   output data.
      input_shape: Shape of the input tensor.
      filter_shape: Shape of the filter tensor.
      strides: A list of ints. 1-D of length 4. The stride of sliding
               window for each dimension of input.
      padding: A string from: "SAME", "VALID". The type of padding
               algorithm to use.  num_iters: Number of iterations to run the
                 benchmark.
      num_iters: number of iterations to run conv2d.
      warmup_iters: number of iterations for warmup runs.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      warmup_outputs, outputs = build_graph(device, dtype, data_format,
                                            input_shape, filter_shape, strides,
                                            padding, num_iters, warmup_iters)

      config = config_pb2.ConfigProto()
      config.graph_options.optimizer_options.opt_level = -1
      rewrite_options = config.graph_options.rewrite_options

      # Disable layout optimizer to not change input data_format.
      rewrite_options.layout_optimizer = (
          rewriter_config_pb2.RewriterConfig.ON if FLAGS.enable_layout_optimizer
          else rewriter_config_pb2.RewriterConfig.OFF)
      # Convolution ops are effectively noop in the test graph as we are not
      # fetching the convolution outputs. Disable dependency optimizer to not
      # remove the conv ops.
      rewrite_options.dependency_optimization = (
          rewriter_config_pb2.RewriterConfig.OFF)

      with session_lib.Session(graph=graph, config=config) as session:
        # TODO(hinsu): Use run_op_benchmark method from test.Benchmark to run
        # benchmark along with warmup.
        variables.global_variables_initializer().run()
        # warmup runs
        session.run(warmup_outputs)

        start_time = time.time()
        session.run(outputs)
        duration = (time.time() - start_time) / num_iters
        print("%s %s %s inputshape:%s filtershape:%s strides:%s padding:%s "
              "%d iters: %.8f sec" %
              (device, str(dtype), data_format, str(input_shape).replace(
                  " ", ""), str(filter_shape).replace(" ", ""),
               str(strides).replace(" ", ""), padding, num_iters, duration))

    name_template = (
        "conv2d_{device}_{datatype}_{data_format}_input_shape_{inputshape}_"
        "filter_shape_{filtershape}_strides_{strides}_padding_{padding}")

    self.report_benchmark(
        name=name_template.format(
            device=device,
            datatype=str(dtype),
            data_format=str(data_format),
            inputshape=str(input_shape).replace(" ", ""),
            filtershape=str(filter_shape).replace(" ", ""),
            strides=str(strides).replace(" ", ""),
            padding=padding).replace(" ", ""),
        iters=num_iters,
        wall_time=duration)

    return duration

  def benchmark_conv2d(self):
    print("conv2d benchmark:")

    data_types = [dtypes.float32, dtypes.float16]
    data_formats = ["NHWC", "NCHW"]
    in_channels = list(range(1, 10)) + list(range(10, 20, 2)) + list(
        range(20, 33, 4))
    out_channels = [4, 16, 32]
    hw_strides = [[2, 2]]
    paddings = ["VALID", "SAME"]

    args_lists = [
        data_types, data_formats, in_channels, out_channels, hw_strides,
        paddings
    ]
    for args in itertools.product(*args_lists):
      dtype, data_format, in_channel, out_channel, hw_stride, padding = args

      # Keep batch size same as out channels just to reduce the number of
      # different configurations to benchmark.
      batch_size = out_channel
      h, w, fh, fw = 500, 500, 3, 3
      if data_format == "NHWC":
        ishape = [batch_size, h, w, in_channel]
        stride = [1] + hw_stride + [1]
      elif data_format == "NCHW":
        ishape = [batch_size, in_channel, h, w]
        stride = [1, 1] + hw_stride
      else:
        raise ValueError("Unknown data_format: " + str(data_format))
      fshape = [fh, fw, in_channel, out_channel]
      num_iters = 80
      warmup_iters = 2
      self._run_graph("gpu", dtype, data_format, ishape, fshape, stride,
                      padding, num_iters, warmup_iters)


if __name__ == "__main__":
  test.main()
