# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Model script to test TF-TensorRT integration."""

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


def conv2d_layer(inputs,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 data_format="channels_last",
                 dilation_rate=(1, 1)):
  dtype = inputs.dtype
  c_axis = -1 if data_format == "channels_last" else 1
  nchan = inputs.shape[c_axis]
  weights_shape = (kernel_size[0], kernel_size[1], nchan, filters)
  weights = constant_op.constant(np.random.randn(*weights_shape), dtype=dtype)
  padding = padding.upper()
  if data_format == "channels_last":
    strides = [1] + list(strides) + [1]
    dilations = [1] + list(dilation_rate) + [1]
    data_format = "NHWC"
  else:
    strides = [1, 1] + list(strides)
    dilations = [1, 1] + list(dilation_rate)
    data_format = "NCHW"
  return gen_nn_ops.conv2d(
      inputs,
      weights,
      strides=strides,
      padding=padding,
      dilations=dilations,
      data_format=data_format)


def div_round_up(n, d):
  return (n - 1) // d + 1


def build_graph(inp,
                dtype,
                num_filters,
                data_format,
                kernel_sizes,
                dilation_rates,
                padding="same"):
  results = []
  for kernel_size in kernel_sizes:
    for dilation_rate in dilation_rates:
      result = conv2d_layer(inp, num_filters, kernel_size, (1, 1), padding,
                            data_format, dilation_rate)
      results.append(result)
  output = sum(results)
  return array_ops.identity(output, name="output_0")


class Conv2DNCHWTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of Conv2D (data_format=NCHW) in TF-TRT conversion."""

  def GraphFn(self, inp):
    np.random.seed(1234)
    return build_graph(
        inp=inp,
        dtype=dtypes.float32,
        num_filters=5,
        data_format="channels_first",
        kernel_sizes=[(3, 3), (3, 2)],
        dilation_rates=[(1, 1), (2, 3)])

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 3, 7, 11]],
                            [[13, 5, 7, 11]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    if trt_test.IsQuantizationWithCalibration(run_params):
      return 4e-02
    return super(Conv2DNCHWTest, self).ExpectedAbsoluteTolerance(run_params)

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    if trt_test.IsQuantizationWithCalibration(run_params):
      return 4e-02
    return super(Conv2DNCHWTest, self).ExpectedRelativeTolerance(run_params)


class Conv2DNHWCTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of Conv2D (data_format=NCHW) in TF-TRT conversion."""

  def GraphFn(self, inp):
    np.random.seed(1234)
    return build_graph(
        inp=inp,
        dtype=dtypes.float32,
        num_filters=5,
        data_format="channels_last",
        kernel_sizes=[(3, 3), (3, 2)],
        dilation_rates=[(1, 1), (2, 3)])

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 7, 11, 3]],
                            [[13, 7, 11, 5]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


class Conv2DStridedNCHWTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of strided Conv2D (data_format=NCHW)."""

  def GraphFn(self, inp):
    np.random.seed(1234)
    num_filters = 5
    output = inp
    output = conv2d_layer(
        output,
        num_filters, (3, 2),
        strides=(2, 2),
        padding="same",
        data_format="channels_first")
    output = conv2d_layer(
        output,
        num_filters, (3, 3),
        strides=(2, 2),
        dilation_rate=(2, 3),
        padding="same",
        data_format="channels_first")
    return array_ops.identity(output, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 3, 7, 11]],
                            [[13, 5, 2, 3]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


class Conv2DTranposeTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of conv2d_transpose (AKA Conv2DBackpropInput)"""

  def GraphFn(self, inp):
    np.random.seed(1234)
    dtype = inp.dtype
    n, c, h, w = 13, 3, 7, 11
    num_filters = 8
    weights_shape = [2, 2, num_filters, c]
    weights = constant_op.constant(np.random.randn(*weights_shape), dtype=dtype)
    output_shape = constant_op.constant([n, num_filters, h * 2, w * 2],
                                        dtype=dtypes.int32)
    output = nn_ops.conv2d_transpose(
        inp,
        weights,
        output_shape,
        strides=[1, 1, 2, 2],
        padding="SAME",
        data_format="NCHW")
    return array_ops.identity(output, name="output_0")

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 3, 7, 11]],
                            [[13, 8, 14, 22]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


if __name__ == "__main__":
  test.main()
