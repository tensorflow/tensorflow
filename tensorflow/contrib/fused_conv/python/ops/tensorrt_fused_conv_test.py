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
"""Script to test TF-TensorRT conversion of FusedConv2DBiasActivation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.fused_conv import fused_conv2d_bias_activation
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def conv2d_fused_layer(inputs,
                       filters,
                       kernel_size,
                       strides=(1, 1),
                       padding="valid",
                       data_format="NHWC",
                       filter_format="HWIO"):
  dtype = inputs.dtype
  c_axis = -1 if data_format == "NHWC" else 1
  nchan = inputs.shape[c_axis]
  if filter_format == "HWIO":
    weights_shape = (kernel_size[0], kernel_size[1], nchan, filters)
  else:
    assert filter_format == "OIHW"
    weights_shape = (filters, nchan, kernel_size[0], kernel_size[1])
  weights = constant_op.constant(np.random.randn(*weights_shape), dtype=dtype)
  biases_shape = (filters,)
  biases = constant_op.constant(np.random.randn(*biases_shape), dtype=dtype)
  padding = padding.upper()
  if data_format == "NHWC":
    strides = [1] + list(strides) + [1]
  else:
    assert data_format == "NCHW"
    strides = [1, 1] + list(strides)
  return fused_conv2d_bias_activation(
      inputs,
      weights,
      biases,
      strides=strides,
      padding=padding,
      data_format=data_format,
      filter_format=filter_format,
      activation_mode="Relu")


def build_fused_conv_graph(inp,
                           num_filters,
                           data_format,
                           filter_format,
                           kernel_sizes,
                           padding="same"):
  results = []
  for kernel_size in kernel_sizes:
    result = conv2d_fused_layer(inp, num_filters, kernel_size, (1, 1),
                                padding, data_format, filter_format)
    results.append(result)
  output = sum(results)
  return array_ops.identity(output, name="output_0")


class Conv2DFusedHWIOTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of Fused Conv2D+Bias+Activation (filter_format=HWIO)."""

  def GraphFn(self, inp):
    np.random.seed(1234)
    return build_fused_conv_graph(
        inp=inp,
        num_filters=5,
        data_format="NHWC",
        filter_format="HWIO",
        kernel_sizes=[(3, 3), (3, 2)])

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 7, 11, 3]],
                            [[13, 7, 11, 5]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


class Conv2DFusedOIHWTest(trt_test.TfTrtIntegrationTestBase):
  """Testing conversion of Fused Conv2D+Bias+Activation (filter_format=HWIO)."""

  def GraphFn(self, inp):
    np.random.seed(1234)
    return build_fused_conv_graph(
        inp=inp,
        num_filters=5,
        data_format="NCHW",
        filter_format="OIHW",
        kernel_sizes=[(3, 3), (3, 2)])

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 3, 7, 11]],
                            [[13, 5, 7, 11]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


if __name__ == "__main__":
  test.main()
