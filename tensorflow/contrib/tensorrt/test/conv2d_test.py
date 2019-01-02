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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


def conv2d_layer(inputs, filters, kernel_size, strides=(1, 1), padding='valid',
                 data_format='channels_last', dilation_rate=(1, 1), name=None):
  dtype = inputs.dtype
  c_axis = -1 if data_format == 'channels_last' else 1
  nchan = inputs.shape[c_axis]
  weights_shape = (kernel_size[0], kernel_size[1], nchan, filters)
  weights = constant_op.constant(np.random.randn(*weights_shape), dtype=dtype)
  padding = padding.upper()
  if data_format == 'channels_last':
    strides = [1] + list(strides) + [1]
    dilations = [1] + list(dilation_rate) + [1]
    data_format = 'NHWC'
  else:
    strides = [1, 1] + list(strides)
    dilations = [1, 1] + list(dilation_rate)
    data_format = 'NCHW'
  return gen_nn_ops.conv2d(inputs, weights, strides=strides, padding=padding,
                           dilations=dilations, data_format=data_format)

def div_round_up(n, d):
  return (n - 1) // d + 1

class Conv2DNCHWTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing conversion of Conv2D (data_format=NCHW) in TF-TRT conversion."""
    np.random.seed(1234)
    dtype = dtypes.float32
    input_name = "input"
    n, c, h, w = 13, 3, 7, 11
    num_filters = 5
    input_dims = [n, c, h, w]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        results = []
        for kernel_size in [(3, 3), (3, 2)]:
          for dilation_rate in [(1, 1), (2, 3)]:
            result = conv2d_layer(inp, num_filters, kernel_size,
                                  dilation_rate=dilation_rate, padding='same',
                                  data_format='channels_first')
            results.append(result)
        output = sum(results)
        output = array_ops.identity(output, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(n, num_filters, h, w)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


class Conv2DStridedNCHWTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing conversion of strided Conv2D (data_format=NCHW) in TF-TRT
    conversion."""
    np.random.seed(1234)
    dtype = dtypes.float32
    input_name = "input"
    n, c, h, w = 13, 3, 7, 11
    num_filters = 5
    input_dims = [n, c, h, w]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        output = inp
        output = conv2d_layer(output, num_filters, (3, 2), strides=(2, 2),
                              padding='same', data_format='channels_first')
        h = div_round_up(h, 2)
        w = div_round_up(w, 2)
        output = conv2d_layer(output, num_filters, (3, 3), strides=(2, 2),
                              dilation_rate=(2, 3), padding='same',
                              data_format='channels_first')
        h = div_round_up(h, 2)
        w = div_round_up(w, 2)
        output = array_ops.identity(output, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(n, num_filters, h, w)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


class Conv2DNHWCTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Testing conversion of Conv2D (data_format=NHWC) in TF-TRT conversion."""
    np.random.seed(1234)
    dtype = dtypes.float32
    input_name = "input"
    n, h, w, c = 13, 7, 11, 3
    num_filters = 5
    input_dims = [n, h, w, c]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        results = []
        for kernel_size in [(3, 3), (3, 2)]:
          for dilation_rate in [(1, 1), (2, 3)]:
            result = conv2d_layer(inp, num_filters, kernel_size,
                                  dilation_rate=dilation_rate, padding='same',
                                  data_format='channels_last')
            results.append(result)
        output = sum(results)
        output = array_ops.identity(output, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(n, h, w, num_filters)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]


if __name__ == "__main__":
  test.main()
