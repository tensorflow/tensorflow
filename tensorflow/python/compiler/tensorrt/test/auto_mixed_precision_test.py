# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.compiler.tensorrt.test import \
    tf_trt_integration_test_base as trt_test

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
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


class AutoMixedPrecisionTest(trt_test.TfTrtIntegrationTestBase):
  """Testing TF-TRT conversion with `auto_mixed_precision` grappler pass."""

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
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 7, 11, 3]],
                            [[13, 7, 11, 5]])

  def ExpectedEnginesToBuild(self, run_params):
    """Returns the expected engines to build."""
    expected_engines = {
        "TRTEngineOp_000": [
            "Conv2D_3",
            "Conv2D_2",
            "Conv2D_1",
            "Conv2D",
            "add_1",
            "add_2",
            "add_3",
        ]
    }

    if run_params.precision_mode != "FP32":
      expected_engines["TRTEngineOp_000"].extend([
          "Conv2D_3-1-CastToFp16-AutoMixedPrecision",
          "Conv2D_2-1-CastToFp16-AutoMixedPrecision",
          "Conv2D_1-1-CastToFp16-AutoMixedPrecision",
          "Conv2D-1-CastToFp16-AutoMixedPrecision",
          "Conv2D-0-CastToFp16-AutoMixedPrecision",
          "add_3-1-CastToFp32-AutoMixedPrecision",
          "add_2-1-CastToFp32-AutoMixedPrecision",
          "add_1-1-CastToFp32-AutoMixedPrecision",
          "add-1-CastToFp32-AutoMixedPrecision",
      ])

    return expected_engines


if __name__ == "__main__":
  test.main()
