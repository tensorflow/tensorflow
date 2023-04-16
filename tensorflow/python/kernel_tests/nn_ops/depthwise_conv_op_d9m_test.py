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
"""Functional tests for determinsitic depthwise convolutional operations."""

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import depthwise_conv_op_base
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
# The following imports are required to register the gradient functions.
from tensorflow.python.ops.nn_grad import _DepthwiseConv2dNativeBackpropFilterGrad  # pylint: disable=unused-import
from tensorflow.python.ops.nn_grad import _DepthwiseConv2dNativeBackpropInputGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_all_without_tensor_float_32("Uses matmul")
class DepthwiseConv2DDeterministicTest(
    depthwise_conv_op_base.DepthwiseConv2DBase):
  """Test determinism-related functionality of tf.nn.depthwise_conv2d."""

  def _genParams(self,
                 use_cudnn=False,
                 data_format="NHWC",
                 dtype=dtypes.float32,
                 seed=123):
    random_seed.set_seed(seed)
    batch_size = 2  # no interaction over batch, so make small
    if use_cudnn:
      # When op-determinism is not enabled, one input channel, plus a
      # cuDNN-supported filter size and number of output channels will result
      # in cuDNN being used for both backprop-to-input and backprop-to-filter on
      # cuDNN 7.6.3 and higher. When op-determnism is enabled, cuDNN is always
      # used for backprop-to-filter.
      input_channels = 1
    else:
      input_channels = 2  # no interaction over channels, so make small
    input_height = 500
    input_width = 1000
    if data_format == "NHWC":
      input_shape = (batch_size, input_height, input_width, input_channels)
    else:  # "NCHW"
      input_shape = (batch_size, input_channels, input_height, input_width)
    input_data = random_ops.random_normal(input_shape, dtype=dtype)
    # The following filter size results in nondeterminism being exercised in
    # cuDNN backprop (when determinism is not enabled) to both input and filter
    # as well as in the specialized (non-cuDNN) depthwise backprop to filter.
    filter_height = 7
    filter_width = 7
    channel_multiplier = 10
    filter_shape = (filter_height, filter_width, input_channels,
                    channel_multiplier)
    filter_data = random_ops.random_normal(filter_shape, dtype=dtype)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    output_height = input_height  # because same padding
    output_width = input_width  # because same padding
    output_channels = input_channels * channel_multiplier
    if data_format == "NHWC":
      output_shape = (batch_size, output_height, output_width, output_channels)
    else:  # "NCHW"
      output_shape = (batch_size, output_channels, output_height, output_width)
    return input_data, filter_data, strides, padding, output_shape

  def _testForwardDeterminismCase(self,
                                  use_cudnn=False,
                                  data_format="NHWC",
                                  dtype=dtypes.float32):
    for seed in range(5):
      p = self._genParams(use_cudnn, data_format, dtype, seed=seed)
      input_data, filter_data, strides, padding, _ = p

      result_a = nn_impl.depthwise_conv2d_v2(input_data, filter_data, strides,
                                             padding, data_format)
      result_b = nn_impl.depthwise_conv2d_v2(input_data, filter_data, strides,
                                             padding, data_format)

      self.assertAllEqual(result_a, result_b)

  @test_util.run_gpu_only
  def testForwardDeterminismGPU(self):
    if test.is_built_with_rocm():
      gpu_dtypes = [dtypes.float16, dtypes.float32]
    else:
      gpu_dtypes = [dtypes.float16, dtypes.float32, dtypes.float64]
    for use_cudnn in [False, True]:
      for data_format in ["NHWC", "NCHW"]:
        for dtype in gpu_dtypes:
          self._testForwardDeterminismCase(use_cudnn, data_format, dtype=dtype)

  def testForwardDeterminismCPU(self):
    if tf_config.list_physical_devices("GPU"):
      self.skipTest("Test only runs when there is no GPU")
    data_format = "NHWC"  # CPU does not implement NCHW version of op
    for dtype in [dtypes.bfloat16.as_numpy_dtype, dtypes.float32,
                  dtypes.float64]:
      self._testForwardDeterminismCase(data_format=data_format, dtype=dtype)

  def _testBackwardDeterminismCase(self,
                                   using_gpu=False,
                                   use_cudnn=False,
                                   data_format="NHWC",
                                   dtype=dtypes.float32):
    p = self._genParams(use_cudnn, data_format, dtype, seed=123)
    input_data, filter_data, strides, padding, output_shape = p

    def Gradients(upstream_gradients):
      with backprop.GradientTape() as tape:
        tape.watch(input_data)
        tape.watch(filter_data)
        op_output = nn_impl.depthwise_conv2d_v2(input_data, filter_data,
                                                strides, padding, data_format)
        gradient_injector_output = op_output * upstream_gradients
      return tape.gradient(gradient_injector_output, [input_data, filter_data])

    # Test only two seeds, since testing takes a long time
    for seed in (987, 988):
      upstream_gradients = random_ops.random_normal(
          output_shape, dtype=dtype, seed=seed)
      input_gradients_a, filter_gradients_a = Gradients(upstream_gradients)
      input_gradients_b, filter_gradients_b = Gradients(upstream_gradients)
      self.assertAllEqual(input_gradients_a, input_gradients_b)
      self.assertAllEqual(filter_gradients_a, filter_gradients_b)

  @test_util.run_gpu_only
  def testBackwardDeterminismGPU(self):
    using_gpu = True
    if test.is_built_with_rocm():
      gpu_dtypes = [dtypes.float16, dtypes.float32]
    else:
      gpu_dtypes = [dtypes.float16, dtypes.float32, dtypes.float64]
    for use_cudnn in [False, True]:
      for data_format in ["NHWC", "NCHW"]:
        for dtype in gpu_dtypes:
          self._testBackwardDeterminismCase(using_gpu, use_cudnn, data_format,
                                            dtype)

  def testBackwardDeterminismCPU(self):
    if tf_config.list_physical_devices("GPU"):
      self.skipTest("Test only runs when there is no GPU")
    data_format = "NHWC"  # CPU does not implement NCHW version of op
    for dtype in [dtypes.bfloat16.as_numpy_dtype, dtypes.float32,
                  dtypes.float64]:
      self._testBackwardDeterminismCase(data_format=data_format, dtype=dtype)


if __name__ == "__main__":
  # The op-determinism setting can be enabled and disabled on-the-fly.
  # However, if cuDNN convolution is used (as it is for these tests) then its
  # setting at the time will influence which algorithm for a particular layer
  # configuration is cached (independently for XLA and non-XLA operation).
  #
  # The tests in this file must be run under a separate test.main from the
  # tests in depthwise_conv_op_test.py to prevent caching the selection of
  # nondeterminsitic algorithms, which would cause the tests defined in this
  # file to fail.
  #
  # Also because of this caching, the tests defined in depthwise_conv_op_base.py
  # should be run with and without op-determinism enabled in separate files.
  #
  # TODO(duncanriach): Implement cuDNN auto-tuning cache invalidation and
  # and execute when op-determinism setting is changed.
  tf_config.enable_op_determinism()
  test.main()
